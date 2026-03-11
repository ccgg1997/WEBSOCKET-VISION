from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from app.config import ServiceSettings
from app.database import DatabaseSettings, ServiceRepository
from app.inference import draw_detections
from app.notifications import TelegramNotifier
from app.storage import ObjectStorage, create_storage


LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def label_to_percent(label: str, aliases: dict[str, int] | None = None) -> int:
    normalized = str(label).strip().replace("%", "")
    if aliases:
        alias_value = aliases.get(normalized)
        if alias_value is not None:
            return alias_value
    if normalized.isdigit():
        return int(normalized)
    if normalized.lower() == "vacio":
        return 0
    return 0


def sanitize_label_for_filename(label: str) -> str:
    normalized = str(label).strip().lower()
    normalized = normalized.replace("%", "pct")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "unknown"


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")
    return frame


@dataclass
class ActiveEvent:
    event_id: str
    session_id: str
    source_id: str
    snapshot_id: str | None
    trigger_label: str
    trigger_percent: int
    confidence: float
    detected_at: str
    file_stem: str
    clip_temp_path: Path
    preview_temp_path: Path
    annotation: str
    detections: list[dict[str, Any]]
    end_time: float
    writer: cv2.VideoWriter


class SessionEventRecorder:
    def __init__(
        self,
        settings: ServiceSettings,
        repository: ServiceRepository,
        storage: ObjectStorage,
        temp_dir: Path,
        session_id: str,
        connection_id: str,
        source_id: str,
        source_type: str,
        source_name: str,
        source_metadata: dict[str, Any],
        model_id: str,
        telegram_notifier: TelegramNotifier | None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._storage = storage
        self._temp_dir = temp_dir
        self._session_id = session_id
        self._connection_id = connection_id
        self._source_id = source_id
        self._source_type = source_type
        self._source_name = source_name
        self._source_metadata = source_metadata
        self._model_id = model_id
        self._telegram_notifier = telegram_notifier
        self._frame_times: deque[float] = deque(maxlen=30)
        self._active_event: ActiveEvent | None = None
        self._last_event_at = 0.0
        self._last_snapshot_at = 0.0
        self._frame_count = 0
        self._snapshot_count = 0
        self._event_count = 0
        self._latency_total_ms = 0.0
        self._last_inference_at: str | None = None
        self._telegram_alert_active = False

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_metadata(self) -> dict[str, Any]:
        return self._source_metadata

    def summary(self) -> dict[str, Any]:
        avg_latency_ms = (
            self._latency_total_ms / self._frame_count if self._frame_count else 0.0
        )
        return {
            "frame_count": self._frame_count,
            "snapshot_count": self._snapshot_count,
            "event_count": self._event_count,
            "avg_latency_ms": avg_latency_ms,
            "last_inference_at": self._last_inference_at,
        }

    def _estimate_fps(self) -> float:
        if len(self._frame_times) < 2:
            return self._settings.event_record_fps

        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return self._settings.event_record_fps

        estimated = (len(self._frame_times) - 1) / span
        return max(1.0, min(estimated, 30.0))

    def _best_detection(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not detections:
            return None
        return max(detections, key=lambda detection: float(detection.get("confidence", 0.0)))

    def _resolve_trigger(self, detections: list[dict[str, Any]]) -> dict[str, Any] | None:
        matches = [
            detection
            for detection in detections
            if str(detection.get("label", "")).strip() in self._settings.trigger_labels
        ]
        if not matches:
            return None
        return max(matches, key=lambda detection: float(detection.get("confidence", 0.0)))

    def _resolve_fill_level(
        self,
        detections: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, int]:
        best_detection: dict[str, Any] | None = None
        best_percent = 0
        for detection in detections:
            percent = label_to_percent(
                str(detection.get("label", "")),
                self._settings.label_percent_aliases,
            )
            if percent > best_percent:
                best_detection = detection
                best_percent = percent
        return best_detection, best_percent

    def _maybe_send_telegram_fill_alert(
        self,
        detections: list[dict[str, Any]],
    ) -> None:
        if self._telegram_notifier is None:
            return
        if self._model_id not in self._settings.telegram_model_ids:
            return

        best_detection, fill_percent = self._resolve_fill_level(detections)
        if best_detection is None or fill_percent <= self._settings.telegram_fill_threshold:
            self._telegram_alert_active = False
            return

        if self._telegram_alert_active:
            return

        self._telegram_alert_active = True
        label = str(best_detection.get("label", ""))
        confidence = round(float(best_detection.get("confidence", 0.0)) * 100, 1)
        source_name = self._source_name.strip() or self._source_id
        source_line = source_name
        if source_name != self._source_id:
            source_line = f"{source_name} ({self._source_id})"

        message = "\n".join(
            [
                "ALERTA DE LLENADO",
                f"camara: {source_line}",
                f"conexion: {self._connection_id}",
                f"modelo: {self._model_id}",
                f"llenado: {fill_percent}%",
                f"label: {label}",
                f"confianza: {confidence}%",
            ]
        )
        self._telegram_notifier.notify_text(message)

    def _build_object_key(self, event_id: str, filename: str) -> str:
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        base_prefix = self._settings.minio_prefix.strip("/").replace("\\", "/")
        source_prefix = self._source_id.replace("\\", "-").replace("/", "-")
        return f"{base_prefix}/{date_prefix}/{source_prefix}/{event_id}/{filename}"

    def _store_snapshot(
        self,
        frame_id: str,
        detections: list[dict[str, Any]],
        counts: dict[str, int],
        latency_ms: float,
        timestamp: float,
        event_state: str | None = None,
    ) -> dict[str, Any]:
        snapshot_id = uuid4().hex
        captured_at = utc_now_iso()
        dominant = self._best_detection(detections)

        record = {
            "id": snapshot_id,
            "session_id": self._session_id,
            "source_id": self._source_id,
            "frame_id": frame_id,
            "captured_at": captured_at,
            "detections_total": len(detections),
            "dominant_label": str(dominant["label"]) if dominant else None,
            "dominant_percent": label_to_percent(
                dominant["label"], self._settings.label_percent_aliases
            )
            if dominant
            else 0,
            "best_confidence": round(float(dominant.get("confidence", 0.0)), 4)
            if dominant
            else 0.0,
            "latency_ms": round(float(latency_ms), 2),
            "counts": counts,
            "detections": detections,
            "event_state": event_state,
        }
        self._repository.insert_snapshot(record)
        self._repository.upsert_source(
            source_id=self._source_id,
            source_type=self._source_type,
            display_name=self._source_name,
            status="online",
            metadata=self._source_metadata,
            created_at=captured_at,
            updated_at=captured_at,
            last_seen_at=captured_at,
        )
        self._snapshot_count += 1
        self._last_snapshot_at = timestamp

        return {
            "id": snapshot_id,
            "captured_at": captured_at,
            "detections_total": record["detections_total"],
            "dominant_label": record["dominant_label"],
            "dominant_percent": record["dominant_percent"],
            "latency_ms": record["latency_ms"],
            "event_state": event_state,
        }

    def _maybe_store_periodic_snapshot(
        self,
        frame_id: str,
        detections: list[dict[str, Any]],
        counts: dict[str, int],
        latency_ms: float,
        timestamp: float,
    ) -> dict[str, Any] | None:
        if timestamp - self._last_snapshot_at < self._settings.snapshot_interval_seconds:
            return None
        if not self._settings.snapshot_save_empty and not detections:
            return None

        return self._store_snapshot(
            frame_id=frame_id,
            detections=detections,
            counts=counts,
            latency_ms=latency_ms,
            timestamp=timestamp,
            event_state=None,
        )

    def _finalize_event(self, timestamp: float | None = None) -> dict[str, Any] | None:
        if self._active_event is None:
            return None

        active = self._active_event
        active.writer.release()
        ended_at = utc_now_iso()
        preview_filename = f"{active.file_stem}_snapshot.jpg"
        clip_filename = f"{active.file_stem}_clip.mp4"

        preview_object = self._storage.upload_file(
            local_path=active.preview_temp_path,
            object_key=self._build_object_key(active.event_id, preview_filename),
            content_type="image/jpeg",
        )
        clip_object = self._storage.upload_file(
            local_path=active.clip_temp_path,
            object_key=self._build_object_key(active.event_id, clip_filename),
            content_type="video/mp4",
        )

        active.preview_temp_path.unlink(missing_ok=True)
        active.clip_temp_path.unlink(missing_ok=True)

        record = {
            "id": active.event_id,
            "session_id": active.session_id,
            "source_id": active.source_id,
            "snapshot_id": active.snapshot_id,
            "trigger_label": active.trigger_label,
            "trigger_percent": active.trigger_percent,
            "confidence": active.confidence,
            "detected_at": active.detected_at,
            "ended_at": ended_at,
            "storage_backend": self._storage.backend_name,
            "clip_bucket": clip_object.bucket,
            "clip_object_key": clip_object.object_key,
            "clip_size_bytes": clip_object.size_bytes,
            "preview_bucket": preview_object.bucket,
            "preview_object_key": preview_object.object_key,
            "preview_size_bytes": preview_object.size_bytes,
            "annotation": active.annotation,
            "status": "saved",
            "detections": active.detections,
            "created_at": active.detected_at,
        }
        self._repository.insert_event(record)

        self._active_event = None
        self._last_event_at = timestamp if timestamp is not None else time.time()
        self._event_count += 1

        LOGGER.info(
            "Event saved: id=%s source=%s clip=%s",
            record["id"],
            record["source_id"],
            record["clip_object_key"],
        )
        return {
            "state": "saved",
            "event_id": record["id"],
            "snapshot_id": record["snapshot_id"],
            "storage_backend": record["storage_backend"],
            "clip_bucket": record["clip_bucket"],
            "clip_object_key": record["clip_object_key"],
            "clip_url": self._storage.build_access_url(
                record["clip_bucket"], record["clip_object_key"]
            ),
            "clip_filename": clip_filename,
            "preview_bucket": record["preview_bucket"],
            "preview_object_key": record["preview_object_key"],
            "preview_url": self._storage.build_access_url(
                record["preview_bucket"], record["preview_object_key"]
            ),
            "preview_filename": preview_filename,
            "annotation": record["annotation"],
        }

    def _start_event(
        self,
        timestamp: float,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        trigger_detection: dict[str, Any],
        snapshot_id: str | None,
    ) -> dict[str, Any]:
        event_id = uuid4().hex
        temp_event_dir = self._temp_dir / datetime.now().strftime("%Y%m%d")
        temp_event_dir.mkdir(parents=True, exist_ok=True)

        clip_temp_path = temp_event_dir / f"{event_id}.mp4"
        preview_temp_path = temp_event_dir / f"{event_id}.jpg"

        annotated = draw_detections(frame.copy(), detections)
        ok = cv2.imwrite(str(preview_temp_path), annotated)
        if not ok:
            raise ValueError("No se pudo guardar la imagen anotada del evento.")

        fps = self._estimate_fps()
        height, width = annotated.shape[:2]
        writer = cv2.VideoWriter(
            str(clip_temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("No se pudo abrir el archivo de video del evento.")

        writer.write(annotated)

        trigger_label = str(trigger_detection.get("label", ""))
        trigger_percent = label_to_percent(
            trigger_label, self._settings.label_percent_aliases
        )
        confidence = round(float(trigger_detection.get("confidence", 0.0)), 4)
        detected_clock = datetime.now()
        detected_at = utc_now_iso()
        file_stem = (
            f"{detected_clock.strftime('%H%M%S')}_{sanitize_label_for_filename(trigger_label)}"
        )
        if trigger_percent > 0:
            annotation = (
                f"Trigger {trigger_label} ({trigger_percent}%) detectado en fuente "
                f"{self._source_id}. Se guardaron {self._settings.clip_seconds} segundos "
                "de evidencia."
            )
        else:
            annotation = (
                f"Trigger {trigger_label} detectado en fuente {self._source_id}. "
                f"Se guardaron {self._settings.clip_seconds} segundos de evidencia."
            )

        self._active_event = ActiveEvent(
            event_id=event_id,
            session_id=self._session_id,
            source_id=self._source_id,
            snapshot_id=snapshot_id,
            trigger_label=trigger_label,
            trigger_percent=trigger_percent,
            confidence=confidence,
            detected_at=detected_at,
            file_stem=file_stem,
            clip_temp_path=clip_temp_path,
            preview_temp_path=preview_temp_path,
            annotation=annotation,
            detections=detections,
            end_time=timestamp + self._settings.clip_seconds,
            writer=writer,
        )

        LOGGER.info(
            "Event started: id=%s source=%s trigger=%s",
            event_id,
            self._source_id,
            trigger_label,
        )
        return {
            "state": "recording",
            "event_id": event_id,
            "snapshot_id": snapshot_id,
            "clip_seconds": self._settings.clip_seconds,
            "trigger_label": trigger_label,
            "trigger_percent": trigger_percent,
            "annotation": annotation,
        }

    def process_frame(
        self,
        frame_id: str,
        image_bytes: bytes,
        detections: list[dict[str, Any]],
        counts: dict[str, int],
        latency_ms: float,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        timestamp = timestamp or time.time()
        self._frame_times.append(timestamp)
        self._frame_count += 1
        self._latency_total_ms += float(latency_ms)
        self._last_inference_at = utc_now_iso()
        self._maybe_send_telegram_fill_alert(detections)

        result: dict[str, Any] = {}
        snapshot = self._maybe_store_periodic_snapshot(
            frame_id=frame_id,
            detections=detections,
            counts=counts,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )
        if snapshot is not None:
            result["snapshot"] = snapshot

        if not self._settings.events_enabled:
            return result or None

        if self._active_event is not None:
            frame = decode_image_bytes(image_bytes)
            annotated = draw_detections(frame.copy(), detections)
            self._active_event.writer.write(annotated)
            if timestamp >= self._active_event.end_time:
                event = self._finalize_event(timestamp=timestamp)
                if event is not None:
                    result["event"] = event
            else:
                result["event"] = {
                    "state": "recording",
                    "event_id": self._active_event.event_id,
                    "snapshot_id": self._active_event.snapshot_id,
                    "clip_seconds": self._settings.clip_seconds,
                    "trigger_label": self._active_event.trigger_label,
                }
            return result or None

        if timestamp - self._last_event_at < self._settings.event_cooldown_seconds:
            return result or None

        trigger_detection = self._resolve_trigger(detections)
        if trigger_detection is None:
            return result or None

        if snapshot is None:
            snapshot = self._store_snapshot(
                frame_id=frame_id,
                detections=detections,
                counts=counts,
                latency_ms=latency_ms,
                timestamp=timestamp,
                event_state="trigger",
            )
            result["snapshot"] = snapshot

        frame = decode_image_bytes(image_bytes)
        event = self._start_event(
            timestamp=timestamp,
            frame=frame,
            detections=detections,
            trigger_detection=trigger_detection,
            snapshot_id=snapshot.get("id"),
        )
        result["event"] = event
        return result or None

    def close(self) -> dict[str, Any] | None:
        return self._finalize_event()


class EventManager:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._repository = ServiceRepository(
            DatabaseSettings(database_url=settings.resolve_database_url())
        )
        self._storage = create_storage(settings)
        self._temp_dir = settings.resolve_temp_dir()
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._telegram_notifier: TelegramNotifier | None = None
        if settings.telegram_enabled:
            self._telegram_notifier = TelegramNotifier(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
                timeout_seconds=settings.telegram_timeout_seconds,
            )
        self._recorders: dict[str, SessionEventRecorder] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        session_id: str,
        connection_id: str,
        source_id: str,
        source_type: str,
        source_name: str,
        source_metadata: dict[str, Any],
        auth_username: str,
        remote_addr: str,
        model_id: str,
        model_name: str,
        model_device: str,
    ) -> None:
        started_at = utc_now_iso()
        self._repository.upsert_source(
            source_id=source_id,
            source_type=source_type,
            display_name=source_name,
            status="online",
            metadata=source_metadata,
            created_at=started_at,
            updated_at=started_at,
            last_seen_at=started_at,
        )
        self._repository.create_session(
            {
                "id": session_id,
                "source_id": source_id,
                "auth_username": auth_username,
                "remote_addr": remote_addr,
                "model_name": model_name,
                "model_device": model_device,
                "started_at": started_at,
                "ended_at": None,
                "status": "open",
                "frame_count": 0,
                "snapshot_count": 0,
                "event_count": 0,
                "avg_latency_ms": 0.0,
                "last_inference_at": None,
            }
        )

        with self._lock:
            self._recorders[session_id] = SessionEventRecorder(
                settings=self._settings,
                repository=self._repository,
                storage=self._storage,
                temp_dir=self._temp_dir,
                session_id=session_id,
                connection_id=connection_id,
                source_id=source_id,
                source_type=source_type,
                source_name=source_name,
                source_metadata=source_metadata,
                model_id=model_id,
                telegram_notifier=self._telegram_notifier,
            )

    def process_frame(
        self,
        session_id: str,
        frame_id: str,
        image_bytes: bytes,
        detections: list[dict[str, Any]],
        counts: dict[str, int],
        latency_ms: float,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        recorder = self._recorders.get(session_id)
        if recorder is None:
            return None
        return recorder.process_frame(
            frame_id=frame_id,
            image_bytes=image_bytes,
            detections=detections,
            counts=counts,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )

    def close_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            recorder = self._recorders.pop(session_id, None)
        if recorder is None:
            return None

        event_update = recorder.close()
        summary = recorder.summary()
        ended_at = utc_now_iso()

        self._repository.close_session(
            session_id=session_id,
            updates={
                "ended_at": ended_at,
                "status": "closed",
                "frame_count": summary["frame_count"],
                "snapshot_count": summary["snapshot_count"],
                "event_count": summary["event_count"],
                "avg_latency_ms": round(summary["avg_latency_ms"], 2),
                "last_inference_at": summary["last_inference_at"],
            },
        )
        updated_at = ended_at
        self._repository.upsert_source(
            source_id=recorder.source_id,
            source_type=recorder.source_type,
            display_name=recorder.source_name,
            status="offline",
            metadata=recorder.source_metadata,
            created_at=updated_at,
            updated_at=updated_at,
            last_seen_at=ended_at,
        )
        return event_update

    def list_sources(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._repository.list_sources(limit=limit)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._repository.list_sessions(limit=limit)

    def list_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._repository.list_snapshots(limit=limit)

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        events = self._repository.list_events(limit=limit)
        for event in events:
            event["clip_url"] = self._storage.build_access_url(
                event["clip_bucket"], event["clip_object_key"]
            )
            event["preview_url"] = self._storage.build_access_url(
                event["preview_bucket"], event["preview_object_key"]
            )
        return events

    def get_stats(self) -> dict[str, int]:
        return self._repository.get_stats()

    def shutdown(self) -> None:
        if self._telegram_notifier is not None:
            self._telegram_notifier.stop()
