from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class SourceModel(Base):
    __tablename__ = "sources"

    source_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    last_seen_at: Mapped[str | None] = mapped_column(String(64))


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("sources.source_id"), nullable=False
    )
    auth_username: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_addr: Mapped[str | None] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_device: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[str] = mapped_column(String(64), nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    snapshot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_inference_at: Mapped[str | None] = mapped_column(String(64))


class StateSnapshotModel(Base):
    __tablename__ = "state_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("sources.source_id"), nullable=False
    )
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[str] = mapped_column(String(64), nullable=False)
    detections_total: Mapped[int] = mapped_column(Integer, nullable=False)
    dominant_label: Mapped[str | None] = mapped_column(String(255))
    dominant_percent: Mapped[int] = mapped_column(Integer, nullable=False)
    best_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    counts_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    detections_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    event_state: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class DetectionEventModel(Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("sources.source_id"), nullable=False
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("state_snapshots.id")
    )
    trigger_label: Mapped[str] = mapped_column(String(255), nullable=False)
    trigger_percent: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[str] = mapped_column(String(64), nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String(64))
    storage_backend: Mapped[str] = mapped_column(String(50), nullable=False)
    clip_bucket: Mapped[str | None] = mapped_column(String(255))
    clip_object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    clip_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    preview_bucket: Mapped[str | None] = mapped_column(String(255))
    preview_object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    preview_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    annotation: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    detections_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str


class ServiceRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._engine = create_engine(settings.database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
        )
        Base.metadata.create_all(self._engine)

    def _session(self) -> Session:
        return self._session_factory()

    @staticmethod
    def _source_to_dict(source: SourceModel) -> dict[str, Any]:
        return {
            "source_id": source.source_id,
            "source_type": source.source_type,
            "display_name": source.display_name,
            "status": source.status,
            "metadata": source.metadata_json,
            "created_at": source.created_at,
            "updated_at": source.updated_at,
            "last_seen_at": source.last_seen_at,
        }

    @staticmethod
    def _session_to_dict(session_model: SessionModel) -> dict[str, Any]:
        return {
            "id": session_model.id,
            "source_id": session_model.source_id,
            "auth_username": session_model.auth_username,
            "remote_addr": session_model.remote_addr,
            "model_name": session_model.model_name,
            "model_device": session_model.model_device,
            "started_at": session_model.started_at,
            "ended_at": session_model.ended_at,
            "status": session_model.status,
            "frame_count": session_model.frame_count,
            "snapshot_count": session_model.snapshot_count,
            "event_count": session_model.event_count,
            "avg_latency_ms": session_model.avg_latency_ms,
            "last_inference_at": session_model.last_inference_at,
        }

    @staticmethod
    def _snapshot_to_dict(snapshot: StateSnapshotModel) -> dict[str, Any]:
        return {
            "id": snapshot.id,
            "session_id": snapshot.session_id,
            "source_id": snapshot.source_id,
            "frame_id": snapshot.frame_id,
            "captured_at": snapshot.captured_at,
            "detections_total": snapshot.detections_total,
            "dominant_label": snapshot.dominant_label,
            "dominant_percent": snapshot.dominant_percent,
            "best_confidence": snapshot.best_confidence,
            "latency_ms": snapshot.latency_ms,
            "counts": snapshot.counts_json,
            "detections": snapshot.detections_json,
            "event_state": snapshot.event_state,
            "created_at": snapshot.created_at,
        }

    @staticmethod
    def _event_to_dict(event: DetectionEventModel) -> dict[str, Any]:
        return {
            "id": event.id,
            "session_id": event.session_id,
            "source_id": event.source_id,
            "snapshot_id": event.snapshot_id,
            "trigger_label": event.trigger_label,
            "trigger_percent": event.trigger_percent,
            "confidence": event.confidence,
            "detected_at": event.detected_at,
            "ended_at": event.ended_at,
            "storage_backend": event.storage_backend,
            "clip_bucket": event.clip_bucket,
            "clip_object_key": event.clip_object_key,
            "clip_size_bytes": event.clip_size_bytes,
            "preview_bucket": event.preview_bucket,
            "preview_object_key": event.preview_object_key,
            "preview_size_bytes": event.preview_size_bytes,
            "annotation": event.annotation,
            "status": event.status,
            "detections": event.detections_json,
            "created_at": event.created_at,
        }

    def upsert_source(
        self,
        source_id: str,
        source_type: str,
        display_name: str,
        status: str,
        metadata: dict[str, Any],
        created_at: str,
        updated_at: str,
        last_seen_at: str | None,
    ) -> None:
        with self._session() as session:
            source = session.get(SourceModel, source_id)
            if source is None:
                source = SourceModel(
                    source_id=source_id,
                    source_type=source_type,
                    display_name=display_name,
                    status=status,
                    metadata_json=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                    last_seen_at=last_seen_at,
                )
                session.add(source)
            else:
                source.source_type = source_type
                source.display_name = display_name
                source.status = status
                source.metadata_json = metadata
                source.updated_at = updated_at
                source.last_seen_at = last_seen_at
            session.commit()

    def create_session(self, record: dict[str, Any]) -> None:
        with self._session() as session:
            session.add(SessionModel(**record))
            session.commit()

    def close_session(self, session_id: str, updates: dict[str, Any]) -> None:
        with self._session() as session:
            session_model = session.get(SessionModel, session_id)
            if session_model is None:
                return
            for key, value in updates.items():
                setattr(session_model, key, value)
            session.commit()

    def insert_snapshot(self, record: dict[str, Any]) -> None:
        payload = {
            **record,
            "counts_json": record["counts"],
            "detections_json": record["detections"],
            "created_at": record.get("created_at", record["captured_at"]),
        }
        payload.pop("counts", None)
        payload.pop("detections", None)
        with self._session() as session:
            session.add(StateSnapshotModel(**payload))
            session.commit()

    def insert_event(self, record: dict[str, Any]) -> None:
        payload = {
            **record,
            "detections_json": record["detections"],
        }
        payload.pop("detections", None)
        with self._session() as session:
            session.add(DetectionEventModel(**payload))
            session.commit()

    def list_sources(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.scalars(
                select(SourceModel).order_by(SourceModel.updated_at.desc()).limit(limit)
            ).all()
        return [self._source_to_dict(row) for row in rows]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.scalars(
                select(SessionModel)
                .order_by(SessionModel.started_at.desc())
                .limit(limit)
            ).all()
        return [self._session_to_dict(row) for row in rows]

    def list_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.scalars(
                select(StateSnapshotModel)
                .order_by(StateSnapshotModel.captured_at.desc())
                .limit(limit)
            ).all()
        return [self._snapshot_to_dict(row) for row in rows]

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.scalars(
                select(DetectionEventModel)
                .order_by(DetectionEventModel.detected_at.desc())
                .limit(limit)
            ).all()
        return [self._event_to_dict(row) for row in rows]

    def get_stats(self) -> dict[str, int]:
        with self._session() as session:
            sources = session.query(SourceModel).count()
            sessions = session.query(SessionModel).count()
            snapshots = session.query(StateSnapshotModel).count()
            events = session.query(DetectionEventModel).count()
        return {
            "sources": sources,
            "sessions": sessions,
            "snapshots": snapshots,
            "events": events,
        }
