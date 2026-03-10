from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shutil
import threading
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import ServiceSettings


LOGGER = logging.getLogger(__name__)

PALETTE = [
    (0, 255, 0),
    (0, 170, 255),
    (255, 85, 0),
    (255, 0, 170),
    (0, 255, 255),
    (170, 0, 255),
]


def _validate_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    received = digest.hexdigest()
    if received.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 invalido para el modelo descargado. esperado={expected_sha256} recibido={received}"
        )


def ensure_model_file(settings: ServiceSettings) -> Path:
    model_path = settings.resolve_model_path()
    if model_path.exists():
        return model_path

    if not settings.model_url:
        raise FileNotFoundError(
            f"No se encontro el modelo en {model_path}. "
            "Define YOLO_WS_MODEL_PATH o YOLO_WS_MODEL_URL."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    LOGGER.info("Downloading model from %s", settings.model_url)
    try:
        with urllib.request.urlopen(
            settings.model_url, timeout=settings.model_download_timeout
        ) as response, temp_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)

        if settings.model_sha256:
            _validate_sha256(temp_path, settings.model_sha256)

        temp_path.replace(model_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return model_path


def extract_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names if isinstance(result.names, dict) else {}
    xyxy = result.boxes.xyxy.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()

    for box, conf, class_id in zip(xyxy, confs, class_ids):
        detections.append(
            {
                "class_id": int(class_id),
                "label": str(names.get(class_id, class_id)),
                "confidence": round(float(conf), 4),
                "xyxy": [round(float(value), 2) for value in box],
            }
        )

    return detections


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width - 1, int(x2)), min(height - 1, int(y2))

        color = PALETTE[detection["class_id"] % len(PALETTE)]
        caption = f"{detection['label']} {detection['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return frame


class ModelRuntime:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.model_path = ensure_model_file(settings)
        self.device = self._resolve_device(settings.device)
        self.model = YOLO(str(self.model_path)).to(self.device)
        self.model_name = self.model_path.name
        self._predict_lock = threading.Lock()

        if self.device == "cpu":
            torch.set_num_threads(settings.torch_threads)

        LOGGER.info("Model loaded: path=%s device=%s", self.model_path, self.device)

    @staticmethod
    def _resolve_device(device_preference: str) -> str:
        normalized = device_preference.lower().strip()
        if normalized in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("YOLO_WS_DEVICE=cuda pero CUDA no esta disponible.")
        return normalized

    async def infer_bytes(
        self,
        image_bytes: bytes,
        frame_id: str,
        return_image: bool,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._infer_sync,
            image_bytes,
            frame_id,
            return_image,
        )

    def _infer_sync(
        self,
        image_bytes: bytes,
        frame_id: str,
        return_image: bool,
    ) -> dict[str, Any]:
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar la imagen enviada.")

        start = perf_counter()
        with self._predict_lock:
            results = self.model.predict(
                source=frame,
                conf=self.settings.conf_threshold,
                iou=self.settings.iou_threshold,
                imgsz=self.settings.imgsz,
                max_det=self.settings.max_det,
                device=self.device,
                verbose=False,
            )
        latency_ms = round((perf_counter() - start) * 1000, 2)

        result = results[0]
        detections = extract_detections(result)
        counts = Counter(detection["label"] for detection in detections)

        payload: dict[str, Any] = {
            "type": "inference",
            "frame_id": frame_id,
            "latency_ms": latency_ms,
            "image": {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },
            "counts": {
                "total": len(detections),
                "by_label": dict(counts),
            },
            "detections": detections,
            "model": {
                "name": self.model_name,
                "device": self.device,
            },
        }

        if return_image:
            annotated = draw_detections(frame.copy(), detections)
            ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality],
            )
            if not ok:
                raise ValueError("No se pudo codificar la imagen anotada.")
            payload["annotated_image_b64"] = base64.b64encode(encoded.tobytes()).decode(
                "ascii"
            )

        return payload

