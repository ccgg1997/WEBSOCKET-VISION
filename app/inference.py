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

from app.config import ModelDefinition, ServiceSettings


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


def ensure_model_file(settings: ServiceSettings, definition: ModelDefinition) -> Path:
    model_path = definition.resolve_path(settings.service_root)
    if model_path.exists():
        return model_path

    if not definition.url:
        raise FileNotFoundError(
            f"No se encontro el modelo '{definition.model_id}' en {model_path}. "
            "Define YOLO_WS_MODELS_FILE, YOLO_WS_MODEL_PATH o usa YOLO_WS_MODEL_IDS con "
            "YOLO_WS_MODEL_<ID>_PATH/URL."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    LOGGER.info("Downloading model '%s' from %s", definition.model_id, definition.url)
    try:
        with urllib.request.urlopen(
            definition.url, timeout=settings.model_download_timeout
        ) as response, temp_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)

        if definition.sha256:
            _validate_sha256(temp_path, definition.sha256)

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
    def __init__(self, settings: ServiceSettings, definition: ModelDefinition) -> None:
        self.settings = settings
        self.model_id = definition.model_id
        self.model_path = ensure_model_file(settings, definition)
        self.device = self._resolve_device(settings.device)
        self.model = YOLO(str(self.model_path)).to(self.device)
        self.model_name = definition.display_name or self.model_path.name
        self.model_file_name = self.model_path.name
        self.selector = definition.selector_raw.strip()
        self.selection_aliases = tuple(alias for alias in definition.selection_aliases if alias)
        self.use_half = self.device.startswith("cuda")
        self._predict_lock = threading.Lock()

        if self.device == "cpu":
            torch.set_num_threads(settings.torch_threads)
        elif self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass

        LOGGER.info(
            "Model loaded: id=%s name=%s path=%s device=%s",
            self.model_id,
            self.model_name,
            self.model_path,
            self.device,
        )
        self._warmup()

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.model_name,
            "file_name": self.model_file_name,
            "device": self.device,
            "half": self.use_half,
            "selector": self.selector,
            "selection_aliases": list(self.selection_aliases),
        }

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
            with torch.inference_mode():
                results = self.model.predict(
                    source=frame,
                    conf=self.settings.conf_threshold,
                    iou=self.settings.iou_threshold,
                    imgsz=self.settings.imgsz,
                    max_det=self.settings.max_det,
                    device=self.device,
                    half=self.use_half,
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
            "model": self.describe(),
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

    def _warmup(self) -> None:
        warmup_frame = np.zeros(
            (self.settings.imgsz, self.settings.imgsz, 3),
            dtype=np.uint8,
        )
        try:
            with self._predict_lock:
                with torch.inference_mode():
                    self.model.predict(
                        source=warmup_frame,
                        conf=self.settings.conf_threshold,
                        iou=self.settings.iou_threshold,
                        imgsz=self.settings.imgsz,
                        max_det=self.settings.max_det,
                        device=self.device,
                        half=self.use_half,
                        verbose=False,
                    )
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            LOGGER.info(
                "Model warmup complete: id=%s device=%s half=%s",
                self.model_id,
                self.device,
                self.use_half,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Model warmup failed for %s: %s", self.model_id, exc)


class ModelRegistry:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.default_model_id = settings.resolve_default_model_id()
        self._selection_values = settings.resolve_model_selection_values()
        self._selectors = {
            model_id: settings.resolve_model_selector(model_id)
            for model_id in settings.resolve_model_definitions()
        }
        self._runtimes: dict[str, ModelRuntime] = {}

        for model_id, definition in settings.resolve_model_definitions().items():
            runtime = ModelRuntime(settings, definition)
            runtime.selector = self._selectors.get(model_id, runtime.selector)
            runtime.selection_aliases = self._selection_values.get(
                model_id,
                runtime.selection_aliases,
            )
            self._runtimes[model_id] = runtime

    def get(self, model_id: str | None = None) -> ModelRuntime:
        selected_id = model_id or self.default_model_id
        runtime = self._runtimes.get(selected_id)
        if runtime is None:
            raise KeyError(selected_id)
        return runtime

    def list_models(self) -> list[dict[str, Any]]:
        return [runtime.describe() for runtime in self._runtimes.values()]
