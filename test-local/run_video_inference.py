from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_FILE = REPO_ROOT / "config" / "models.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ENV_FILE = Path(__file__).with_name(".env")
WINDOW_NAME = "local-yolo-video-test"
TEXT_COLOR = (0, 255, 0)
TEXT_SHADOW = (0, 0, 0)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    name: str
    selector: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba local de video para comparar YOLO en CPU vs GPU."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.getenv("LOCAL_TEST_VIDEO_PATH", ""),
        help="Ruta al video a evaluar. Si no se envia, se pedira por consola.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=os.getenv("LOCAL_TEST_VIDEO_DIR", ""),
        help="Carpeta donde buscar videos si no se envia --video.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LOCAL_TEST_MODEL", ""),
        help="Modelo a usar por id o selector. Si no se envia, se pedira por consola.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=os.getenv("LOCAL_TEST_DEVICE", "auto"),
        help="Dispositivo preferido. 'auto' pregunta por GPU y hace fallback seguro.",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=os.getenv("LOCAL_TEST_MODELS_FILE", str(DEFAULT_MODELS_FILE)),
        help="Archivo JSON con la lista de modelos.",
    )
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("LOCAL_TEST_IMGSZ", "640")))
    parser.add_argument("--conf", type=float, default=float(os.getenv("LOCAL_TEST_CONF", "0.35")))
    parser.add_argument("--iou", type=float, default=float(os.getenv("LOCAL_TEST_IOU", "0.45")))
    parser.add_argument(
        "--max-det",
        type=int,
        default=int(os.getenv("LOCAL_TEST_MAX_DET", "100")),
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=os.getenv("LOCAL_TEST_TRACK", "true").strip().lower() in {"1", "true", "yes", "si"},
        help="Activa tracking persistente entre frames.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=os.getenv("LOCAL_TEST_TRACKER", "bytetrack.yaml"),
        help="Tracker de Ultralytics a usar, por ejemplo bytetrack.yaml o botsort.yaml.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guarda el video anotado en disco.",
    )
    return parser.parse_args()


def load_models(models_file: Path) -> tuple[str, list[ModelOption]]:
    raw_payload = json.loads(models_file.read_text(encoding="utf-8"))
    default_model_id = ""
    if isinstance(raw_payload, dict):
        default_model_id = str(raw_payload.get("default_model_id") or "").strip()
        raw_models = raw_payload.get("models")
    elif isinstance(raw_payload, list):
        raw_models = raw_payload
    else:
        raise ValueError(f"Archivo de modelos invalido: {models_file}")

    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"No hay modelos definidos en {models_file}")

    options: list[ModelOption] = []
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError(f"El modelo #{index} no es un objeto JSON valido.")

        model_id = str(raw_model.get("id") or raw_model.get("model_id") or "").strip()
        if not model_id:
            raise ValueError(f"El modelo #{index} no tiene id.")

        name = str(raw_model.get("name") or raw_model.get("display_name") or model_id).strip()
        selector = str(raw_model.get("selector") or "").strip()
        path_raw = str(raw_model.get("path") or raw_model.get("model_path") or "").strip()
        if not path_raw:
            raise ValueError(f"El modelo '{model_id}' no tiene path.")

        model_path = Path(path_raw).expanduser()
        if not model_path.is_absolute():
            model_path = (REPO_ROOT / model_path).resolve()

        options.append(
            ModelOption(
                model_id=model_id,
                name=name or model_id,
                selector=selector,
                path=model_path,
            )
        )

    if not default_model_id:
        default_model_id = options[0].model_id

    return default_model_id, options


def resolve_path(raw_value: str, base_dir: Path | None = None) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        root = base_dir or Path.cwd()
        path = (root / path).resolve()
    return path


def find_videos(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def prompt_video_from_directory(video_dir: Path) -> Path:
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No se encontraron videos en la carpeta: {video_dir}")
    if len(videos) == 1:
        print(f"[info] Video detectado automaticamente: {videos[0]}")
        return videos[0]

    print("\nVideos disponibles:")
    for index, video in enumerate(videos, start=1):
        print(f"  {index}. {video.name}")

    while True:
        selected = input("Escoge el numero del video: ").strip()
        if not selected:
            return videos[0]
        if selected.isdigit():
            index = int(selected)
            if 1 <= index <= len(videos):
                return videos[index - 1]
        print(f"[error] Seleccion no valida: {selected}")


def prompt_video_path(initial_value: str, video_dir_value: str) -> Path:
    candidate = initial_value.strip()
    video_dir = resolve_path(video_dir_value) if video_dir_value.strip() else None
    if not candidate and video_dir is not None:
        if not video_dir.exists() or not video_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de videos: {video_dir}")
        return prompt_video_from_directory(video_dir)
    while True:
        raw_value = candidate or input("Ruta del video: ").strip()
        video_path = resolve_path(raw_value)
        if video_path.exists() and video_path.is_file():
            return video_path
        print(f"[error] No existe el video: {video_path}")
        candidate = ""


def prompt_model_choice(
    requested_value: str,
    default_model_id: str,
    options: list[ModelOption],
) -> ModelOption:
    option_map: dict[str, ModelOption] = {}
    print("\nModelos disponibles:")
    for index, option in enumerate(options, start=1):
        selector_label = option.selector or str(index)
        print(
            f"  {index}. selector={selector_label} id={option.model_id} name={option.name} path={option.path.name}"
        )
        option_map[str(index)] = option
        option_map[option.model_id] = option
        if option.selector:
            option_map[option.selector] = option

    candidate = requested_value.strip()
    default_choice = next(
        (option for option in options if option.model_id == default_model_id),
        options[0],
    )
    while True:
        if candidate:
            normalized = candidate.strip()
        else:
            normalized = input(
                f"\nModelo a usar [default={default_choice.model_id}/{default_choice.selector or 'sin-selector'}]: "
            ).strip()
        if not normalized:
            return default_choice
        option = option_map.get(normalized)
        if option is not None:
            return option
        print(f"[error] Modelo no valido: {normalized}")
        candidate = ""


def prompt_device(device_arg: str) -> str:
    normalized = device_arg.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            print("[warn] Se pidio GPU pero CUDA no esta disponible. Se usara CPU.")
            return "cpu"
        return "cuda:0"

    if normalized != "auto":
        raise ValueError(f"Device no soportado: {device_arg}")

    wants_gpu = input("Usar GPU/CUDA? [y/N]: ").strip().lower()
    if wants_gpu in {"y", "yes", "s", "si"}:
        if torch.cuda.is_available():
            return "cuda:0"
        print("[warn] CUDA no esta disponible. Se usara CPU.")
    return "cpu"


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    return capture


def warmup_model(
    model: YOLO,
    video_path: Path,
    *,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> None:
    capture = open_capture(video_path)
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer un frame para warmup: {video_path}")
        sync_device(device)
        model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        sync_device(device)
    finally:
        capture.release()


def create_writer(
    capture: cv2.VideoCapture,
    sample_frame: Any,
    output_path: Path,
) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_height, frame_width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")
    return writer


def draw_text(frame: Any, text: str, x: int, y: int) -> None:
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_SHADOW,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def summarize_labels(result: Any) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return "sin detecciones"
    names = result.names if isinstance(result.names, dict) else {}
    class_ids = result.boxes.cls.int().cpu().tolist()
    counts = Counter(str(names.get(class_id, class_id)) for class_id in class_ids)
    return ", ".join(f"{label}:{count}" for label, count in counts.items())


def count_active_tracks(result: Any) -> int:
    boxes = getattr(result, "boxes", None)
    track_ids = getattr(boxes, "id", None)
    if track_ids is None:
        return 0
    try:
        return len(track_ids.int().cpu().tolist())
    except Exception:
        return 0


def run_inference(
    model: YOLO,
    frame: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    device: str,
    track: bool,
    tracker: str,
) -> list[Any]:
    kwargs = {
        "source": frame,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "device": device,
        "verbose": False,
    }
    if track:
        return model.track(
            persist=True,
            tracker=tracker,
            **kwargs,
        )
    return model.predict(**kwargs)


def main() -> None:
    args = parse_args()
    models_file = resolve_path(args.models_file, REPO_ROOT)
    if not models_file.exists():
        raise FileNotFoundError(f"No existe el archivo de modelos: {models_file}")

    default_model_id, models = load_models(models_file)
    video_path = prompt_video_path(args.video, args.video_dir)
    selected_model = prompt_model_choice(args.model, default_model_id, models)
    if not selected_model.path.exists():
        raise FileNotFoundError(
            f"No existe el archivo del modelo '{selected_model.model_id}': {selected_model.path}"
        )

    device = prompt_device(args.device)
    using_gpu = device.startswith("cuda")

    print("\nConfiguracion elegida:")
    print(f"  video   : {video_path}")
    print(f"  modelo  : {selected_model.model_id} ({selected_model.name})")
    print(f"  weights : {selected_model.path}")
    print(f"  device  : {device}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  conf    : {args.conf}")
    print(f"  iou     : {args.iou}")
    print(f"  max_det : {args.max_det}")
    print(f"  track   : {'si' if args.track else 'no'}")
    if args.track:
        print(f"  tracker : {args.tracker}")

    print("\nCargando modelo...")
    model = YOLO(str(selected_model.path)).to(device)
    print("Haciendo warmup...")
    warmup_model(
        model,
        video_path,
        device=device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )

    capture = open_capture(video_path)
    writer: cv2.VideoWriter | None = None
    display_enabled = True

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}__{selected_model.model_id}__{'gpu' if using_gpu else 'cpu'}.mp4"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_predict_ms = 0.0
    total_loop_started = perf_counter()
    frame_index = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    normalized_source_fps = source_fps if source_fps and source_fps > 0 else 0.0
    wait_ms = 1 if normalized_source_fps <= 0 else max(1, int(1000 / normalized_source_fps))

    print("\nControles: q = salir, espacio = pausar/reanudar")
    try:
        while True:
            loop_started = perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("\nFin del video.")
                break

            sync_device(device)
            predict_started = perf_counter()
            results = run_inference(
                model,
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=device,
                track=args.track,
                tracker=args.tracker,
            )
            sync_device(device)
            predict_ms = (perf_counter() - predict_started) * 1000
            total_predict_ms += predict_ms

            result = results[0]
            annotated = result.plot()
            frame_index += 1
            avg_predict_ms = total_predict_ms / frame_index
            elapsed = perf_counter() - total_loop_started
            effective_fps = frame_index / elapsed if elapsed > 0 else 0.0
            realtime_ratio = (
                effective_fps / normalized_source_fps if normalized_source_fps > 0 else 0.0
            )
            total_boxes = 0 if result.boxes is None else len(result.boxes)
            active_tracks = count_active_tracks(result)
            labels_summary = summarize_labels(result)
            loop_ms = (perf_counter() - loop_started) * 1000

            draw_text(
                annotated,
                f"frame={frame_index} det={total_boxes} tracks={active_tracks} device={device}",
                12,
                28,
            )
            draw_text(
                annotated,
                f"pred={predict_ms:.1f}ms avg={avg_predict_ms:.1f}ms fps={effective_fps:.2f}",
                12,
                56,
            )
            draw_text(
                annotated,
                f"loop={loop_ms:.1f}ms src_fps={normalized_source_fps:.2f} ratio={realtime_ratio:.2f}x",
                12,
                84,
            )
            draw_text(
                annotated,
                f"model={selected_model.model_id} track={'on' if args.track else 'off'}",
                12,
                112,
            )
            draw_text(annotated, labels_summary, 12, 140)

            if writer is None and not args.no_save:
                writer = create_writer(capture, annotated, output_path)
            if writer is not None:
                writer.write(annotated)

            if display_enabled:
                try:
                    cv2.imshow(WINDOW_NAME, annotated)
                    pressed_key = cv2.waitKey(wait_ms) & 0xFF
                except cv2.error as exc:
                    display_enabled = False
                    pressed_key = -1
                    print(
                        "[warn] No se pudo abrir una ventana de OpenCV. "
                        f"Se seguira procesando y guardando video. Detalle: {exc}"
                    )
                if pressed_key == ord("q"):
                    print("\nPrueba interrumpida por el usuario.")
                    break
                if pressed_key == ord(" "):
                    while True:
                        pause_key = cv2.waitKey(0) & 0xFF
                        if pause_key in {ord(" "), ord("q")}:
                            if pause_key == ord("q"):
                                print("\nPrueba interrumpida por el usuario.")
                                return
                            break

            print(
                f"\rframe={frame_index} det={total_boxes} tracks={active_tracks} "
                f"pred={predict_ms:.1f}ms avg={avg_predict_ms:.1f}ms "
                f"fps={effective_fps:.2f} ratio={realtime_ratio:.2f}x labels={labels_summary[:60]}",
                end="",
                flush=True,
            )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("\n\nResumen:")
    print(f"  frames procesados : {frame_index}")
    if frame_index:
        print(f"  promedio predict  : {total_predict_ms / frame_index:.2f} ms")
    print(f"  video salida      : {output_path if not args.no_save else 'no guardado'}")


if __name__ == "__main__":
    main()
