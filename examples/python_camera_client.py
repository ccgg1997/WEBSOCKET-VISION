from __future__ import annotations

import argparse
import asyncio
import base64
import json
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cliente de camara para consumir el servicio YOLO WebSocket."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL del WebSocket, por ejemplo ws://localhost:8000/ws/infer",
    )
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source-id", type=str, default="camera-local-01")
    parser.add_argument("--source-name", type=str, default="Camara Local")
    parser.add_argument("--interval-ms", type=int, default=300)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--return-image",
        action="store_true",
        help="Solicita la imagen anotada en base64.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la vista local de la camara.",
    )
    parser.add_argument(
        "--simulate-raspberry-alert",
        action="store_true",
        help="Imprime una alerta cuando el backend dispara un evento.",
    )
    return parser.parse_args()


async def authenticate(
    websocket: websockets.ClientConnection,
    source_id: str,
    source_name: str,
    username: str,
    password: str,
) -> None:
    greeting = json.loads(await websocket.recv())
    print(f"[server] {greeting}")

    await websocket.send(
        json.dumps(
            {
                "type": "auth",
                "username": username,
                "password": password,
                "source_id": source_id,
                "source_type": "camera",
                "source_name": source_name,
            }
        )
    )

    auth_response = json.loads(await websocket.recv())
    if auth_response.get("type") != "auth_ok":
        raise RuntimeError(f"Autenticacion fallida: {auth_response}")

    print(f"[auth] sesion iniciada: {auth_response['session_id']}")
    print(f"[auth] modelo: {auth_response['model']}")


async def stream_camera(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {args.camera}.")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    last_reported_event_id: str | None = None
    minio_overlay_until = 0.0
    minio_overlay_text = ""

    try:
        async with websockets.connect(
            args.url,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            await authenticate(
                websocket,
                args.source_id,
                args.source_name,
                args.username,
                args.password,
            )

            while True:
                started = perf_counter()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("No se pudo leer un frame de la camara.")

                encoded_ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if not encoded_ok:
                    raise RuntimeError("No se pudo codificar el frame a JPEG.")

                frame_id = uuid4().hex
                if args.return_image:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "frame",
                                "frame_id": frame_id,
                                "return_image": True,
                                "image_b64": base64.b64encode(encoded.tobytes()).decode(
                                    "ascii"
                                ),
                            }
                        )
                    )
                else:
                    await websocket.send(encoded.tobytes())

                response = json.loads(await websocket.recv())
                server_frame_id = response.get("frame_id", frame_id)
                inference_ms = response.get("latency_ms")
                total = response.get("counts", {}).get("total", 0)
                labels = response.get("counts", {}).get("by_label", {})
                snapshot = response.get("snapshot")
                event = response.get("event")
                roundtrip_ms = (perf_counter() - started) * 1000

                print(
                    f"[local={frame_id} server={server_frame_id}] server={inference_ms}ms roundtrip={roundtrip_ms:.1f}ms det={total} labels={labels}"
                )
                if snapshot:
                    print(
                        f"[snapshot] id={snapshot.get('id')} dominant={snapshot.get('dominant_label')} "
                        f"percent={snapshot.get('dominant_percent')} event_state={snapshot.get('event_state')}"
                    )
                if event:
                    event_id = str(event.get("event_id") or event.get("id") or "")
                    if event_id and event_id != last_reported_event_id:
                        event_state = str(event.get("state") or "")
                        print(
                            f"[event] id={event_id} "
                            f"state={event_state} "
                            f"trigger={event.get('trigger_label')} percent={event.get('trigger_percent')} "
                            f"annotation={event.get('annotation')}"
                        )
                        if event_state == "saved":
                            print(
                                f"[minio] preview={event.get('preview_filename')} clip={event.get('clip_filename')}"
                            )
                            minio_overlay_until = perf_counter() + 3.0
                            minio_overlay_text = (
                                f"SUBIDO A MINIO: {event.get('preview_filename')}"
                            )
                        if args.simulate_raspberry_alert:
                            print(
                                "[ALERTA] Se mandaria una senal simulada para apagar/encender Raspberry."
                            )
                        last_reported_event_id = event_id

                if args.show:
                    preview = frame.copy()
                    annotated_b64 = response.get("annotated_image_b64")
                    if args.return_image and annotated_b64:
                        decoded = base64.b64decode(annotated_b64)
                        preview_buffer = cv2.imdecode(
                            np.frombuffer(decoded, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if preview_buffer is not None:
                            preview = preview_buffer
                    cv2.putText(
                        preview,
                        f"det={total} server={inference_ms}ms",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    if minio_overlay_text and perf_counter() < minio_overlay_until:
                        cv2.rectangle(preview, (8, 48), (620, 92), (0, 120, 255), -1)
                        cv2.putText(
                            preview,
                            minio_overlay_text[:60],
                            (18, 78),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (255, 255, 255),
                            2,
                        )
                    cv2.imshow("camera-client", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                await asyncio.sleep(max(args.interval_ms / 1000, 0.01))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    asyncio.run(stream_camera(args))


if __name__ == "__main__":
    main()
