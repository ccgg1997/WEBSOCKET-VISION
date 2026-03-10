from __future__ import annotations

import argparse
import asyncio
import json
from time import perf_counter

import cv2
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Puente RTSP/video/archivo hacia el servicio YOLO WebSocket."
    )
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="RTSP, ruta a video o indice de camara.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default="dvr-01",
        help="Identificador logico del DVR/canal/fuente.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="DVR Canal 1",
        help="Nombre legible de la fuente.",
    )
    parser.add_argument("--interval-ms", type=int, default=300)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_capture_source(raw_source: str) -> str | int:
    return int(raw_source) if raw_source.isdigit() else raw_source


async def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(resolve_capture_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {args.source}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    try:
        async with websockets.connect(
            args.url,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            hello = json.loads(await websocket.recv())
            print(f"[hello] {hello}")

            await websocket.send(
                json.dumps(
                    {
                        "type": "auth",
                        "username": args.username,
                        "password": args.password,
                        "source_id": args.source_id,
                        "source_type": "rtsp",
                        "source_name": args.source_name,
                        "source_metadata": {
                            "capture_source": args.source,
                        },
                    }
                )
            )
            auth_ok = json.loads(await websocket.recv())
            print(f"[auth] {auth_ok}")

            while True:
                started = perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break

                encoded_ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if not encoded_ok:
                    raise RuntimeError("No se pudo codificar el frame.")

                await websocket.send(encoded.tobytes())
                response = json.loads(await websocket.recv())

                event = response.get("event")
                if event:
                    print(f"[event] {event}")

                counts = response.get("counts", {})
                roundtrip_ms = round((perf_counter() - started) * 1000, 1)
                print(
                    f"[frame] det={counts.get('total', 0)} labels={counts.get('by_label', {})} roundtrip={roundtrip_ms}ms"
                )

                if args.show:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"det={counts.get('total', 0)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("rtsp-video-bridge", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                await asyncio.sleep(max(args.interval_ms / 1000, 0.01))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
