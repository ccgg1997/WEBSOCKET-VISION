from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cliente de prueba que envia una imagen JPEG repetidamente al WebSocket."
    )
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Modelo a usar en el backend, por ejemplo default o cana.",
    )
    parser.add_argument("--source-id", type=str, default="sample-image-01")
    parser.add_argument("--source-name", type=str, default="Sample Image Stream")
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="URL publica de una imagen JPEG para la prueba.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Ruta local a una imagen JPEG. Tiene prioridad sobre --image-url.",
    )
    parser.add_argument("--interval-ms", type=int, default=1500)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Cantidad de frames a enviar. 0 significa infinito.",
    )
    return parser.parse_args()


def load_image_bytes(args: argparse.Namespace) -> bytes:
    if args.image_path:
        return Path(args.image_path).expanduser().read_bytes()
    with urlopen(args.image_url, timeout=30) as response:
        return response.read()


async def main() -> None:
    args = parse_args()
    image_bytes = load_image_bytes(args)
    sent = 0

    async with websockets.connect(
        args.url,
        max_size=8 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        hello = json.loads(await websocket.recv())
        print(f"[hello] {hello}")

        auth_payload = {
            "type": "auth",
            "username": args.username,
            "password": args.password,
            "source_id": args.source_id,
            "source_type": "sample-image",
            "source_name": args.source_name,
            "source_metadata": {
                "image_url": args.image_url,
                "image_path": args.image_path,
            },
        }
        if args.model_id.strip():
            auth_payload["model_id"] = args.model_id.strip()

        await websocket.send(
            json.dumps(auth_payload)
        )

        auth_response = json.loads(await websocket.recv())
        if auth_response.get("type") != "auth_ok":
            raise RuntimeError(f"Autenticacion fallida: {auth_response}")
        print(f"[auth] {auth_response}")
        print(f"[auth] conexion corta: {auth_response.get('connection_id')}")

        while args.frame_count <= 0 or sent < args.frame_count:
            started = perf_counter()
            await websocket.send(image_bytes)
            response = json.loads(await websocket.recv())
            counts = response.get("counts", {})
            labels = counts.get("by_label", {})
            det = counts.get("total", 0)
            event = response.get("event")
            latency_ms = response.get("latency_ms")
            roundtrip_ms = round((perf_counter() - started) * 1000, 1)
            print(
                f"[frame={sent + 1}] det={det} labels={labels} server={latency_ms}ms roundtrip={roundtrip_ms}ms"
            )
            if event:
                print(f"[event] {event}")
            sent += 1
            await asyncio.sleep(max(args.interval_ms / 1000, 0.01))


if __name__ == "__main__":
    asyncio.run(main())
