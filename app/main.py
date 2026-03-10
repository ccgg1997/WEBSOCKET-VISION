from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.auth import AuthManager
from app.config import ServiceSettings, load_settings
from app.events import EventManager
from app.inference import ModelRuntime


LOGGER = logging.getLogger("yolo.websocket")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _error_payload(code: str, detail: str, frame_id: str | None = None) -> dict[str, str]:
    payload = {
        "type": "error",
        "code": code,
        "detail": detail,
    }
    if frame_id:
        payload["frame_id"] = frame_id
    return payload


def _require_text_json(raw_text: str) -> dict:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("El mensaje recibido no es JSON valido.") from exc

    if not isinstance(payload, dict):
        raise ValueError("El mensaje JSON debe ser un objeto.")
    return payload


async def _authenticate(
    websocket: WebSocket,
    settings: ServiceSettings,
    auth_manager: AuthManager,
    runtime: ModelRuntime,
) -> dict[str, Any] | None:
    await websocket.send_json(
        {
            "type": "hello",
            "protocol": "yolo-ws-v1",
            "message": "Send auth message first.",
        }
    )

    try:
        incoming = await asyncio.wait_for(
            websocket.receive(),
            timeout=settings.auth_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await websocket.send_json(
            _error_payload("auth_timeout", "No auth message received in time.")
        )
        await websocket.close(code=1008)
        return None

    raw_text = incoming.get("text")
    if incoming.get("bytes") is not None or raw_text is None:
        await websocket.send_json(
            _error_payload("auth_required", "The first message must be JSON auth.")
        )
        await websocket.close(code=1008)
        return None

    try:
        payload = _require_text_json(raw_text)
    except ValueError as exc:
        await websocket.send_json(_error_payload("invalid_auth", str(exc)))
        await websocket.close(code=1008)
        return None

    if payload.get("type") != "auth":
        await websocket.send_json(
            _error_payload("auth_required", "The first message must be type=auth.")
        )
        await websocket.close(code=1008)
        return None

    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if not auth_manager.verify(username, password):
        await websocket.send_json(
            _error_payload("invalid_credentials", "Authentication failed.")
        )
        await websocket.close(code=1008)
        return None

    session_id = uuid4().hex
    source_id = str(payload.get("source_id") or session_id)
    source_type = str(payload.get("source_type") or "websocket").strip() or "websocket"
    source_name = str(payload.get("source_name") or source_id).strip() or source_id
    source_metadata = payload.get("source_metadata")
    if not isinstance(source_metadata, dict):
        source_metadata = {}

    await websocket.send_json(
        {
            "type": "auth_ok",
            "session_id": session_id,
            "source_id": source_id,
            "source_type": source_type,
            "source_name": source_name,
            "model": {
                "name": runtime.model_name,
                "device": runtime.device,
            },
        }
    )
    return {
        "session_id": session_id,
        "source_id": source_id,
        "source_type": source_type,
        "source_name": source_name,
        "source_metadata": source_metadata,
        "auth_username": username,
    }


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await asyncio.to_thread(ModelRuntime, settings)
        app.state.settings = settings
        app.state.runtime = runtime
        app.state.event_manager = EventManager(settings)
        app.state.auth_manager = AuthManager(
            username=settings.auth_username,
            password=settings.auth_password,
            password_hash=settings.auth_password_hash,
        )
        yield

    app = FastAPI(
        title="YOLO WebSocket Inference Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root() -> dict:
        runtime: ModelRuntime = app.state.runtime
        return {
            "service": "yolo-websocket-inference",
            "status": "ok",
            "ws_path": "/ws/infer",
            "model": {
                "name": runtime.model_name,
                "device": runtime.device,
            },
        }

    @app.get("/healthz")
    async def healthz() -> dict:
        runtime: ModelRuntime = app.state.runtime
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": runtime.model_name,
            "device": runtime.device,
        }

    @app.get("/events")
    async def list_events(limit: int = 50) -> dict:
        event_manager: EventManager = app.state.event_manager
        return {
            "items": event_manager.list_recent(limit=limit),
        }

    @app.get("/sources")
    async def list_sources(limit: int = 50) -> dict:
        event_manager: EventManager = app.state.event_manager
        return {
            "items": event_manager.list_sources(limit=limit),
        }

    @app.get("/sessions")
    async def list_sessions(limit: int = 50) -> dict:
        event_manager: EventManager = app.state.event_manager
        return {
            "items": event_manager.list_sessions(limit=limit),
        }

    @app.get("/snapshots")
    async def list_snapshots(limit: int = 50) -> dict:
        event_manager: EventManager = app.state.event_manager
        return {
            "items": event_manager.list_snapshots(limit=limit),
        }

    @app.get("/db/stats")
    async def db_stats() -> dict:
        event_manager: EventManager = app.state.event_manager
        return event_manager.get_stats()

    @app.websocket("/ws/infer")
    async def infer_socket(websocket: WebSocket) -> None:
        await websocket.accept()

        runtime: ModelRuntime = app.state.runtime
        event_manager: EventManager = app.state.event_manager
        auth_manager: AuthManager = app.state.auth_manager
        settings: ServiceSettings = app.state.settings

        session = await _authenticate(websocket, settings, auth_manager, runtime)
        if session is None:
            return
        session_id = session["session_id"]
        source_id = session["source_id"]
        source_type = session["source_type"]
        source_name = session["source_name"]
        source_metadata = session["source_metadata"]
        remote_addr = ""
        if websocket.client is not None:
            remote_addr = f"{websocket.client.host}:{websocket.client.port}"
        event_manager.create_session(
            session_id=session_id,
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            source_metadata=source_metadata,
            auth_username=session["auth_username"],
            remote_addr=remote_addr,
            model_name=runtime.model_name,
            model_device=runtime.device,
        )

        LOGGER.info("Session opened: %s", session_id)

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                raw_bytes = message.get("bytes")
                raw_text = message.get("text")
                frame_id = uuid4().hex
                return_image = settings.default_return_image

                if raw_bytes is not None:
                    frame_bytes = raw_bytes
                elif raw_text is not None:
                    try:
                        payload = _require_text_json(raw_text)
                    except ValueError as exc:
                        await websocket.send_json(
                            _error_payload("invalid_message", str(exc))
                        )
                        continue

                    message_type = payload.get("type")
                    if message_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    if message_type != "frame":
                        await websocket.send_json(
                            _error_payload(
                                "unsupported_message",
                                "Use type=frame or send binary JPEG bytes.",
                            )
                        )
                        continue

                    frame_id = str(payload.get("frame_id") or frame_id)
                    return_image = bool(
                        payload.get("return_image", settings.default_return_image)
                    )

                    image_b64 = payload.get("image_b64")
                    if not image_b64:
                        await websocket.send_json(
                            _error_payload(
                                "missing_image",
                                "The frame message requires image_b64.",
                                frame_id=frame_id,
                            )
                        )
                        continue

                    try:
                        frame_bytes = base64.b64decode(image_b64, validate=True)
                    except (ValueError, TypeError):
                        await websocket.send_json(
                            _error_payload(
                                "invalid_image",
                                "image_b64 is not valid base64.",
                                frame_id=frame_id,
                            )
                        )
                        continue
                else:
                    await websocket.send_json(
                        _error_payload(
                            "unsupported_message",
                            "Empty WebSocket message received.",
                        )
                    )
                    continue

                if len(frame_bytes) > settings.max_frame_bytes:
                    await websocket.send_json(
                        _error_payload(
                            "frame_too_large",
                            f"Frame exceeds {settings.max_frame_bytes} bytes.",
                            frame_id=frame_id,
                        )
                    )
                    continue

                try:
                    response = await runtime.infer_bytes(
                        image_bytes=frame_bytes,
                        frame_id=frame_id,
                        return_image=return_image,
                    )
                except ValueError as exc:
                    await websocket.send_json(
                        _error_payload("invalid_frame", str(exc), frame_id=frame_id)
                    )
                    continue
                except Exception as exc:  # pragma: no cover
                    LOGGER.exception("Inference error on session %s", session_id)
                    await websocket.send_json(
                        _error_payload("internal_error", str(exc), frame_id=frame_id)
                    )
                    continue

                event_update = event_manager.process_frame(
                    session_id=session_id,
                    frame_id=frame_id,
                    image_bytes=frame_bytes,
                    detections=response.get("detections", []),
                    counts=response.get("counts", {}).get("by_label", {}),
                    latency_ms=response.get("latency_ms", 0.0),
                    timestamp=time.time(),
                )
                if event_update is not None:
                    if "snapshot" in event_update:
                        response["snapshot"] = event_update["snapshot"]
                    if "event" in event_update:
                        response["event"] = event_update["event"]

                await websocket.send_json(response)

        except WebSocketDisconnect:
            pass
        finally:
            event_update = event_manager.close_session(session_id)
            if event_update is not None:
                LOGGER.info(
                    "Session %s finalized event %s",
                    session_id,
                    event_update.get("event_id"),
                )
            LOGGER.info("Session closed: %s", session_id)

    return app
