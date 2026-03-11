from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from uuid import uuid4
import urllib.error
import urllib.request


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramTextNotification:
    text: str


@dataclass(frozen=True)
class TelegramPhotoNotification:
    photo_bytes: bytes
    filename: str
    caption: str


@dataclass(frozen=True)
class TelegramVideoNotification:
    video_bytes: bytes
    filename: str
    caption: str


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: int = 10,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout_seconds = timeout_seconds
        self._queue: queue.Queue[
            TelegramTextNotification
            | TelegramPhotoNotification
            | TelegramVideoNotification
            | None
        ] = queue.Queue(maxsize=200)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def notify_text(self, text: str) -> None:
        try:
            self._queue.put_nowait(TelegramTextNotification(text=text))
        except queue.Full:
            LOGGER.warning("Telegram queue is full. Dropping message.")

    def notify_photo(self, photo_bytes: bytes, filename: str, caption: str) -> None:
        if not photo_bytes:
            LOGGER.warning("Telegram photo payload is empty. Dropping message.")
            return
        safe_filename = filename.strip() or "event.jpg"
        try:
            self._queue.put_nowait(
                TelegramPhotoNotification(
                    photo_bytes=photo_bytes,
                    filename=safe_filename,
                    caption=caption,
                )
            )
        except queue.Full:
            LOGGER.warning("Telegram queue is full. Dropping photo.")

    def notify_video(self, video_bytes: bytes, filename: str, caption: str) -> None:
        if not video_bytes:
            LOGGER.warning("Telegram video payload is empty. Dropping message.")
            return
        safe_filename = filename.strip() or "event.mp4"
        try:
            self._queue.put_nowait(
                TelegramVideoNotification(
                    video_bytes=video_bytes,
                    filename=safe_filename,
                    caption=caption,
                )
            )
        except queue.Full:
            LOGGER.warning("Telegram queue is full. Dropping video.")

    def _run(self) -> None:
        while True:
            notification = self._queue.get()
            if notification is None:
                return
            if isinstance(notification, TelegramTextNotification):
                self._send_message(notification.text)
                continue
            if isinstance(notification, TelegramPhotoNotification):
                self._send_photo(
                    photo_bytes=notification.photo_bytes,
                    filename=notification.filename,
                    caption=notification.caption,
                )
                continue
            self._send_video(
                video_bytes=notification.video_bytes,
                filename=notification.filename,
                caption=notification.caption,
            )

    @staticmethod
    def _extract_error_description(body: str) -> str:
        description = body
        if body:
            try:
                response_payload = json.loads(body)
            except json.JSONDecodeError:
                response_payload = None
            if isinstance(response_payload, dict):
                description = str(response_payload.get("description") or body)
        return description

    @staticmethod
    def _encode_multipart(
        fields: dict[str, str],
        *,
        file_field: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> tuple[str, bytes]:
        boundary = f"----telegram-{uuid4().hex}"
        body: list[bytes] = []
        for name, value in fields.items():
            body.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    (
                        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    ).encode("utf-8"),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )

        body.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    'Content-Disposition: form-data; name="'
                    f'{file_field}"; filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        return boundary, b"".join(body)

    def _send_message(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = json.dumps(
            {
                "chat_id": self._chat_id,
                "text": text,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            description = self._extract_error_description(body)
            LOGGER.error(
                "Telegram notification failed: status=%s reason=%s detail=%s",
                exc.code,
                exc.reason,
                description,
            )
        except urllib.error.URLError as exc:
            LOGGER.error("Telegram notification failed: %s", exc)

    def _send_binary(
        self,
        *,
        endpoint: str,
        file_field: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        caption: str,
        extra_fields: dict[str, str] | None = None,
    ) -> bool:
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        fields = {
            "chat_id": self._chat_id,
            "caption": caption,
        }
        if extra_fields:
            fields.update(extra_fields)
        boundary, payload = self._encode_multipart(
            fields,
            file_field=file_field,
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
        )
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                response.read()
            return True
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            description = self._extract_error_description(body)
            LOGGER.error(
                "Telegram %s notification failed: status=%s reason=%s detail=%s",
                endpoint,
                exc.code,
                exc.reason,
                description,
            )
        except urllib.error.URLError as exc:
            LOGGER.error("Telegram %s notification failed: %s", endpoint, exc)
        return False

    def _send_photo(self, photo_bytes: bytes, filename: str, caption: str) -> None:
        self._send_binary(
            endpoint="sendPhoto",
            file_field="photo",
            filename=filename,
            content_type="image/jpeg",
            file_bytes=photo_bytes,
            caption=caption,
        )

    def _send_video(self, video_bytes: bytes, filename: str, caption: str) -> None:
        sent = self._send_binary(
            endpoint="sendVideo",
            file_field="video",
            filename=filename,
            content_type="video/mp4",
            file_bytes=video_bytes,
            caption=caption,
            extra_fields={"supports_streaming": "true"},
        )
        if sent:
            return

        LOGGER.warning(
            "Falling back to sendDocument for Telegram video delivery: filename=%s",
            filename,
        )
        self._send_binary(
            endpoint="sendDocument",
            file_field="document",
            filename=filename,
            content_type="video/mp4",
            file_bytes=video_bytes,
            caption=caption,
        )
