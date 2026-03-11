from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.error
import urllib.request


LOGGER = logging.getLogger(__name__)


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
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=200)
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
            self._queue.put_nowait(text)
        except queue.Full:
            LOGGER.warning("Telegram queue is full. Dropping message.")

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                return
            self._send_message(message)

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
            description = body
            if body:
                try:
                    response_payload = json.loads(body)
                except json.JSONDecodeError:
                    response_payload = None
                if isinstance(response_payload, dict):
                    description = str(response_payload.get("description") or body)
            LOGGER.error(
                "Telegram notification failed: status=%s reason=%s detail=%s",
                exc.code,
                exc.reason,
                description,
            )
        except urllib.error.URLError as exc:
            LOGGER.error("Telegram notification failed: %s", exc)
