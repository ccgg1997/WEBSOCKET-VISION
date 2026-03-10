from __future__ import annotations

import argparse
import json
import time
from urllib.request import urlopen


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor simple para observar stats, snapshots y eventos del servicio."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://websocket-api:8000",
        help="URL base del backend HTTP.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Intervalo de consulta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seen_snapshot_ids: set[str] = set()
    seen_event_ids: set[str] = set()

    while True:
        try:
            stats = fetch_json(f"{args.base_url}/db/stats")
            print(f"[stats] {stats}", flush=True)

            snapshots = fetch_json(f"{args.base_url}/snapshots?limit=3").get("items", [])
            for snapshot in reversed(snapshots):
                snapshot_id = str(snapshot.get("id"))
                if not snapshot_id or snapshot_id in seen_snapshot_ids:
                    continue
                seen_snapshot_ids.add(snapshot_id)
                print(
                    "[snapshot] "
                    f"id={snapshot_id} source={snapshot.get('source_id')} "
                    f"dominant={snapshot.get('dominant_label')} "
                    f"percent={snapshot.get('dominant_percent')} "
                    f"event_state={snapshot.get('event_state')}",
                    flush=True,
                )

            events = fetch_json(f"{args.base_url}/events?limit=5").get("items", [])
            for event in reversed(events):
                event_id = str(event.get("id"))
                if not event_id or event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)
                print(
                    "[event] "
                    f"id={event_id} source={event.get('source_id')} "
                    f"trigger={event.get('trigger_label')} "
                    f"percent={event.get('trigger_percent')} "
                    f"preview={event.get('preview_object_key')} "
                    f"clip_url={event.get('clip_url')} "
                    f"preview_url={event.get('preview_url')}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[monitor-error] {exc}", flush=True)

        time.sleep(max(args.interval_seconds, 0.5))


if __name__ == "__main__":
    main()
