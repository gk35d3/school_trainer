import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "trainer_data.jsonl")


def now_ts() -> float:
    return time.time()


def ensure_data_file() -> None:
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "a", encoding="utf-8"):
            pass


def append_event(event: Dict[str, Any]) -> None:
    ensure_data_file()
    record = dict(event)
    record.setdefault("t", now_ts())
    with open(DATA_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_recent_events(max_lines: int = 5000) -> List[Dict[str, Any]]:
    ensure_data_file()
    lines: Deque[str] = deque(maxlen=max_lines)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)

    events: List[Dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events
