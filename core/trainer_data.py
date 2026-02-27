import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "trainer_data.jsonl")


# Objective: Provide a shared monotonic timestamp source for log records.
def now_ts() -> float:
    return time.time()


# Objective: Ensure the shared JSONL data file exists before reading/writing.
def ensure_data_file() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "a", encoding="utf-8"):
            pass


# Objective: Append one JSON event line to the shared data stream.
def append_event(event: Dict[str, Any]) -> None:
    ensure_data_file()
    record = dict(event)
    record.setdefault("t", now_ts())
    with open(DATA_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Objective: Load recent valid JSON events, trimming very old lines for speed.
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
