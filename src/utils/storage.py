from __future__ import annotations
from typing import Dict, Any
import json, time, os

def log_session_event(path: str, event: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event = {**event, "timestamp": int(time.time()*1000)}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + "\n")