"""JSON-lines request logger shared by the serving APIs.

Every prediction is appended as one JSON object per line, tagged with the
serving model version, so the log can later be used to compare v1 vs v2
traffic or to detect data drift.
"""

import json
import os
import time
from pathlib import Path

LOG_DIR = Path(os.environ.get("REQUEST_LOG_DIR", "/app/logs"))
LOG_FILE = LOG_DIR / "requests.jsonl"


def log_prediction(text: str, sentiment: str, confidence: float, model_version: str, latency_ms: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "text": text,
        "prediction": sentiment,
        "confidence": confidence,
        "model_version": model_version,
        "latency_ms": latency_ms,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
