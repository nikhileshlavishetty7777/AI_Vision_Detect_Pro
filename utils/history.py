"""
History
========
Lightweight detection history tracking. Primary storage is
`st.session_state["detection_history"]` (fast, per-session), with an
optional JSON file mirror in `reports/history.json` so a "recent files"
view can survive a page reload without needing a database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
HISTORY_FILE = REPORTS_DIR / "history.json"
MAX_DISK_ENTRIES = 200
ALL_DETECTIONS_KEY = "all_detections_rows"


def log_detection(kind: str, source_name: str, detections_df: pd.DataFrame, extra: dict | None = None) -> dict:
    """Record a completed detection run into session state (and disk).

    Args:
        kind: "image" or "video".
        source_name: original uploaded filename.
        detections_df: the per-detection DataFrame for this run.
        extra: additional fields to merge in (e.g. inference_time_ms, model).

    Returns:
        The entry dict that was recorded.
    """
    extra = extra or {}
    class_counts = (
        detections_df["class_name"].value_counts().to_dict() if not detections_df.empty else {}
    )

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "timestamp": timestamp,
        "kind": kind,
        "source_name": source_name,
        "object_count": int(len(detections_df)),
        "class_counts": class_counts,
        **extra,
    }

    history = st.session_state.setdefault("detection_history", [])
    history.append(entry)

    _append_detection_rows(detections_df, timestamp=timestamp, kind=kind, source_name=source_name)

    try:
        _append_to_disk(entry)
    except OSError:
        logger.warning("Could not persist detection history to disk", exc_info=True)

    return entry


def _append_detection_rows(detections_df: pd.DataFrame, timestamp: str, kind: str, source_name: str) -> None:
    """Append raw per-detection rows (with run metadata) to session state.

    Kept separate from the compact `detection_history` summary list so
    the analytics page can build real confidence histograms / class
    charts across an entire session, without bloating the lightweight
    disk-persisted history file with per-box data.
    """
    if detections_df.empty:
        return

    rows = detections_df.copy()
    rows["timestamp"] = timestamp
    rows["kind"] = kind
    rows["source_name"] = source_name

    existing = st.session_state.get(ALL_DETECTIONS_KEY)
    if existing is None or existing.empty:
        st.session_state[ALL_DETECTIONS_KEY] = rows
    else:
        st.session_state[ALL_DETECTIONS_KEY] = pd.concat([existing, rows], ignore_index=True)


def get_all_detections_dataframe() -> pd.DataFrame:
    """Return every individual detection box logged this session, across all runs."""
    df = st.session_state.get(ALL_DETECTIONS_KEY)
    if df is None:
        return pd.DataFrame(columns=["class_id", "class_name", "confidence", "timestamp", "kind", "source_name"])
    return df


def get_history() -> list[dict]:
    """Return the current session's detection history (most recent last)."""
    return st.session_state.get("detection_history", [])


def get_history_dataframe() -> pd.DataFrame:
    """Flatten session history into a DataFrame for analytics/history pages."""
    history = get_history()
    if not history:
        return pd.DataFrame(
            columns=["timestamp", "kind", "source_name", "object_count", "model"]
        )
    return pd.DataFrame(history)


def clear_history() -> None:
    """Clear in-memory session history (does not touch the disk mirror)."""
    st.session_state["detection_history"] = []
    st.session_state[ALL_DETECTIONS_KEY] = pd.DataFrame(
        columns=["class_id", "class_name", "confidence", "timestamp", "kind", "source_name"]
    )


def _append_to_disk(entry: dict) -> None:
    """Append a single entry to the JSON history file on disk, capped at MAX_DISK_ENTRIES."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    if HISTORY_FILE.exists():
        try:
            entries = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("history.json was unreadable; starting a fresh file")
            entries = []

    entries.append(entry)
    entries = entries[-MAX_DISK_ENTRIES:]
    HISTORY_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def load_disk_history() -> list[dict]:
    """Load the persisted history file from disk, if present."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse history.json", exc_info=True)
        return []
