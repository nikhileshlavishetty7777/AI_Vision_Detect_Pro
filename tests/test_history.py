"""Tests for utils/history.py"""

from __future__ import annotations

import pandas as pd
import pytest

from utils import history


@pytest.fixture(autouse=True)
def isolate_disk_history(tmp_path, monkeypatch):
    """Redirect disk persistence to a temp dir so tests never touch the real reports/ folder."""
    monkeypatch.setattr(history, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "history.json")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"class_id": 0, "class_name": "person", "confidence": 0.9},
            {"class_id": 2, "class_name": "car", "confidence": 0.6},
        ]
    )


class TestLogDetection:
    def test_appends_to_session_history(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df, extra={"model": "Nano"})
        entries = history.get_history()
        assert len(entries) == 1
        assert entries[0]["source_name"] == "a.jpg"
        assert entries[0]["object_count"] == 2
        assert entries[0]["model"] == "Nano"

    def test_class_counts_are_aggregated(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df)
        entries = history.get_history()
        assert entries[0]["class_counts"] == {"person": 1, "car": 1}

    def test_multiple_runs_accumulate(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df)
        history.log_detection("video", "b.mp4", sample_df)
        assert len(history.get_history()) == 2

    def test_empty_dataframe_logs_zero_objects(self):
        empty_df = pd.DataFrame(columns=["class_id", "class_name", "confidence"])
        history.log_detection("image", "empty.jpg", empty_df)
        entries = history.get_history()
        assert entries[0]["object_count"] == 0
        assert entries[0]["class_counts"] == {}

    def test_persists_to_disk(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df)
        assert history.HISTORY_FILE.exists()
        disk_entries = history.load_disk_history()
        assert len(disk_entries) == 1


class TestGetAllDetectionsDataframe:
    def test_accumulates_raw_rows_across_runs(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df)
        history.log_detection("image", "b.jpg", sample_df)
        all_df = history.get_all_detections_dataframe()
        assert len(all_df) == 4  # 2 detections x 2 runs
        assert "source_name" in all_df.columns

    def test_empty_when_no_runs_logged(self):
        all_df = history.get_all_detections_dataframe()
        assert all_df.empty


class TestClearHistory:
    def test_clears_both_summary_and_raw_rows(self, sample_df):
        history.log_detection("image", "a.jpg", sample_df)
        assert len(history.get_history()) == 1
        history.clear_history()
        assert history.get_history() == []
        assert history.get_all_detections_dataframe().empty
