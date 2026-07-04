"""Tests for utils/export.py"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils import export


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"class_id": 0, "class_name": "person", "confidence": 0.91},
            {"class_id": 2, "class_name": "car", "confidence": 0.75},
        ]
    )


class TestDataframeToCsvBytes:
    def test_produces_valid_csv(self, sample_df):
        raw = export.dataframe_to_csv_bytes(sample_df)
        assert isinstance(raw, bytes)
        text = raw.decode("utf-8")
        assert "person" in text
        assert "car" in text
        assert text.count("\n") >= 2  # header + 2 rows


class TestDataframeToJsonBytes:
    def test_produces_valid_json(self, sample_df):
        raw = export.dataframe_to_json_bytes(sample_df)
        records = json.loads(raw.decode("utf-8"))
        assert len(records) == 2
        assert records[0]["class_name"] == "person"


class TestImageToBytes:
    def test_round_trips_through_pil(self):
        array = (np.random.rand(32, 32, 3) * 255).astype("uint8")
        raw = export.image_to_bytes(array, fmt="PNG")
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        # Should be decodable back into an image of the same size
        decoded = Image.open(__import__("io").BytesIO(raw))
        assert decoded.size == (32, 32)


class TestGenerateMarkdownReport:
    def test_includes_title_and_summary_stats(self):
        entries = [
            {
                "timestamp": "2026-01-01T00:00:00",
                "kind": "image",
                "source_name": "a.jpg",
                "object_count": 3,
                "model": "Nano",
            },
            {
                "timestamp": "2026-01-01T00:01:00",
                "kind": "video",
                "source_name": "b.mp4",
                "object_count": 5,
                "model": "Small",
            },
        ]
        report = export.generate_markdown_report("My Report", entries)
        assert "# My Report" in report
        assert "**Total detection runs:** 2" in report
        assert "**Total objects detected:** 8" in report
        assert "a.jpg" in report
        assert "b.mp4" in report

    def test_handles_empty_history(self):
        report = export.generate_markdown_report("Empty Report", [])
        assert "**Total detection runs:** 0" in report
        assert "**Total objects detected:** 0" in report
