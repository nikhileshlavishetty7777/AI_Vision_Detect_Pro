"""Tests for utils/visualization.py"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from utils import visualization


@dataclass
class FakeDetection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple


@pytest.fixture
def sample_detections():
    return [
        FakeDetection(0, "person", 0.9, (0, 0, 10, 10)),
        FakeDetection(2, "car", 0.6, (0, 0, 20, 20)),
        FakeDetection(0, "person", 0.4, (0, 0, 10, 10)),
    ]


class TestBgrToRgb:
    def test_swaps_channel_order(self):
        image = np.zeros((2, 2, 3), dtype="uint8")
        image[0, 0] = [255, 0, 0]  # pure blue in BGR
        rgb = visualization.bgr_to_rgb(image)
        assert list(rgb[0, 0]) == [0, 0, 255]  # pure blue should now be last channel


class TestDetectionsToDataframe:
    def test_converts_nonempty_list(self, sample_detections):
        df = visualization.detections_to_dataframe(sample_detections)
        assert len(df) == 3
        assert set(df.columns) == {"class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"}
        assert df.iloc[0]["class_name"] == "person"

    def test_empty_list_returns_empty_dataframe_with_columns(self):
        df = visualization.detections_to_dataframe([])
        assert df.empty
        assert "class_name" in df.columns


class TestCharts:
    def test_confidence_histogram_with_data(self, sample_detections):
        df = visualization.detections_to_dataframe(sample_detections)
        fig = visualization.confidence_histogram(df)
        assert isinstance(fig, go.Figure)

    def test_confidence_histogram_empty(self):
        df = visualization.detections_to_dataframe([])
        fig = visualization.confidence_histogram(df)
        assert isinstance(fig, go.Figure)

    def test_class_frequency_bar_with_data(self, sample_detections):
        df = visualization.detections_to_dataframe(sample_detections)
        fig = visualization.class_frequency_bar(df)
        assert isinstance(fig, go.Figure)

    def test_class_frequency_pie_with_data(self, sample_detections):
        df = visualization.detections_to_dataframe(sample_detections)
        fig = visualization.class_frequency_pie(df)
        assert isinstance(fig, go.Figure)

    def test_detections_over_time_requires_timestamp_column(self):
        df = pd.DataFrame({"object_count": [1, 2, 3]})
        fig = visualization.detections_over_time(df)
        assert isinstance(fig, go.Figure)  # falls back to empty-state figure gracefully

    def test_detections_over_time_with_valid_data(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "object_count": [2, 5],
            }
        )
        fig = visualization.detections_over_time(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
