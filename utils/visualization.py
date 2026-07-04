"""
Visualization
=============
Conversion of detection results into pandas DataFrames plus reusable
Plotly chart builders (confidence histogram, class frequency bar/pie).
Kept separate from `detector.py` so both the single-image page and the
aggregate analytics page can share identical chart styling.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TEMPLATE = "plotly_dark"
ACCENT_COLORWAY = ["#7f5af0", "#2cb67d", "#ff8906", "#e53170", "#3da5d9", "#f25f4c"]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for display in Streamlit/PIL."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detections_to_dataframe(detections: list) -> pd.DataFrame:
    """Convert a list of `Detection` objects into a flat DataFrame.

    Accepts an empty list gracefully, returning a DataFrame with the
    expected columns but zero rows (so `.empty` checks work everywhere).
    """
    columns = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    if not detections:
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "x1": d.box_xyxy[0],
            "y1": d.box_xyxy[1],
            "x2": d.box_xyxy[2],
            "y2": d.box_xyxy[3],
        }
        for d in detections
    ]
    return pd.DataFrame(rows, columns=columns)


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Build a histogram of detection confidence scores."""
    if df.empty:
        return _empty_figure("No detections to chart")

    fig = px.histogram(
        df,
        x="confidence",
        nbins=20,
        color_discrete_sequence=[ACCENT_COLORWAY[0]],
        title="Confidence Distribution",
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=320,
    )
    return fig


def class_frequency_bar(df: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart of object class frequency."""
    if df.empty:
        return _empty_figure("No detections to chart")

    counts = df["class_name"].value_counts().reset_index()
    counts.columns = ["class_name", "count"]
    counts = counts.sort_values("count", ascending=True)

    fig = px.bar(
        counts,
        x="count",
        y="class_name",
        orientation="h",
        color="class_name",
        color_discrete_sequence=ACCENT_COLORWAY,
        title="Object Frequency",
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title="Count",
        yaxis_title="",
        height=320,
    )
    return fig


def class_frequency_pie(df: pd.DataFrame) -> go.Figure:
    """Build a pie chart of object class share."""
    if df.empty:
        return _empty_figure("No detections to chart")

    counts = df["class_name"].value_counts().reset_index()
    counts.columns = ["class_name", "count"]

    fig = px.pie(
        counts,
        names="class_name",
        values="count",
        color_discrete_sequence=ACCENT_COLORWAY,
        title="Class Share",
        hole=0.45,
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    return fig


def detections_over_time(df: pd.DataFrame) -> go.Figure:
    """Build a line chart of detection volume over session history (analytics page).

    Expects a DataFrame with a `timestamp` column (datetime-like) and one
    row per historical detection run, plus an `object_count` column.
    """
    if df.empty or "timestamp" not in df.columns:
        return _empty_figure("No history yet")

    fig = px.line(
        df.sort_values("timestamp"),
        x="timestamp",
        y="object_count",
        markers=True,
        color_discrete_sequence=[ACCENT_COLORWAY[1]],
        title="Detections Over Time",
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Objects detected",
        height=320,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Placeholder figure shown when there's no data to chart yet."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#8b8ba8"),
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=320,
    )
    return fig
