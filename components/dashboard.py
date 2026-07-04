"""
Dashboard Component
====================
Renders the "Home" page: a welcoming overview with session statistics,
quick-navigation feature cards, and a peek at recent detection activity.
"""

from __future__ import annotations

import streamlit as st

from utils import history

FEATURE_CARDS: list[dict] = [
    {
        "page": "Image Detection",
        "icon": "🖼️",
        "title": "Image Detection",
        "description": "Upload one or more images and detect objects with adjustable confidence & IoU thresholds.",
    },
    {
        "page": "Video Detection",
        "icon": "🎬",
        "title": "Video Detection",
        "description": "Run frame-by-frame YOLOv8 detection on uploaded video files and download the annotated result.",
    },
    {
        "page": "Live Webcam",
        "icon": "📷",
        "title": "Live Webcam",
        "description": "Real-time detection straight from your browser's webcam feed, powered by WebRTC.",
    },
    {
        "page": "Analytics",
        "icon": "📊",
        "title": "Analytics",
        "description": "Confidence histograms, class-frequency charts, and exportable session reports.",
    },
]


def render(config: dict) -> None:
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 1.5rem;">
            <p style="font-size:1.05rem; opacity:0.85;">
                Currently running <b>{config.get('model_label', 'YOLOv8 Nano')}</b> ·
                Confidence ≥ {config.get('confidence_threshold', 0.25):.2f} ·
                IoU {config.get('iou_threshold', 0.45):.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_stats_overview()
    st.markdown("### Get Started")
    _render_feature_cards()
    st.markdown("### Recent Activity")
    _render_recent_activity()


def _render_stats_overview() -> None:
    entries = history.get_history()
    total_runs = len(entries)
    total_objects = sum(e.get("object_count", 0) for e in entries)
    image_runs = sum(1 for e in entries if e.get("kind") == "image")
    video_runs = sum(1 for e in entries if e.get("kind") == "video")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", total_runs)
    col2.metric("Objects Detected", total_objects)
    col3.metric("Images Processed", image_runs)
    col4.metric("Videos Processed", video_runs)


def _render_feature_cards() -> None:
    cols = st.columns(len(FEATURE_CARDS))
    for col, card in zip(cols, FEATURE_CARDS):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:2rem;'>{card['icon']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{card['title']}**")
                st.caption(card["description"])
                if st.button("Open →", key=f"open_{card['page']}", use_container_width=True):
                    st.session_state["active_page"] = card["page"]
                    st.rerun()


def _render_recent_activity() -> None:
    entries = history.get_history()
    if not entries:
        st.info("No detections yet this session. Try **Image Detection** or **Live Webcam** to get started.")
        return

    recent = list(reversed(entries))[:5]
    for entry in recent:
        icon = "🖼️" if entry.get("kind") == "image" else "🎬"
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 2])
            col1.markdown(f"{icon} **{entry.get('source_name', 'Unknown')}**")
            col2.caption(f"{entry.get('object_count', 0)} objects · {entry.get('model', '—')}")
            col3.caption(entry.get("timestamp", "—"))
