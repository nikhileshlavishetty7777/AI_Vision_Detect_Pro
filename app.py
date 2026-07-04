"""
AI Vision Detect Pro
=====================
Real-Time Object Detection using YOLOv8, OpenCV & Streamlit.

This is the main application entrypoint. It owns page configuration,
global theming (glassmorphism + gradient dark/light themes), top-level
navigation, and session-state initialization. All actual page content is
delegated to the `components/` package so this file stays a thin,
readable router.

Run with:
    streamlit run app.py

Author: Generated for CodeAlpha AI Internship Portfolio Project
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from components import analytics, dashboard, sidebar, uploader, webcam
from utils.helpers import ensure_project_directories, inject_global_css, setup_logging

# --------------------------------------------------------------------------- #
# Bootstrap: env vars, logging, and required project directories
# --------------------------------------------------------------------------- #
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ensure_project_directories(PROJECT_ROOT)

# --------------------------------------------------------------------------- #
# Page configuration — must be the first Streamlit call
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Vision Detect Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": (
            "## AI Vision Detect Pro\n"
            "Real-time object detection powered by YOLOv8, OpenCV, and Streamlit.\n\n"
            "Built as a CodeAlpha Artificial Intelligence Internship project."
        ),
    },
)

# --------------------------------------------------------------------------- #
# Session state defaults — centralized so every component can rely on these
# keys existing without repeated `if "x" not in st.session_state` checks.
# --------------------------------------------------------------------------- #
_DEFAULT_STATE: dict = {
    "theme": "dark",
    "active_page": "Home",
    "model_size": "n",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "detection_history": [],
    "last_results": None,
    "last_image_result": None,
    "last_video_result_path": None,
}

for _key, _default in _DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# --------------------------------------------------------------------------- #
# Global styling (theme-aware glassmorphism + gradients)
# --------------------------------------------------------------------------- #
inject_global_css(theme=st.session_state["theme"])

# --------------------------------------------------------------------------- #
# Sidebar — navigation + model/detection controls.
# Returns a plain dict snapshot of current settings so page renderers don't
# need to reach into st.session_state directly (keeps them testable).
# --------------------------------------------------------------------------- #
nav_config = sidebar.render_sidebar()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">🎯 AI Vision Detect Pro</div>
        <div class="app-header-subtitle">
            Real-Time Object Detection · YOLOv8 · OpenCV · Streamlit
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Page routing
# --------------------------------------------------------------------------- #
PAGES = {
    "Home": dashboard.render,
    "Image Detection": uploader.render_image_page,
    "Video Detection": uploader.render_video_page,
    "Live Webcam": webcam.render,
    "Analytics": analytics.render,
    "History": analytics.render_history,
}

active_page = nav_config.get("active_page", "Home")
page_renderer = PAGES.get(active_page)

if page_renderer is None:
    logger.warning("Unknown page requested: %s — falling back to Home", active_page)
    page_renderer = dashboard.render

try:
    page_renderer(nav_config)
except Exception:  # noqa: BLE001 - top-level guard so the whole app never white-screens
    logger.exception("Unhandled error while rendering page '%s'", active_page)
    st.error(
        "⚠️ Something went wrong while rendering this page. "
        "The error has been logged. Try adjusting your inputs or reloading."
    )
    with st.expander("Technical details"):
        st.exception(st.session_state.get("_last_exception"))

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="app-footer">
        Built with YOLOv8 &amp; Streamlit · AI Vision Detect Pro ·
        CodeAlpha Artificial Intelligence Internship
    </div>
    """,
    unsafe_allow_html=True,
)
