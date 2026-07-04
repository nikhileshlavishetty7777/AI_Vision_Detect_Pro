"""
Sidebar Component
==================
Renders the left navigation sidebar: page navigation, YOLOv8 model size
selector, confidence/IoU threshold sliders, and theme toggle.

This is the single source of truth for the "control panel" of the app.
It writes selections into `st.session_state` (so other reruns persist
state) and also returns a plain dict snapshot for convenient passing
into page renderers.
"""

from __future__ import annotations

import streamlit as st

from utils import history

# Maps the friendly model size label -> (YOLOv8 weight filename, description)
MODEL_OPTIONS: dict[str, tuple[str, str]] = {
    "Nano (fastest)": ("yolov8n.pt", "Smallest & fastest — best for CPU / real-time webcam"),
    "Small": ("yolov8s.pt", "Good balance of speed and accuracy"),
    "Medium": ("yolov8m.pt", "Higher accuracy, moderate speed"),
    "Large": ("yolov8l.pt", "High accuracy — slower on CPU"),
    "XLarge (most accurate)": ("yolov8x.pt", "Best accuracy — recommended with GPU"),
}

NAV_PAGES: list[tuple[str, str]] = [
    ("Home", "🏠"),
    ("Image Detection", "🖼️"),
    ("Video Detection", "🎬"),
    ("Live Webcam", "📷"),
    ("Analytics", "📊"),
    ("History", "🕘"),
]


def render_sidebar() -> dict:
    """Render the sidebar UI and return the current configuration snapshot.

    Returns:
        dict with keys: active_page, model_size, model_weight, model_label,
        confidence_threshold, iou_threshold, theme.
    """
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">🎯 <span>AI Vision</span> Detect Pro</div>',
            unsafe_allow_html=True,
        )
        st.caption("YOLOv8 · OpenCV · Streamlit")
        st.markdown("---")

        # --- Navigation ------------------------------------------------- #
        st.markdown("##### Navigation")
        labels = [f"{icon}  {name}" for name, icon in NAV_PAGES]
        default_index = next(
            (i for i, (name, _) in enumerate(NAV_PAGES) if name == st.session_state["active_page"]),
            0,
        )
        chosen_label = st.radio(
            "Navigate",
            options=labels,
            index=default_index,
            label_visibility="collapsed",
            key="nav_radio",
        )
        active_page = NAV_PAGES[labels.index(chosen_label)][0]
        st.session_state["active_page"] = active_page

        st.markdown("---")

        # --- Model selection ---------------------------------------------- #
        st.markdown("##### Detection Model")
        model_labels = list(MODEL_OPTIONS.keys())
        default_model_index = next(
            (i for i, lbl in enumerate(model_labels) if MODEL_OPTIONS[lbl][0].startswith(f"yolov8{st.session_state['model_size']}"))
            , 0,
        )
        chosen_model_label = st.selectbox(
            "YOLOv8 variant",
            options=model_labels,
            index=default_model_index,
            help="Larger models are more accurate but slower. Nano is recommended for CPU/live webcam.",
        )
        model_weight, model_description = MODEL_OPTIONS[chosen_model_label]
        st.session_state["model_size"] = model_weight.replace("yolov8", "").replace(".pt", "")
        st.caption(f"ℹ️ {model_description}")

        st.markdown("---")

        # --- Detection thresholds --------------------------------------- #
        st.markdown("##### Detection Thresholds")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state["confidence_threshold"]),
            step=0.05,
            help="Minimum confidence score for a detection to be kept.",
        )
        st.session_state["confidence_threshold"] = confidence_threshold

        iou_threshold = st.slider(
            "IoU threshold (NMS)",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state["iou_threshold"]),
            step=0.05,
            help="Intersection-over-Union threshold used for non-max suppression.",
        )
        st.session_state["iou_threshold"] = iou_threshold

        st.markdown("---")

        # --- Theme toggle -------------------------------------------------- #
        st.markdown("##### Appearance")
        theme_choice = st.toggle(
            "🌙 Dark mode",
            value=st.session_state["theme"] == "dark",
            help="Switch between dark and light dashboard themes.",
        )
        st.session_state["theme"] = "dark" if theme_choice else "light"

        st.markdown("---")

        # --- Quick stats ------------------------------------------------- #
        history_count = len(st.session_state.get("detection_history", []))
        st.markdown("##### Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Detections run", history_count)
        col2.metric("Model", chosen_model_label.split(" ")[0])

        if history_count > 0 and st.button("🗑️ Clear session history", use_container_width=True):
            history.clear_history()
            st.session_state["last_results"] = None
            st.session_state["last_image_result"] = None
            st.session_state["last_video_result_path"] = None
            st.rerun()

    return {
        "active_page": active_page,
        "model_size": st.session_state["model_size"],
        "model_weight": model_weight,
        "model_label": chosen_model_label,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "theme": st.session_state["theme"],
    }
