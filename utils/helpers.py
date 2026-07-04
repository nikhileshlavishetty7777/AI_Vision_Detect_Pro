"""
Helpers
=======
General-purpose utilities shared across the app: logging setup, project
directory bootstrapping, global CSS injection (glassmorphism + gradient
themes), and image/video upload validation & I/O.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MAX_IMAGE_DIM = 1600          # longest side, in pixels, before we downscale
MAX_VIDEO_SIZE_MB = 200        # upload size cap for the video page
REQUIRED_DIRS = ["assets", "models", "uploads", "outputs", "reports", "screenshots"]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once per process (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (Streamlit reruns the script repeatedly)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Project directory bootstrap
# --------------------------------------------------------------------------- #
def ensure_project_directories(project_root: Path) -> None:
    """Create required runtime directories (models/uploads/outputs/...) if missing."""
    for dirname in REQUIRED_DIRS:
        target = project_root / dirname
        target.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Image upload handling
# --------------------------------------------------------------------------- #
def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode a Streamlit UploadedFile into a BGR numpy array (OpenCV convention).

    Raises:
        ValueError: if the file is empty, unreadable, or not a valid image.
    """
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("the uploaded file is empty.")

    try:
        pil_image = Image.open(uploaded_file)
        pil_image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"not a valid image file ({exc}).") from exc

    # verify() invalidates the file handle for further reads — reopen it.
    uploaded_file.seek(0)
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def resize_if_needed(image_bgr: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    """Downscale an image (preserving aspect ratio) if its longest side exceeds max_dim."""
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dim:
        return image_bgr

    scale = max_dim / longest_side
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Video upload handling
# --------------------------------------------------------------------------- #
def save_uploaded_video(uploaded_file, uploads_dir: Path | None = None) -> Path:
    """Persist an uploaded video to the `uploads/` directory and return its path.

    Raises:
        ValueError: if the file is empty or exceeds the configured size cap.
    """
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("the uploaded file is empty.")

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise ValueError(f"file is {size_mb:.1f} MB, exceeds the {MAX_VIDEO_SIZE_MB} MB limit.")

    if uploads_dir is None:
        uploads_dir = Path(__file__).resolve().parent.parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    destination = uploads_dir / unique_name
    destination.write_bytes(raw_bytes)
    logger.info("Saved uploaded video to %s (%.1f MB)", destination, size_mb)
    return destination


def human_readable_size(num_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. '4.2 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# --------------------------------------------------------------------------- #
# Theming — glassmorphism + gradient dark/light CSS
# --------------------------------------------------------------------------- #
def inject_global_css(theme: str = "dark") -> None:
    """Inject the app's global stylesheet for the given theme ('dark' or 'light')."""
    if theme == "light":
        bg_gradient = "linear-gradient(135deg, #eef2ff 0%, #f7f9fc 50%, #eafaf1 100%)"
        text_color = "#1a1a2e"
        card_bg = "rgba(255, 255, 255, 0.55)"
        card_border = "rgba(30, 30, 60, 0.08)"
        subtitle_color = "#4a4a68"
        footer_color = "#6b6b85"
    else:
        bg_gradient = "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)"
        text_color = "#f1f1f6"
        card_bg = "rgba(255, 255, 255, 0.06)"
        card_border = "rgba(255, 255, 255, 0.12)"
        subtitle_color = "#b8b8d1"
        footer_color = "#8b8ba8"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg_gradient};
            color: {text_color};
        }}

        [data-testid="stSidebar"] {{
            background: {card_bg};
            backdrop-filter: blur(18px);
            border-right: 1px solid {card_border};
        }}

        .sidebar-brand {{
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        .app-header {{
            text-align: center;
            padding: 1.75rem 1rem 1.25rem 1rem;
        }}

        .app-header-title {{
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7f5af0, #2cb67d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}

        .app-header-subtitle {{
            color: {subtitle_color};
            font-size: 1.0rem;
            margin-top: 0.25rem;
        }}

        .app-footer {{
            text-align: center;
            color: {footer_color};
            font-size: 0.85rem;
            padding: 2rem 0 0.5rem 0;
            border-top: 1px solid {card_border};
            margin-top: 2rem;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {card_bg};
            backdrop-filter: blur(14px);
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 0.5rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }}

        [data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 0.75rem;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.1s ease;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
