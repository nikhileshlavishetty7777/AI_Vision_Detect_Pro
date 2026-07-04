"""
Detector
========
Thin, well-typed wrapper around Ultralytics YOLOv8 for both single-image
and full-video inference. Model loading is cached via
`st.cache_resource` so switching pages/reruns doesn't reload weights
from disk every time.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class DetectionResult:
    annotated_image: np.ndarray  # BGR, boxes already drawn
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def object_count(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(d.class_name for d in self.detections))

    @property
    def average_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


@dataclass
class VideoDetectionResult:
    output_path: Path
    total_frames: int
    average_fps: float
    detections_df: pd.DataFrame


class DetectorError(RuntimeError):
    """Raised when model loading or inference fails in a user-actionable way."""


# ============================================================================
# Model loading (cached)
# ============================================================================
@st.cache_resource(show_spinner="🔄 Loading YOLOv8 model…")
def load_model(model_weight: str) -> YOLO:
    """Load (and cache) a YOLOv8 model by weight filename, e.g. 'yolov8n.pt'.

    Ultralytics will auto-download the weight file on first use if it is
    not already present. We point it at our own `models/` directory so
    downloads are consistent across reruns and easy to `.gitignore`.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODELS_DIR / model_weight
    source = str(local_path) if local_path.exists() else model_weight

    try:
        model = YOLO(source)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load YOLO model %s", model_weight)
        raise DetectorError(
            f"Could not load model weights '{model_weight}'. "
            "Check your internet connection (weights auto-download on first use) "
            f"or verify the file exists in {MODELS_DIR}."
        ) from exc

    # If Ultralytics downloaded fresh weights into the CWD, move them into
    # our managed models/ directory for future reruns.
    cwd_copy = Path.cwd() / model_weight
    if cwd_copy.exists() and not local_path.exists():
        cwd_copy.replace(local_path)

    return model


# ============================================================================
# Image detection
# ============================================================================
def run_image_detection(
    image_bgr: np.ndarray,
    model_weight: str,
    conf: float = 0.25,
    iou: float = 0.45,
) -> DetectionResult:
    """Run YOLOv8 inference on a single BGR image and return an annotated result."""
    if image_bgr is None or image_bgr.size == 0:
        raise DetectorError("received an empty image.")

    model = load_model(model_weight)

    start = time.perf_counter()
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    detections = _extract_detections(result)
    annotated = result.plot()  # BGR numpy array with boxes/labels drawn

    return DetectionResult(
        annotated_image=annotated,
        detections=detections,
        inference_time_ms=elapsed_ms,
    )


def _extract_detections(result) -> list[Detection]:
    """Convert an Ultralytics Results object into our plain Detection list."""
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is None:
        return detections

    names = result.names
    for box in boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=confidence,
                box_xyxy=xyxy,
            )
        )
    return detections


# ============================================================================
# Video detection
# ============================================================================
def process_video(
    input_path: Path,
    model_weight: str,
    conf: float = 0.25,
    iou: float = 0.45,
    frame_skip: int = 2,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> VideoDetectionResult:
    """Run YOLOv8 detection across a video file, writing an annotated copy to outputs/.

    Frames not divisible by `frame_skip` are copied through unannotated to
    keep output video duration/timing consistent, while inference only
    runs on every Nth frame for performance.
    """
    model = load_model(model_weight)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise DetectorError(f"could not open video file: {input_path.name}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{uuid.uuid4().hex}_detected.mp4"

    # NOTE: we deliberately do NOT use cv2.VideoWriter with the common
    # 'mp4v' fourcc here — it encodes MPEG-4 Part 2, which most browsers
    # (Chrome included) will not play back inline via <video>/st.video(),
    # even though the file itself is perfectly valid and opens fine in
    # players like VLC. We use imageio's ffmpeg backend to write real
    # H.264 instead, so the processed video actually previews in the app.
    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
    except Exception as exc:  # noqa: BLE001
        capture.release()
        raise DetectorError(
            "could not initialize the video writer (ffmpeg backend). "
            "Ensure 'imageio-ffmpeg' is installed."
        ) from exc

    all_detections: list[Detection] = []
    frame_idx = 0
    processing_start = time.perf_counter()
    last_annotated_frame_bgr: np.ndarray | None = None

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_idx += 1

            if frame_idx % frame_skip == 0 or last_annotated_frame_bgr is None:
                results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
                result = results[0]
                all_detections.extend(_extract_detections(result))
                last_annotated_frame_bgr = result.plot()

            # imageio/ffmpeg expects RGB frames; OpenCV/Ultralytics work in BGR.
            writer.append_data(cv2.cvtColor(last_annotated_frame_bgr, cv2.COLOR_BGR2RGB))

            if progress_callback is not None:
                elapsed = time.perf_counter() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                progress_callback(frame_idx, total_frames, current_fps)
    finally:
        capture.release()
        writer.close()

    total_elapsed = time.perf_counter() - processing_start
    average_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

    df = pd.DataFrame(
        [
            {"class_name": d.class_name, "confidence": d.confidence, "class_id": d.class_id}
            for d in all_detections
        ]
    )

    return VideoDetectionResult(
        output_path=output_path,
        total_frames=frame_idx,
        average_fps=average_fps,
        detections_df=df,
    )
