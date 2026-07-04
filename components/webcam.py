"""
Webcam Component
=================
Real-time object detection from the browser's webcam using
`streamlit-webrtc`. Plain OpenCV `VideoCapture` doesn't work well inside
Streamlit's rerun-on-every-interaction model (it blocks), so we use
WebRTC: the browser streams frames to a background thread that runs
YOLOv8 on each one and streams annotated frames back.

Live stats (FPS, live class counts) are read from the processor's
thread-safe shared state and rendered in a bounded refresh loop — this
is the standard pattern from the streamlit-webrtc project's own object
detection example, since Streamlit has no native "push" update
mechanism for background-thread state.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from utils import detector, history, visualization

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Bound the live-stats refresh loop so the script always returns control
# back to Streamlit (e.g. if the user changes a sidebar setting) rather
# than spinning forever.
MAX_REFRESH_ITERATIONS = 300  # ~5 minutes at 1s intervals
REFRESH_INTERVAL_SECONDS = 1.0


class YOLOVideoProcessor:
    """Runs YOLOv8 on each incoming WebRTC video frame.

    Settings (`model_weight`, `conf`, `iou`) are updated from the main
    thread via `update_settings()` and read under a lock, since `recv()`
    executes on a separate WebRTC worker thread.
    """

    def __init__(self) -> None:
        self.model_weight = "yolov8n.pt"
        self.conf = 0.25
        self.iou = 0.45
        self.lock = threading.Lock()
        self.class_counts: dict[str, int] = {}
        self.object_count = 0
        self.fps = 0.0
        self.last_annotated_frame = None
        self._last_tick = time.perf_counter()

    def update_settings(self, model_weight: str, conf: float, iou: float) -> None:
        with self.lock:
            self.model_weight = model_weight
            self.conf = conf
            self.iou = iou

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            model_weight, conf, iou = self.model_weight, self.conf, self.iou

        try:
            result = detector.run_image_detection(image_bgr, model_weight=model_weight, conf=conf, iou=iou)
            annotated = result.annotated_image
            counts = dict(Counter(d.class_name for d in result.detections))
            object_count = result.object_count
        except Exception:  # noqa: BLE001 - never crash the video pipeline
            logger.exception("Webcam frame detection failed; passing frame through unannotated")
            annotated = image_bgr
            counts = {}
            object_count = 0

        now = time.perf_counter()
        elapsed = now - self._last_tick
        with self.lock:
            self.class_counts = counts
            self.object_count = object_count
            self.fps = (1.0 / elapsed) if elapsed > 0 else 0.0
            self.last_annotated_frame = annotated
        self._last_tick = now

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render(config: dict) -> None:
    st.markdown("### 📷 Live Webcam Detection")
    st.caption(
        "Grant camera access in your browser to start real-time YOLOv8 detection. "
        "Runs entirely client-side video → server-side inference over WebRTC."
    )

    webrtc_ctx = webrtc_streamer(
        key="yolo-webcam-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    processor = webrtc_ctx.video_processor
    if processor is not None:
        processor.update_settings(
            config["model_weight"], config["confidence_threshold"], config["iou_threshold"]
        )

    if not webrtc_ctx.state.playing:
        st.info("▶️ Click **Start** above to begin live detection.")
        return

    st.markdown("#### Live Stats")
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()

    snapshot_col1, snapshot_col2 = st.columns([1, 3])
    take_snapshot = snapshot_col1.button("📸 Capture snapshot", use_container_width=True)

    if take_snapshot and processor is not None:
        with processor.lock:
            frame = processor.last_annotated_frame
            counts_snapshot = dict(processor.class_counts)
            object_count_snapshot = processor.object_count
        if frame is not None:
            df = pd.DataFrame(
                [{"class_name": k, "confidence": None} for k, v in counts_snapshot.items() for _ in range(v)]
            )
            snapshot_col2.image(visualization.bgr_to_rgb(frame), caption="Captured frame", width=320)
            history.log_detection(
                kind="image",
                source_name=f"webcam_snapshot_{int(time.time())}.png",
                detections_df=df,
                extra={"object_count": object_count_snapshot, "model": config["model_label"]},
            )
            st.success("Snapshot saved to session history.")

    # Bounded live-refresh loop for FPS / class-count display.
    for _ in range(MAX_REFRESH_ITERATIONS):
        if processor is None or not webrtc_ctx.state.playing:
            break
        with processor.lock:
            fps = processor.fps
            object_count = processor.object_count
            class_counts = dict(processor.class_counts)

        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Live FPS", f"{fps:.1f}")
            col2.metric("Objects in frame", object_count)
            col3.metric("Unique classes", len(class_counts))

        with chart_placeholder.container():
            if class_counts:
                st.bar_chart(class_counts)
            else:
                st.caption("No objects detected in the current frame.")

        time.sleep(REFRESH_INTERVAL_SECONDS)
