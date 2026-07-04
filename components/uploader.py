"""
Uploader Component
===================
Handles the "Image Detection" and "Video Detection" pages: drag & drop /
multi-file upload, running YOLOv8 inference, rendering annotated results
with per-item statistics, and offering downloads (image, CSV, JSON).

This module is UI orchestration only — actual inference lives in
`utils/detector.py`, chart/dataframe helpers live in
`utils/visualization.py`, file/byte packaging lives in `utils/export.py`,
and history persistence lives in `utils/history.py`.
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from utils import detector, export, helpers, history, visualization

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


# ============================================================================
# IMAGE DETECTION PAGE
# ============================================================================
def render_image_page(config: dict) -> None:
    st.markdown("### 🖼️ Image Object Detection")
    st.caption("Upload one or more images (drag & drop supported) to run YOLOv8 detection.")

    uploaded_files = st.file_uploader(
        "Drop images here or click to browse",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
        key="image_uploader",
    )

    if not uploaded_files:
        st.info("👆 Upload at least one image to begin detection.")
        return

    st.markdown(f"**{len(uploaded_files)} image(s) queued.**")
    run_clicked = st.button("🚀 Run Detection", type="primary", use_container_width=True)

    if not run_clicked:
        return

    overall_progress = st.progress(0.0, text="Starting…")

    for idx, uploaded_file in enumerate(uploaded_files):
        overall_progress.progress(
            idx / len(uploaded_files),
            text=f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})…",
        )
        _process_single_image(uploaded_file, config)

    overall_progress.progress(1.0, text="Done ✅")
    st.success(f"Finished processing {len(uploaded_files)} image(s).")


def _process_single_image(uploaded_file, config: dict) -> None:
    """Run detection on a single uploaded image and render its result card."""
    try:
        image_bgr = helpers.read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(f"❌ Could not read **{uploaded_file.name}**: {exc}")
        return

    image_bgr = helpers.resize_if_needed(image_bgr)

    try:
        result = detector.run_image_detection(
            image_bgr,
            model_weight=config["model_weight"],
            conf=config["confidence_threshold"],
            iou=config["iou_threshold"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Detection failed for %s", uploaded_file.name)
        st.error(f"❌ Detection failed for **{uploaded_file.name}**: {exc}")
        return

    df = visualization.detections_to_dataframe(result.detections)

    with st.container(border=True):
        st.markdown(f"#### {uploaded_file.name}")

        col_original, col_annotated = st.columns(2)
        with col_original:
            st.caption("Original")
            st.image(visualization.bgr_to_rgb(image_bgr), use_container_width=True)
        with col_annotated:
            st.caption(f"Detected ({result.object_count} objects · {result.inference_time_ms:.1f} ms)")
            st.image(visualization.bgr_to_rgb(result.annotated_image), use_container_width=True)

        _render_stat_cards(result)

        if not df.empty:
            with st.expander("📈 Detection breakdown"):
                chart_col1, chart_col2 = st.columns(2)
                chart_col1.plotly_chart(
                    visualization.confidence_histogram(df), use_container_width=True
                )
                chart_col2.plotly_chart(
                    visualization.class_frequency_bar(df), use_container_width=True
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

        _render_download_row(
            annotated_image_rgb=visualization.bgr_to_rgb(result.annotated_image),
            df=df,
            base_name=uploaded_file.name.rsplit(".", 1)[0],
        )

    history.log_detection(
        kind="image",
        source_name=uploaded_file.name,
        detections_df=df,
        extra={
            "inference_time_ms": result.inference_time_ms,
            "object_count": result.object_count,
            "model": config["model_label"],
        },
    )


def _render_stat_cards(result: detector.DetectionResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Objects detected", result.object_count)
    cols[1].metric("Unique classes", len(result.class_counts))
    cols[2].metric(
        "Avg. confidence",
        f"{result.average_confidence * 100:.1f}%" if result.detections else "—",
    )
    cols[3].metric("Inference time", f"{result.inference_time_ms:.1f} ms")


def _render_download_row(annotated_image_rgb, df, base_name: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇️ Download image",
        data=export.image_to_bytes(annotated_image_rgb, fmt="PNG"),
        file_name=f"{base_name}_detected.png",
        mime="image/png",
        use_container_width=True,
    )
    col2.download_button(
        "⬇️ Download CSV",
        data=export.dataframe_to_csv_bytes(df),
        file_name=f"{base_name}_detections.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=df.empty,
    )
    col3.download_button(
        "⬇️ Download JSON",
        data=export.dataframe_to_json_bytes(df),
        file_name=f"{base_name}_detections.json",
        mime="application/json",
        use_container_width=True,
        disabled=df.empty,
    )


# ============================================================================
# VIDEO DETECTION PAGE
# ============================================================================
def render_video_page(config: dict) -> None:
    st.markdown("### 🎬 Video Object Detection")
    st.caption("Upload a video to run frame-by-frame YOLOv8 detection.")

    uploaded_file = st.file_uploader(
        "Drop a video here or click to browse",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=False,
        key="video_uploader",
    )

    if uploaded_file is None:
        st.info("👆 Upload a video to begin detection.")
        return

    st.video(uploaded_file)

    max_mb = helpers.MAX_VIDEO_SIZE_MB
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_mb:
        st.error(f"❌ File is {size_mb:.1f} MB — the limit for this demo is {max_mb} MB.")
        return

    frame_skip = st.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=10,
        value=2,
        help="Higher values process fewer frames (faster, choppier output).",
    )

    if not st.button("🚀 Run Detection on Video", type="primary", use_container_width=True):
        return

    try:
        input_path = helpers.save_uploaded_video(uploaded_file)
    except ValueError as exc:
        st.error(f"❌ Could not save video: {exc}")
        return

    progress_bar = st.progress(0.0, text="Starting video processing…")
    status_text = st.empty()

    def _on_progress(frame_idx: int, total_frames: int, current_fps: float) -> None:
        fraction = min(frame_idx / max(total_frames, 1), 1.0)
        progress_bar.progress(fraction, text=f"Frame {frame_idx}/{total_frames} · {current_fps:.1f} FPS")

    start_time = time.perf_counter()
    try:
        result = detector.process_video(
            input_path=input_path,
            model_weight=config["model_weight"],
            conf=config["confidence_threshold"],
            iou=config["iou_threshold"],
            frame_skip=frame_skip,
            progress_callback=_on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Video detection failed for %s", uploaded_file.name)
        st.error(f"❌ Video processing failed: {exc}")
        return
    total_elapsed = time.perf_counter() - start_time

    progress_bar.progress(1.0, text="Done ✅")
    status_text.success(
        f"Processed {result.total_frames} frames in {total_elapsed:.1f}s "
        f"(avg {result.average_fps:.1f} FPS)."
    )

    st.markdown("#### Result")
    st.video(str(result.output_path))

    df = result.detections_df
    cols = st.columns(4)
    cols[0].metric("Frames processed", result.total_frames)
    cols[1].metric("Total detections", len(df))
    cols[2].metric("Unique classes", df["class_name"].nunique() if not df.empty else 0)
    cols[3].metric("Avg. FPS", f"{result.average_fps:.1f}")

    if not df.empty:
        with st.expander("📈 Detection breakdown"):
            chart_col1, chart_col2 = st.columns(2)
            chart_col1.plotly_chart(visualization.confidence_histogram(df), use_container_width=True)
            chart_col2.plotly_chart(visualization.class_frequency_bar(df), use_container_width=True)

    with open(result.output_path, "rb") as f:
        st.download_button(
            "⬇️ Download processed video",
            data=f.read(),
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_detected.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    history.log_detection(
        kind="video",
        source_name=uploaded_file.name,
        detections_df=df,
        extra={
            "frames_processed": result.total_frames,
            "average_fps": result.average_fps,
            "model": config["model_label"],
        },
    )
