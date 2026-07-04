"""
Analytics Component
=====================
Two page renderers:

- `render()`      — the "Analytics" page: aggregate confidence histogram,
                    class frequency bar/pie, detections-over-time trend,
                    and export buttons (CSV / JSON / Markdown report).
- `render_history()` — the "History" page: a filterable table of every
                    detection run this session, with per-run deletion
                    context and a full clear action.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import export, history, visualization


# ============================================================================
# ANALYTICS PAGE
# ============================================================================
def render(config: dict) -> None:
    st.markdown("### 📊 Detection Analytics")
    st.caption("Aggregate statistics across every detection run in this session.")

    all_detections = history.get_all_detections_dataframe()
    run_history = history.get_history()

    if not run_history:
        st.info(
            "No detections yet this session. Run detection on an image, video, "
            "or the live webcam first, then come back here."
        )
        return

    _render_summary_metrics(run_history, all_detections)

    st.markdown("#### Charts")
    col1, col2 = st.columns(2)
    col1.plotly_chart(visualization.confidence_histogram(all_detections), use_container_width=True)
    col2.plotly_chart(visualization.class_frequency_pie(all_detections), use_container_width=True)

    st.plotly_chart(visualization.class_frequency_bar(all_detections), use_container_width=True)

    history_df = history.get_history_dataframe()
    if "timestamp" in history_df.columns:
        history_df = history_df.copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        st.plotly_chart(visualization.detections_over_time(history_df), use_container_width=True)

    st.markdown("#### Export")
    _render_export_buttons(run_history, all_detections)


def _render_summary_metrics(run_history: list[dict], all_detections: pd.DataFrame) -> None:
    total_runs = len(run_history)
    total_objects = sum(e.get("object_count", 0) for e in run_history)
    unique_classes = all_detections["class_name"].nunique() if not all_detections.empty else 0
    avg_confidence = all_detections["confidence"].mean() if not all_detections.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", total_runs)
    col2.metric("Total Objects", total_objects)
    col3.metric("Unique Classes", unique_classes)
    col4.metric("Avg. Confidence", f"{avg_confidence * 100:.1f}%" if not all_detections.empty else "—")


def _render_export_buttons(run_history: list[dict], all_detections: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "⬇️ Export all detections (CSV)",
        data=export.dataframe_to_csv_bytes(all_detections),
        file_name="detections_export.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=all_detections.empty,
    )
    col2.download_button(
        "⬇️ Export all detections (JSON)",
        data=export.dataframe_to_json_bytes(all_detections),
        file_name="detections_export.json",
        mime="application/json",
        use_container_width=True,
        disabled=all_detections.empty,
    )
    report_md = export.generate_markdown_report("AI Vision Detect Pro — Session Report", run_history)
    col3.download_button(
        "⬇️ Download session report (.md)",
        data=report_md.encode("utf-8"),
        file_name="detection_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ============================================================================
# HISTORY PAGE
# ============================================================================
def render_history(config: dict) -> None:
    st.markdown("### 🕘 Detection History")
    st.caption("Every detection run from this session, most recent first.")

    entries = list(reversed(history.get_history()))
    if not entries:
        st.info("No history yet. Run a detection to see it appear here.")
        return

    kind_filter = st.multiselect(
        "Filter by type",
        options=["image", "video"],
        default=["image", "video"],
    )
    filtered = [e for e in entries if e.get("kind") in kind_filter]

    st.caption(f"Showing {len(filtered)} of {len(entries)} run(s).")

    for entry in filtered:
        icon = "🖼️" if entry.get("kind") == "image" else "🎬"
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            col1.markdown(f"{icon} **{entry.get('source_name', 'Unknown')}**")
            col2.caption(f"{entry.get('object_count', 0)} objects")
            col3.caption(entry.get("model", "—"))
            col4.caption(entry.get("timestamp", "—"))

            class_counts = entry.get("class_counts", {})
            if class_counts:
                with st.expander("Class breakdown"):
                    st.write(", ".join(f"{name} × {count}" for name, count in class_counts.items()))

    st.markdown("---")
    if st.button("🗑️ Clear all history", type="secondary"):
        history.clear_history()
        st.rerun()
