"""
Integration tests for the full app.

Uses Streamlit's official `AppTest` harness to actually run `app.py`
top-to-bottom — page config, sidebar, theming, and routing — the same
way `streamlit run app.py` would, just without a browser. This is what
catches wiring bugs that pure unit tests on individual functions can't:
missing imports, mismatched function signatures between `app.py` and
`components/*.py`, and pages that crash on first render.

`ultralytics` and `streamlit-webrtc` are stubbed in `conftest.py` so this
runs fast and without the real multi-GB model dependency.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run()
    return at


class TestAppLoadsCleanly:
    def test_home_page_renders_without_exception(self):
        at = _run_app()
        assert at.exception == []

    def test_header_title_is_present(self):
        at = _run_app()
        markdown_html = " ".join(m.value for m in at.get("markdown"))
        assert "AI Vision Detect Pro" in markdown_html


class TestPageNavigation:
    """Drive the sidebar's navigation radio to each page and confirm no crash."""

    @staticmethod
    def _navigate_to(at: AppTest, page_label_substring: str) -> AppTest:
        radio = at.sidebar.radio(key="nav_radio")
        target = next(opt for opt in radio.options if page_label_substring in opt)
        radio.set_value(target)
        at.run()
        return at

    def test_image_detection_page_renders(self):
        at = _run_app()
        at = self._navigate_to(at, "Image Detection")
        assert at.exception == []

    def test_video_detection_page_renders(self):
        at = _run_app()
        at = self._navigate_to(at, "Video Detection")
        assert at.exception == []

    def test_live_webcam_page_renders(self):
        at = _run_app()
        at = self._navigate_to(at, "Live Webcam")
        assert at.exception == []

    def test_analytics_page_renders_with_empty_history(self):
        at = _run_app()
        at = self._navigate_to(at, "Analytics")
        assert at.exception == []

    def test_history_page_renders_with_empty_history(self):
        at = _run_app()
        at = self._navigate_to(at, "History")
        assert at.exception == []


class TestSidebarControls:
    def test_model_selection_updates_session_state(self):
        at = _run_app()
        assert at.session_state["model_size"] in {"n", "s", "m", "l", "x"}

    def test_confidence_slider_defaults_match_session_state(self):
        at = _run_app()
        assert at.session_state["confidence_threshold"] == 0.25
        assert at.session_state["iou_threshold"] == 0.45

    def test_theme_toggle_switches_theme(self):
        at = _run_app()
        toggle = at.sidebar.toggle[0]
        original = at.session_state["theme"]
        toggle.set_value(not toggle.value)
        at.run()
        assert at.session_state["theme"] != original
