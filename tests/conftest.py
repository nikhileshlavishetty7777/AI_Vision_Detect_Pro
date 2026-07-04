"""
Shared pytest fixtures for the AI Vision Detect Pro test suite.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# `ultralytics` (and its multi-GB torch dependency) is not installed in this
# CI/test environment. We stub it with a minimal fake module so `utils/
# detector.py` can be imported and unit-tested without requiring the real
# YOLOv8 weights or PyTorch. When the project is actually run via
# `streamlit run app.py` with the real `requirements.txt` installed, the
# genuine `ultralytics` package is used instead — this stub only exists for
# the test collection process.
# --------------------------------------------------------------------------- #
if "ultralytics" not in sys.modules:
    fake_ultralytics = types.ModuleType("ultralytics")

    class _StubYOLO:  # pragma: no cover - replaced by real YOLO at runtime
        def __init__(self, *args, **kwargs):
            self._args = args

        def predict(self, *args, **kwargs):
            raise RuntimeError(
                "Stub YOLO.predict() called directly in a test — "
                "monkeypatch `detector.load_model` instead."
            )

    fake_ultralytics.YOLO = _StubYOLO
    sys.modules["ultralytics"] = fake_ultralytics

# --------------------------------------------------------------------------- #
# `streamlit-webrtc` pulls in `aiortc` and native codec bindings that are
# heavy to install in a lightweight test environment. Stub it too, so
# `components/webcam.py` (and therefore the full `app.py`) can be imported
# and integration-tested. The stub's `webrtc_streamer()` simply returns a
# context object in the "not playing" state, which exercises the page's
# early-return path — a real browser session is required to test the live
# video loop itself, which is out of scope for automated unit tests.
# --------------------------------------------------------------------------- #
if "streamlit_webrtc" not in sys.modules:
    fake_webrtc = types.ModuleType("streamlit_webrtc")

    class _StubWebRtcMode:
        SENDRECV = "SENDRECV"

    class _StubRTCConfiguration:
        def __init__(self, *args, **kwargs):
            pass

    class _StubState:
        playing = False

    class _StubWebRtcContext:
        def __init__(self):
            self.video_processor = None
            self.state = _StubState()

    def _stub_webrtc_streamer(*args, **kwargs):  # pragma: no cover - UI glue only
        return _StubWebRtcContext()

    fake_webrtc.WebRtcMode = _StubWebRtcMode
    fake_webrtc.RTCConfiguration = _StubRTCConfiguration
    fake_webrtc.webrtc_streamer = _stub_webrtc_streamer
    sys.modules["streamlit_webrtc"] = fake_webrtc

# `av` is streamlit-webrtc's transitive dependency, imported directly by
# `components/webcam.py` for type hints on VideoFrame. Stub it minimally.
if "av" not in sys.modules:
    fake_av = types.ModuleType("av")

    class _StubVideoFrame:  # pragma: no cover - type-hint stand-in only
        pass

    fake_av.VideoFrame = _StubVideoFrame
    sys.modules["av"] = fake_av


@pytest.fixture(autouse=True)
def clean_session_state():
    """Reset Streamlit's session_state before every test so tests don't leak state."""
    st.session_state.clear()
    yield
    st.session_state.clear()
