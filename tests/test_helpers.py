"""Tests for utils/helpers.py"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from utils import helpers


class FakeUploadedFile(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile (adds a .name and .getvalue rewind)."""

    def __init__(self, data: bytes, name: str = "test.png"):
        super().__init__(data)
        self.name = name

    def getvalue(self) -> bytes:  # noqa: D102 - matches Streamlit's UploadedFile API
        return self.getbuffer().tobytes()


def _make_fake_image_bytes(width: int, height: int, fmt: str = "PNG") -> bytes:
    array = (np.random.rand(height, width, 3) * 255).astype("uint8")
    pil_image = Image.fromarray(array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return buffer.getvalue()


class TestResizeIfNeeded:
    def test_no_resize_when_within_bounds(self):
        image = np.zeros((100, 200, 3), dtype="uint8")
        result = helpers.resize_if_needed(image, max_dim=1600)
        assert result.shape == image.shape

    def test_resizes_when_exceeding_bounds(self):
        image = np.zeros((1000, 2000, 3), dtype="uint8")
        result = helpers.resize_if_needed(image, max_dim=1000)
        height, width = result.shape[:2]
        assert max(height, width) == 1000
        # Aspect ratio should be preserved (2:1)
        assert round(width / height) == 2


class TestHumanReadableSize:
    @pytest.mark.parametrize(
        "num_bytes,expected_unit",
        [(500, "B"), (2048, "KB"), (5 * 1024 * 1024, "MB"), (3 * 1024**3, "GB")],
    )
    def test_returns_expected_unit(self, num_bytes, expected_unit):
        result = helpers.human_readable_size(num_bytes)
        assert expected_unit in result


class TestReadUploadedImage:
    def test_valid_image_decodes_to_bgr_array(self):
        fake_file = FakeUploadedFile(_make_fake_image_bytes(64, 48))
        image_bgr = helpers.read_uploaded_image(fake_file)
        assert isinstance(image_bgr, np.ndarray)
        assert image_bgr.shape == (48, 64, 3)

    def test_empty_file_raises_value_error(self):
        fake_file = FakeUploadedFile(b"")
        with pytest.raises(ValueError, match="empty"):
            helpers.read_uploaded_image(fake_file)

    def test_garbage_bytes_raises_value_error(self):
        fake_file = FakeUploadedFile(b"this is definitely not an image")
        with pytest.raises(ValueError):
            helpers.read_uploaded_image(fake_file)


class TestEnsureProjectDirectories:
    def test_creates_all_required_dirs(self, tmp_path):
        helpers.ensure_project_directories(tmp_path)
        for dirname in helpers.REQUIRED_DIRS:
            assert (tmp_path / dirname).is_dir()

    def test_is_idempotent(self, tmp_path):
        helpers.ensure_project_directories(tmp_path)
        helpers.ensure_project_directories(tmp_path)  # should not raise
        assert (tmp_path / "models").is_dir()


class TestSaveUploadedVideo:
    def test_saves_video_bytes_to_disk(self, tmp_path):
        fake_file = FakeUploadedFile(b"fake video bytes" * 100, name="clip.mp4")
        path = helpers.save_uploaded_video(fake_file, uploads_dir=tmp_path)
        assert path.exists()
        assert path.read_bytes() == fake_file.getvalue()

    def test_empty_video_raises_value_error(self, tmp_path):
        fake_file = FakeUploadedFile(b"", name="clip.mp4")
        with pytest.raises(ValueError, match="empty"):
            helpers.save_uploaded_video(fake_file, uploads_dir=tmp_path)

    def test_oversized_video_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(helpers, "MAX_VIDEO_SIZE_MB", 0.0001)
        fake_file = FakeUploadedFile(b"x" * 10_000, name="clip.mp4")
        with pytest.raises(ValueError, match="exceeds"):
            helpers.save_uploaded_video(fake_file, uploads_dir=tmp_path)
