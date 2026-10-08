"""MediaPipe access layer: probing, model discovery, neutral containers.

Real inference is not exercised (it needs the model bundles and a webcam), but
the *failure modes* are -- those are what a user hits on first run, and the
error text has to tell them what to do.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mp_backend as mpb


class TestProbe:
    def test_probe_returns_a_report(self):
        info = mpb.probe()
        for key in ("installed", "version", "api", "errors"):
            assert key in info

    @pytest.mark.skipif(not os.path.exists(
        os.path.join(os.path.dirname(mpb.__file__))), reason="mediapipe not importable")
    def test_api_is_recognised_when_mediapipe_is_present(self):
        info = mpb.probe()
        if info["installed"]:
            assert info["api"] in ("tasks", "solutions", None)
            if info["api"] is None:
                assert info["errors"], "a None api must explain itself"

    def test_tasks_backend_reports_iris_support(self):
        """The Tasks face landmarker emits 478 landmarks (with iris)."""
        obs = mpb.FaceObservation(landmarks=np.zeros((478, 3)), present=True, n_landmarks=478)
        assert obs.has_iris
        assert not mpb.FaceObservation(n_landmarks=468).has_iris


class TestModelDiscovery:
    def test_missing_models_report_none(self, tmp_path):
        found = mpb.find_models(str(tmp_path))
        assert found[mpb.FACE_MODEL_NAME] is None
        assert found[mpb.HAND_MODEL_NAME] is None

    def test_finds_models_in_the_models_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        (d / mpb.FACE_MODEL_NAME).write_bytes(b"x" * 4096)
        (d / mpb.HAND_MODEL_NAME).write_bytes(b"x" * 4096)
        found = mpb.find_models(str(tmp_path))
        assert found[mpb.FACE_MODEL_NAME] == str(d / mpb.FACE_MODEL_NAME)
        assert found[mpb.HAND_MODEL_NAME] == str(d / mpb.HAND_MODEL_NAME)

    def test_finds_models_in_the_project_root(self, tmp_path):
        (tmp_path / mpb.FACE_MODEL_NAME).write_bytes(b"x" * 4096)
        (tmp_path / mpb.HAND_MODEL_NAME).write_bytes(b"x" * 4096)
        found = mpb.find_models(str(tmp_path))
        assert all(v is not None for v in found.values())

    def test_truncated_file_is_rejected(self, tmp_path):
        """A stub left behind by an interrupted download must not look valid."""
        d = tmp_path / "models"
        d.mkdir()
        (d / mpb.FACE_MODEL_NAME).write_bytes(b"tiny")
        found = mpb.find_models(str(tmp_path))
        assert found[mpb.FACE_MODEL_NAME] is None

    def test_models_dir_path(self):
        assert mpb.model_dir("/x").endswith(mpb.MODEL_DIRNAME)

    def test_urls_are_https_and_named_consistently(self):
        for name, url in mpb.MODEL_URLS.items():
            assert url.startswith("https://")
            assert name in url
        assert mpb.FACE_MODEL_NAME in mpb.MODEL_URLS
        assert mpb.HAND_MODEL_NAME in mpb.MODEL_URLS


class TestCreateTracker:
    def test_missing_models_produce_an_actionable_error(self, tmp_path):
        """The single most common first-run failure must say how to fix it."""
        info = mpb.probe()
        if not info["installed"] or info["api"] != "tasks":
            pytest.skip("needs the Tasks API with models absent")
        if all(v is not None for v in mpb.find_models(str(tmp_path)).values()):
            pytest.skip("models unexpectedly present")
        with pytest.raises(mpb.TrackerSetupError) as ei:
            mpb.create_tracker({}, root=str(tmp_path))
        msg = str(ei.value)
        assert "download_models" in msg
        assert "models" in msg.lower()

    def test_error_mentions_both_model_files(self, tmp_path):
        info = mpb.probe()
        if not info["installed"] or info["api"] != "tasks":
            pytest.skip("needs the Tasks API")
        with pytest.raises(mpb.TrackerSetupError) as ei:
            mpb.create_tracker({}, root=str(tmp_path))
        assert mpb.FACE_MODEL_NAME in str(ei.value)
        assert mpb.HAND_MODEL_NAME in str(ei.value)


class TestContainers:
    def test_face_observation_defaults_are_absent(self):
        obs = mpb.FaceObservation()
        assert obs.present is False and obs.landmarks is None and obs.n_landmarks == 0

    def test_hand_observation_count(self):
        assert mpb.HandObservation().count == 0
        obs = mpb.HandObservation(landmarks=np.zeros((2, 21, 3)), present=True)
        assert obs.count == 2

    def test_hand_handedness_defaults_to_empty(self):
        assert mpb.HandObservation().handedness == []


class TestNullTracker:
    def test_cycles_supplied_observations(self):
        faces = [mpb.FaceObservation(landmarks=np.zeros((478, 3)), present=True,
                                     n_landmarks=478) for _ in range(3)]
        t = mpb.NullTracker(faces=faces)
        seen = [t.process_face(None, i * 33) for i in range(6)]
        assert len(seen) == 6
        assert all(o.present for o in seen)

    def test_empty_tracker_reports_no_face(self):
        t = mpb.NullTracker()
        assert t.process_face(None, 0).present is False
        assert t.process_hands(None, 0).present is False

    def test_timestamp_is_written_through(self):
        t = mpb.NullTracker(faces=[mpb.FaceObservation(present=True)])
        assert t.process_face(None, 4242).timestamp_ms == 4242


class TestColourConversion:
    def test_channels_are_swapped(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 10    # B
        bgr[..., 1] = 20    # G
        bgr[..., 2] = 30    # R
        rgb = mpb.bgr_to_rgb(bgr)
        assert rgb[0, 0, 0] == 30 and rgb[0, 0, 2] == 10

    def test_output_is_contiguous_uint8(self):
        rgb = mpb.bgr_to_rgb(np.zeros((4, 4, 3), dtype=np.uint8))
        assert rgb.dtype == np.uint8
        assert rgb.flags["C_CONTIGUOUS"]

    def test_shape_is_preserved(self):
        assert mpb.bgr_to_rgb(np.zeros((5, 7, 3), np.uint8)).shape == (5, 7, 3)


class TestMonotonicTimestamps:
    def test_stalled_timestamps_are_pushed_forward(self):
        """VIDEO running mode rejects non-increasing timestamps."""
        f = mpb.TasksBackend._mono
        assert f(-1, 100) == 100
        assert f(100, 100) == 101
        assert f(101, 50) == 102
        assert f(102, 9000) == 9000
