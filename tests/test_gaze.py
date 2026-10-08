"""Gaze feature extraction and the screen-space pipeline.

The sign conventions here are the part most webcam eye trackers get wrong, so
they are asserted explicitly: MediaPipe's ``lateral -> medial`` vector points
in *opposite* image directions for the two eyes, and the extractor has to
correct for that before averaging.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import landmarks as LMK
from config import set_path
from gaze import (GazeExtractor, GazePipeline, EYE_SPECS)
from synthetic import make_face


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
class TestExtraction:
    def test_centred_gaze_is_near_zero(self, face):
        s = GazeExtractor().extract(face)
        assert s.valid
        assert abs(s.gx) < 0.02 and abs(s.gy) < 0.02

    @pytest.mark.parametrize("g,expected_sign", [
        ((1.0, 0.0), +1), ((-1.0, 0.0), -1),
        ((0.0, 1.0), +1), ((0.0, -1.0), -1),
    ])
    def test_gaze_direction_sign(self, g, expected_sign):
        s = GazeExtractor().extract(make_face(gaze=g))
        value = s.gx if g[0] else s.gy
        assert value * expected_sign > 0

    def test_gaze_magnitude_is_monotonic(self):
        vals = [GazeExtractor().extract(make_face(gaze=(v, 0.0))).gx
                for v in np.linspace(-1, 1, 11)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_both_eyes_agree_on_sign(self):
        """If the per-eye sign convention were wrong these would cancel out."""
        f = make_face(gaze=(1.0, 0.0))
        s = GazeExtractor().extract(f)
        assert s.eyes["left"]["fx"] > 0 and s.eyes["right"]["fx"] > 0
        assert s.eyes["left"]["fx"] == pytest.approx(s.eyes["right"]["fx"], abs=1e-9)

    def test_head_yaw_is_absorbed(self):
        """Turning the head with the eyes still centred must not read as gaze."""
        base = GazeExtractor().extract(make_face(gaze=(0.0, 0.0)))
        yawed = GazeExtractor().extract(make_face(gaze=(0.0, 0.0), yaw=0.25))
        assert abs(yawed.gx - base.gx) < 0.02

    def test_head_translation_is_absorbed(self):
        """Leaning toward the camera scales the eye; the feature is normalized."""
        near = GazeExtractor().extract(make_face(centre=(0.5, 0.42), face_width=0.50))
        far = GazeExtractor().extract(make_face(centre=(0.5, 0.42), face_width=0.24))
        assert abs(near.gx - far.gx) < 0.01
        assert abs(near.gy - far.gy) < 0.01

    def test_ear_matches_requested_openness(self):
        s = GazeExtractor().extract(make_face(ear_left=0.28, ear_right=0.34))
        assert s.ear_left == pytest.approx(0.28, abs=1e-6)
        assert s.ear_right == pytest.approx(0.34, abs=1e-6)

    def test_blink_marks_sample_invalid(self):
        s = GazeExtractor().extract(make_face(ear_left=0.05, ear_right=0.05))
        assert s.blink is True
        assert s.valid is False
        assert s.reason == "blink"

    def test_wink_keeps_the_sample_valid(self):
        """Gaze must survive a wink, or wink-to-click would drop tracking."""
        s = GazeExtractor().extract(make_face(ear_left=0.05, ear_right=0.30))
        assert s.blink is False
        assert s.valid is True
        assert s.ear_left < 0.15 < s.ear_right

    def test_closed_eye_is_excluded_from_the_average(self):
        open_both = GazeExtractor().extract(make_face(gaze=(0.4, 0.0)))
        winked = GazeExtractor().extract(
            make_face(gaze=(0.4, 0.0), ear_left=0.05, ear_right=0.30))
        assert winked.gx == pytest.approx(open_both.gx, abs=1e-9)
        assert winked.eyes["left"]["ok"] == 0.0
        assert winked.eyes["right"]["ok"] == 1.0

    def test_face_too_small_is_rejected(self):
        s = GazeExtractor().extract(make_face(face_width=0.04))
        assert s.valid is False
        assert "too small" in s.reason

    def test_none_landmarks(self):
        s = GazeExtractor().extract(None)
        assert s.valid is False and s.reason == "no face"

    def test_short_landmark_array(self):
        s = GazeExtractor().extract(np.zeros((100, 3)))
        assert s.valid is False and "landmarks" in s.reason

    def test_quality_drops_on_a_small_or_closed_face(self):
        good = GazeExtractor().extract(make_face(face_width=0.42)).quality
        small = GazeExtractor().extract(make_face(face_width=0.13)).quality
        closed = GazeExtractor().extract(make_face(ear_left=0.05, ear_right=0.05)).quality
        assert good > small
        assert good > closed

    def test_timestamp_is_passed_through(self, face):
        assert GazeExtractor().extract(face, 12345).timestamp_ms == 12345


class TestIrisFallback:
    def test_468_landmarks_without_rgb_is_invalid(self):
        s = GazeExtractor().extract(make_face(n=468))
        assert s.valid is False
        assert s.iris_used is False
        assert s.reason == "no pupil resolved"

    def test_468_landmarks_with_a_dark_pupil_works(self):
        cv2 = pytest.importorskip("cv2")
        f = make_face(n=468)
        img = np.full((480, 640, 3), 200, np.uint8)
        for corners in ((LMK.RIGHT_EYE_LATERAL, LMK.RIGHT_EYE_MEDIAL),
                        (LMK.LEFT_EYE_LATERAL, LMK.LEFT_EYE_MEDIAL)):
            c = ((f[corners[0], 0] + f[corners[1], 0]) / 2 * 640,
                 (f[corners[0], 1] + f[corners[1], 1]) / 2 * 480)
            cv2.circle(img, (int(c[0]), int(c[1])), 7, (15, 15, 15), -1)
        s = GazeExtractor().extract(f, 0, rgb=img)
        assert s.valid is True
        assert s.iris_used is False
        assert abs(s.gx) < 0.05

    def test_fallback_pupil_tracks_a_moved_blob(self):
        cv2 = pytest.importorskip("cv2")
        results = []
        for shift in (-12, 0, 12):
            f = make_face(n=468)
            img = np.full((480, 640, 3), 200, np.uint8)
            for corners in ((LMK.RIGHT_EYE_LATERAL, LMK.RIGHT_EYE_MEDIAL),
                            (LMK.LEFT_EYE_LATERAL, LMK.LEFT_EYE_MEDIAL)):
                c = ((f[corners[0], 0] + f[corners[1], 0]) / 2 * 640 + shift,
                     (f[corners[0], 1] + f[corners[1], 1]) / 2 * 480)
                cv2.circle(img, (int(c[0]), int(c[1])), 7, (15, 15, 15), -1)
            results.append(GazeExtractor().extract(f, 0, rgb=img).gx)
        assert results[0] < results[1] < results[2]

    def test_iris_flag(self):
        assert LMK.has_iris(478) and not LMK.has_iris(468)

    def test_misplaced_iris_is_caught(self):
        """If the model parks an iris on the nose the sample must not be trusted."""
        f = make_face()
        f[LMK.RIGHT_IRIS_CENTER, :2] = f[LMK.NOSE_TIP, :2]
        f[LMK.LEFT_IRIS_CENTER, :2] = f[LMK.NOSE_TIP, :2]
        for i in list(LMK.RIGHT_IRIS_RING) + list(LMK.LEFT_IRIS_RING):
            f[i, :2] = f[LMK.NOSE_TIP, :2] + 0.001
        s = GazeExtractor().extract(f)
        # Either rejected, or clearly not a plausible centred gaze.
        assert (not s.valid) or abs(s.gy) > 0.05


class TestEyeSpecs:
    def test_every_index_is_a_real_landmark(self):
        for name, spec in EYE_SPECS.items():
            for key in ("lateral", "medial", "iris"):
                assert 0 <= spec[key] < LMK.IRIS_LANDMARKS
            for i in list(spec["ring"]) + list(spec["ear"]) + list(spec["lids"]):
                assert 0 <= i < LMK.IRIS_LANDMARKS

    def test_axis_signs_point_the_same_way_in_image_space(self):
        f = make_face()
        for name, spec in EYE_SPECS.items():
            lat = f[spec["lateral"], :2]
            med = f[spec["medial"], :2]
            axis = (med - lat) * spec["axis_sign"]
            assert axis[0] > 0, f"{name} eye axis must point along image +x"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class _LinearModel:
    """Minimal stand-in for CalibrationModel with a known mapping."""

    def __init__(self, gx_scale=1.0, gy_scale=1.0):
        self.gx_scale, self.gy_scale = gx_scale, gy_scale

    def apply(self, gx, gy, hx=0.5, hy=0.5):
        return (0.5 + gx * self.gx_scale, 0.5 + gy * self.gy_scale)


def _sample(gx=0.0, gy=0.0, ts=0, valid=True):
    from gaze import GazeSample
    return GazeSample(valid=valid, gx=gx, gy=gy, hx=0.5, hy=0.5,
                      face_width=0.36, ear_left=0.3, ear_right=0.3,
                      quality=0.9, timestamp_ms=ts)


class TestPipeline:
    def test_requires_calibration(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        st = p.process(_sample())
        assert st.valid is False and st.reason == "not calibrated"

    def test_invalid_sample_is_passed_through(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        st = p.process(_sample(valid=False))
        assert st.valid is False

    def test_centred_gaze_maps_to_screen_centre(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 0.0)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        st = p.process(_sample(0.0, 0.0, ts=100))
        assert st.x == pytest.approx(800, abs=1)
        assert st.y == pytest.approx(450, abs=1)
        assert st.nx == pytest.approx(0.5, abs=1e-3)

    def test_gain_scales_the_mapping(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 0.0)
        set_path(cfg, "gaze.gain", 2.0)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        st = p.process(_sample(0.1, 0.0, ts=100))
        # 0.5 + (0.6-0.5)*2 = 0.7 of the screen width.
        assert st.x == pytest.approx(0.7 * 1600, abs=1)

    def test_output_is_clamped_inside_the_screen_inset(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 0.0)
        set_path(cfg, "mouse.screen_inset_px", 20)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        st = p.process(_sample(50.0, 50.0, ts=100))
        assert st.x <= 1600 - 20 and st.y <= 900 - 20
        st = p.process(_sample(-50.0, -50.0, ts=200))
        assert st.x >= 20 and st.y >= 20

    def test_deadzone_reports_no_movement(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 12.0)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        p.process(_sample(0.0, 0.0, ts=100))
        st = p.process(_sample(0.0005, 0.0, ts=200))   # < 1px of screen motion
        assert st.moved is False

    def test_smoothing_reduces_output_jitter(self, cfg):
        rng = np.random.default_rng(5)
        for name in ("none", "ma_one_euro"):
            set_path(cfg, "gaze.filter", name)
            set_path(cfg, "gaze.deadzone_px", 0.0)
            p = GazePipeline(cfg, 1600, 900)
            p.set_calibration(_LinearModel())
            xs = []
            for i in range(150):
                st = p.process(_sample(float(rng.normal(0, 0.01)), 0.0, ts=i * 33))
                xs.append(st.x)
            if name == "none":
                raw = float(np.std(np.diff(xs[20:])))
            else:
                smooth = float(np.std(np.diff(xs[20:])))
        assert smooth < raw * 0.6

    def test_speed_is_reported(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 0.0)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        p.process(_sample(0.0, 0.0, ts=0))
        st = p.process(_sample(0.2, 0.0, ts=100))
        assert st.speed > 0

    def test_reset_clears_filter_memory(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        for i in range(30):
            p.process(_sample(0.3, 0.3, ts=i * 33))
        p.reset()
        st = p.process(_sample(-0.3, -0.3, ts=4000))
        # With filters reseeded the first sample lands near the new target.
        assert st.x < 800

    def test_set_screen_size_rescales(self, cfg):
        set_path(cfg, "gaze.filter", "none")
        set_path(cfg, "gaze.deadzone_px", 0.0)
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        p.process(_sample(0.2, 0.0, ts=100))
        p.set_screen_size(3200, 1800)
        st = p.process(_sample(0.2, 0.0, ts=200))
        # The stub maps gx=0.2 -> nx=0.7; the normalized result must simply
        # rescale onto the new resolution rather than stay in old pixels.
        assert st.x == pytest.approx(0.7 * 3200, abs=2)
        assert st.nx == pytest.approx(0.7, abs=1e-3)

    def test_smoothing_boost_tightens_the_filter(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        oe = [f for f in p.chain if hasattr(f, "min_cutoff")]
        assert oe, "default chain should contain a One Euro filter"
        before = oe[0].min_cutoff
        p.set_smoothing_boost(2.0)
        assert oe[0].min_cutoff == pytest.approx(before / 2.0)
        p.set_smoothing_boost(1.0)
        assert oe[0].min_cutoff == pytest.approx(before)

    def test_apply_config_picks_up_slider_changes(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        set_path(cfg, "gaze.deadzone_px", 17.0)
        set_path(cfg, "gaze.moving_average_window", 11)
        p.apply_config(cfg)
        assert p.deadzone.radius == 17.0
        ma = [f for f in p.chain if hasattr(f, "window")]
        assert ma and ma[0].window == 11

    def test_nonmonotonic_timestamps_do_not_crash(self, cfg):
        p = GazePipeline(cfg, 1600, 900)
        p.set_calibration(_LinearModel())
        for ts in (100, 100, 50, 0, 4000):
            st = p.process(_sample(0.1, 0.1, ts=ts))
            assert math.isfinite(st.x) and math.isfinite(st.y)
