"""Overlay rendering and the live-tuning sliders.

The slider range invariant is the important one: every trackbar position must
map onto a value the config validator would accept. A slider that could push
``clicks.debounce_ms`` to 100000 would silently break the app, and it is the
kind of bug that only shows up when a user drags a control to its limit.
"""

from __future__ import annotations

import numpy as np
import pytest

import config as C
from config import get, load_config, set_path
from gaze import GazeSample, GazeState
from gestures import GestureState
from mouse_control import MouseStats
from overlay import SLIDERS, DesktopDot, Overlay
from util import Clock

NOFILE = "/definitely/not/here/config.json"


@pytest.fixture
def cfg():
    c = load_config(NOFILE)
    c["_meta"]["notes"] = []
    return c


# --------------------------------------------------------------------------
# Sliders
# --------------------------------------------------------------------------
class TestSliderSpecs:
    def test_every_slider_targets_a_real_config_key(self, cfg):
        for spec in SLIDERS:
            assert get(cfg, spec.path) is not None, f"{spec.path} is not a config key"

    @pytest.mark.parametrize("spec", SLIDERS, ids=lambda s: s.label)
    def test_trackbar_range_maps_inside_the_valid_config_range(self, spec):
        lo, hi = C.RANGES.get(spec.path, (None, None))
        assert lo is not None, f"{spec.path} has no declared range"
        assert spec.from_trackbar(spec.lo) >= lo - 1e-9
        assert spec.from_trackbar(spec.hi) <= hi + 1e-9

    @pytest.mark.parametrize("spec", SLIDERS, ids=lambda s: s.label)
    def test_trackbar_names_are_unique_and_short(self, spec):
        labels = [s.label for s in SLIDERS]
        assert labels.count(spec.label) == 1
        assert len(spec.label) <= 32, "cv2 truncates long trackbar names"
        assert " " not in spec.label

    @pytest.mark.parametrize("spec", SLIDERS, ids=lambda s: s.label)
    def test_round_trip_at_the_extremes(self, spec):
        for value in (spec.from_trackbar(spec.lo), spec.from_trackbar(spec.hi)):
            assert spec.from_trackbar(spec.to_trackbar(value)) == pytest.approx(value, abs=1e-6)

    def test_to_trackbar_clamps_out_of_range_values(self):
        spec = SLIDERS[0]
        assert spec.to_trackbar(-1e9) == spec.lo
        assert spec.to_trackbar(1e9) == spec.hi

    def test_slider_writes_land_in_config(self, cfg):
        ov = Overlay(cfg)
        spec = next(s for s in SLIDERS if s.path == "gaze.gain")
        ov._on_slider(spec, spec.to_trackbar(2.5))
        assert get(cfg, "gaze.gain") == pytest.approx(2.5, abs=0.02)

    def test_integer_paths_stay_integer(self, cfg):
        ov = Overlay(cfg)
        spec = next(s for s in SLIDERS if s.path == "gaze.moving_average_window")
        ov._on_slider(spec, 7)
        assert isinstance(get(cfg, "gaze.moving_average_window"), int)

    def test_read_sliders_is_a_noop_before_setup(self, cfg):
        ov = Overlay(cfg)
        before = get(cfg, "gaze.gain")
        ov.read_sliders("no-such-window")
        assert get(cfg, "gaze.gain") == before


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _gaze(valid=True, nx=0.5, ny=0.5):
    return GazeState(valid=valid, x=nx * 1600, y=ny * 900, nx=nx, ny=ny, moved=True)


def _frame(w=320, h=240):
    return np.full((h, w, 3), 60, dtype=np.uint8)


def _face():
    from synthetic import make_face
    return make_face()


class TestRender:
    def test_output_width_matches_config(self, cfg):
        ov = Overlay(cfg)
        img = ov.render(_frame(), gaze=_gaze(), gesture=GestureState(), fps=30.0,
                        screen=(1600, 900))
        assert img is not None
        assert img.shape[1] == int(get(cfg, "overlay.width"))

    def test_renders_without_a_camera_frame(self, cfg):
        ov = Overlay(cfg)
        img = ov.render(None, gaze=_gaze(), gesture=GestureState(), fps=0.0,
                        screen=(1600, 900))
        assert img is not None and img.ndim == 3

    def test_renders_with_no_gaze_and_no_gesture(self, cfg):
        ov = Overlay(cfg)
        img = ov.render(_frame(), gaze=None, gesture=None, fps=0.0)
        assert img is not None

    def test_landmark_layers_can_all_be_enabled(self, cfg):
        from synthetic import hands_from, make_hand
        set_path(cfg, "overlay.draw_face", True)
        set_path(cfg, "overlay.draw_iris", True)
        set_path(cfg, "overlay.draw_hands", True)
        set_path(cfg, "overlay.draw_temple_zones", True)
        ov = Overlay(cfg)
        hands = hands_from([(make_hand((0.5, 0.8), 0.09, 3), "Right")])
        img = ov.render(_frame(), gaze=_gaze(), gesture=GestureState(),
                        face=_face(), hands=hands, fps=25.0, screen=(1600, 900))
        assert img is not None

    def test_mirroring_moves_the_gaze_marker(self, cfg):
        """The marker must mirror with the preview or it points the wrong way."""
        def marker_x(mirror):
            set_path(cfg, "camera.mirror_preview", mirror)
            ov = Overlay(cfg)
            ov.render(_frame(), gaze=_gaze(nx=0.1, ny=0.5), gesture=GestureState(),
                      fps=30.0, screen=(1600, 900))
            return ov._last_gaze[0]

        w = int(get(cfg, "overlay.width"))
        direct, mirrored = marker_x(False), marker_x(True)
        assert direct == pytest.approx(0.1 * w, abs=2)
        assert mirrored == pytest.approx(w - 0.1 * w, abs=2)

    def test_hud_can_be_disabled(self, cfg):
        set_path(cfg, "overlay.hud", False)
        ov = Overlay(cfg)
        assert ov.render(_frame(), fps=30.0) is not None

    def test_scroll_bands_are_drawn(self, cfg):
        ov = Overlay(cfg)
        gs = GestureState()
        gs.edge_scroll_dir = -1
        gs.edge_scroll_active = True
        gs.edge_dwell_progress = 0.5
        img = ov.render(_frame(), gaze=_gaze(ny=0.02), gesture=gs, fps=30.0,
                        screen=(1600, 900))
        assert img is not None


class TestFlashAndMessages:
    def test_flash_expires(self, cfg):
        clk = Clock(manual=True)
        ov = Overlay(cfg, clk)
        ov.flash("LEFT CLICK", "left")
        assert len(ov._flashes) == 1
        clk.advance(int(get(cfg, "overlay.flash_ms")) + 10)
        ov._prune()
        assert len(ov._flashes) == 0

    def test_flash_colours_by_kind(self, cfg):
        ov = Overlay(cfg, Clock(manual=True))
        ov.flash("l", "left")
        ov.flash("r", "right")
        ov.flash("s", "scroll")
        colours = {f[2] for f in ov._flashes}
        assert len(colours) == 3

    def test_message_expires(self, cfg):
        clk = Clock(manual=True)
        ov = Overlay(cfg, clk)
        ov.say("hello", ms=500)
        assert ov._messages == ["hello"]
        clk.advance(600)
        ov._prune()
        assert ov._messages == []

    def test_only_the_last_few_messages_are_kept(self, cfg):
        ov = Overlay(cfg, Clock(manual=True))
        for i in range(10):
            ov.say(f"m{i}")
        assert len(ov._messages) == 3
        assert ov._messages[-1] == "m9"

    def test_trail_is_capped_and_cleared_on_invalid_gaze(self, cfg):
        ov = Overlay(cfg)
        for i in range(60):
            ov.render(_frame(), gaze=_gaze(nx=i / 60.0), gesture=GestureState(),
                      fps=30.0, screen=(1600, 900))
        assert len(ov._trail) <= 24
        ov.render(_frame(), gaze=_gaze(valid=False), gesture=GestureState(),
                  fps=30.0, screen=(1600, 900))
        assert len(ov._trail) == 0


class TestDesktopDot:
    def test_disabled_by_default(self, cfg):
        assert get(cfg, "overlay.desktop_dot") is False
        dot = DesktopDot(cfg)
        assert dot.enabled is False
        assert dot.start() is False

    def test_stop_without_start_is_safe(self, cfg):
        dot = DesktopDot(cfg)
        dot.stop()
        dot.stop()

    def test_update_before_start_is_harmless(self, cfg):
        dot = DesktopDot(cfg)
        dot.update(100, 100, "idle")
        assert dot._target == (100, 100, "idle")

    def test_notes_is_a_list(self, cfg):
        assert isinstance(DesktopDot(cfg).notes, list)


class TestStats:
    def test_mouse_stats_defaults(self):
        s = MouseStats()
        assert s.moves == 0 and s.scrolls == 0
        assert s.clicks == {"left": 0, "right": 0}
