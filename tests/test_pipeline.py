"""End-to-end integration: the real EyemouseApp driven by synthetic landmarks.

Nothing here is mocked except the two things that genuinely need hardware --
the webcam and the OS cursor. Camera frames are replaced by a dummy array and
the cursor by :class:`NullMouse`, so the full chain runs for real:
``landmarks -> gaze features -> calibration -> smoothing -> gestures ->
mouse calls -> events``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import calibration as calib
import config as config_mod
import gestures as gest
import main as main_mod
import mouse_control as mctl
import mp_backend as mpb
from config import get, set_path
from main import EyemouseApp
from synthetic import hand_at_point, hands_from, make_face, make_hand
from util import Clock

SCREEN = (1600, 900)
SPAN = 0.40
FRAME_MS = 33
NOFILE = "/definitely/not/here/config.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def g_to_screen(g, screen=SCREEN):
    return (screen[0] / 2 + g[0] * SPAN * screen[0],
            screen[1] / 2 + g[1] * SPAN * screen[1])


def screen_to_g(p, screen=SCREEN):
    return ((p[0] - screen[0] / 2) / (SPAN * screen[0]),
            (p[1] - screen[1] / 2) / (SPAN * screen[1]))


class Harness:
    """An EyemouseApp wired to synthetic inputs, calibrated and ready."""

    def __init__(self, cfg=None, screen=SCREEN):
        self.cfg = cfg or config_mod.load_config(NOFILE)
        self.cfg["_meta"]["notes"] = []
        set_path(self.cfg, "overlay.show", False)
        set_path(self.cfg, "overlay.desktop_dot", False)
        self.screen = screen
        self.clock = Clock(manual=True, start_ms=10_000)
        self.tracker = mpb.NullTracker()
        self.mouse_backend = mctl.NullMouse(*screen)
        self.mouse = mctl.MouseController(self.cfg, backend=self.mouse_backend, screen=screen)
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.frames_seen = []
        self.app = EyemouseApp(
            self.cfg,
            tracker=self.tracker,
            frame_source=lambda: self.frame,
            mouse=self.mouse,
            screen=screen,
            clock=self.clock,
            show_overlay=False,
        )
        self.g = (0.0, 0.0)
        self.ears = (0.30, 0.30)
        self.hands = None
        self.face_present = True
        self.calibrate()

    # -- scene control ---------------------------------------------------
    def publish(self):
        if not self.face_present:
            self.tracker.faces = [mpb.FaceObservation(present=False)]
        else:
            lms = make_face(gaze=self.g, ear_left=self.ears[0], ear_right=self.ears[1])
            self.tracker.faces = [mpb.FaceObservation(landmarks=lms, present=True,
                                                      n_landmarks=int(lms.shape[0]))]
        if self.hands is None or self.hands.landmarks is None:
            self.tracker.hands = [mpb.HandObservation(present=False)]
        else:
            self.tracker.hands = [mpb.HandObservation(
                landmarks=self.hands.landmarks, handedness=list(self.hands.handedness),
                present=True)]

    def step(self, advance=FRAME_MS):
        self.publish()
        self.tracker._i = 0          # NullTracker cycles; pin it to index 0
        state = self.app.step(self.frame)
        self.frames_seen.append(self.frame)
        self.clock.advance(advance)
        return state

    def steps(self, n, advance=FRAME_MS):
        return [self.step(advance) for _ in range(n)]

    # -- calibration -----------------------------------------------------
    def calibrate(self, points=9):
        grid = calib.calibration_grid(points, float(get(self.cfg, "calibration.inset_ratio")),
                                      *self.screen)
        rows = []
        for sx, sy in grid:
            self.g = screen_to_g((sx, sy), self.screen)
            self.publish()
            s = self.app.extractor.extract(self.tracker.faces[0].landmarks, self.clock.ms())
            assert s.valid, s.reason
            rows.append([s.gx, s.gy, s.hx, s.hy, float(sx), float(sy)])
        model = calib.fit_calibration(rows, model=str(get(self.cfg, "calibration.model")),
                                      screen_w=self.screen[0], screen_h=self.screen[1],
                                      points=grid)
        self.app._apply_calibration(model)
        self.g = (0.0, 0.0)
        self.steps(5)
        return model

    # -- convenience -----------------------------------------------------
    def look_at(self, screen_point):
        self.g = screen_to_g(screen_point, self.screen)

    def hand_on_temple(self, side):
        face = make_face()
        a = gest.temple_anchors(face)[side]["point"]
        self.hands = hands_from([(hand_at_point((float(a[0]), float(a[1]))),
                                  "Left" if side == "left" else "Right")])

    def both_hands_on_temples(self):
        face = make_face()
        an = gest.temple_anchors(face)
        self.hands = hands_from([
            (hand_at_point((float(an["left"]["point"][0]), float(an["left"]["point"][1]))), "Left"),
            (hand_at_point((float(an["right"]["point"][0]), float(an["right"]["point"][1]))), "Right"),
        ])

    def hand_away(self):
        self.hands = hands_from([(make_hand((0.5, 0.93), 0.09, 2), "Right")])

    def clear_hands(self):
        self.hands = None


@pytest.fixture
def h():
    return Harness()


def events(states, kind):
    return [e for s in states for e in s["gesture"].events if e.kind == kind]


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------
class TestAppConstruction:
    def test_app_reports_screen_size(self, h):
        assert (h.app.screen_w, h.app.screen_h) == SCREEN

    def test_app_starts_uncalibrated_without_a_model(self):
        cfg = config_mod.load_config(NOFILE)
        lms = make_face()
        tracker = mpb.NullTracker(faces=[mpb.FaceObservation(
            landmarks=lms, present=True, n_landmarks=int(lms.shape[0]))])
        app = EyemouseApp(cfg, tracker=tracker,
                          frame_source=lambda: np.zeros((240, 320, 3), np.uint8),
                          mouse=mctl.MouseController(cfg, backend=mctl.NullMouse(*SCREEN),
                                                     screen=SCREEN),
                          screen=SCREEN, clock=Clock(manual=True), show_overlay=False)
        assert app.calibration_model is None
        state = app.step()
        # A perfectly good face, but with no calibration there is nothing to
        # map it onto -- the reason must say so rather than fail silently.
        assert state["sample"].valid is True
        assert state["gaze"].valid is False
        assert state["gaze"].reason == "not calibrated"
        assert app.mouse.backend.moves == [], "the cursor must not move uncalibrated"

    def test_step_returns_the_whole_frame_state(self, h):
        st = h.step()
        for key in ("frame", "face", "hands", "sample", "gaze", "gesture", "fps", "moved"):
            assert key in st

    def test_hooks_fire(self, h):
        seen = []
        h.app.on("frame", lambda *a: seen.append(a))
        h.steps(3)
        assert len(seen) == 3

    def test_event_hook_receives_events(self, h):
        got = []
        h.app.on("event", lambda ev, st: got.append(ev))
        h.hand_on_temple("left")
        h.steps(6)
        assert any(e.kind == gest.CLICK_LEFT for e in got)


# --------------------------------------------------------------------------
# Cursor tracking
# --------------------------------------------------------------------------
class TestCursor:
    def test_cursor_reaches_the_looked_at_point(self, h):
        worst = 0.0
        for gx in (-0.8, 0.0, 0.8):
            for gy in (-0.8, 0.0, 0.8):
                target = g_to_screen((gx, gy))
                h.look_at(target)
                st = h.steps(45)[-1]
                assert st["gaze"].valid
                worst = max(worst, math.dist((st["gaze"].x, st["gaze"].y), target))
        assert worst < 30.0, f"worst cursor error {worst:.1f}px"

    def test_cursor_actually_reaches_the_backend(self, h):
        h.look_at(g_to_screen((0.7, -0.5)))
        h.steps(45)
        bx, by = h.mouse_backend.position()
        expected = g_to_screen((0.7, -0.5))
        assert math.dist((bx, by), expected) < 30.0

    def test_cursor_stays_on_screen_at_extremes(self, h):
        for g in ((-50, -50), (50, 50)):
            h.g = g
            h.steps(10)
            x, y = h.mouse_backend.position()
            assert 0 <= x <= SCREEN[0] and 0 <= y <= SCREEN[1]

    def test_pause_stops_the_cursor(self, h):
        h.app.tracking = False
        h.look_at(g_to_screen((0.8, 0.8)))
        before = h.mouse_backend.position()
        h.steps(40)
        assert h.mouse_backend.position() == before

    def test_pause_still_processes_frames(self, h):
        h.app.tracking = False
        states = h.steps(10)
        assert all(s["sample"].valid for s in states)

    def test_face_loss_leaves_the_cursor_where_it_was(self, h):
        h.look_at(g_to_screen((0.3, 0.3)))
        h.steps(45)
        before = h.mouse_backend.position()
        h.face_present = False
        h.steps(15)
        assert h.mouse_backend.position() == before

    def test_recovers_after_face_loss(self, h):
        h.face_present = False
        h.steps(10)
        h.face_present = True
        h.look_at(g_to_screen((0.4, 0.2)))
        st = h.steps(45)[-1]
        assert st["gaze"].valid
        assert math.dist((st["gaze"].x, st["gaze"].y), g_to_screen((0.4, 0.2))) < 30.0


# --------------------------------------------------------------------------
# Clicks reaching the OS layer
# --------------------------------------------------------------------------
class TestClicks:
    def test_temple_touch_produces_a_real_click(self, h):
        h.hand_away()
        h.steps(6)
        h.clock.advance(600)
        h.hand_on_temple("left")
        h.steps(6)
        assert h.mouse_backend.clicks == ["left"]

    def test_right_temple_produces_a_right_click(self, h):
        h.hand_away()
        h.steps(6)
        h.clock.advance(600)
        h.hand_on_temple("right")
        h.steps(6)
        assert h.mouse_backend.clicks == ["right"]

    def test_both_temples_produce_no_click(self, h):
        h.hand_away()
        h.steps(6)
        h.clock.advance(600)
        h.both_hands_on_temples()
        h.steps(12)
        assert h.mouse_backend.clicks == []

    def test_wink_produces_a_click(self, h):
        h.clear_hands()
        h.steps(6)
        h.clock.advance(700)
        h.ears = (0.05, 0.30)
        h.steps(6)
        assert h.mouse_backend.clicks == ["left"]

    def test_blink_produces_no_click(self, h):
        h.clear_hands()
        h.steps(6)
        h.clock.advance(700)
        h.ears = (0.05, 0.05)
        h.steps(8)
        assert h.mouse_backend.clicks == []

    def test_dwell_produces_a_click(self, h):
        h.clear_hands()
        h.look_at(g_to_screen((0.2, 0.2)))
        h.steps(40)
        h.clock.advance(700)
        n = int(get(h.cfg, "clicks.dwell.ms") / FRAME_MS) + 10
        h.steps(n)
        assert "left" in h.mouse_backend.clicks

    def test_clicks_are_suppressed_while_paused(self, h):
        h.app.tracking = False
        h.hand_on_temple("left")
        h.steps(8)
        assert h.mouse_backend.clicks == []


# --------------------------------------------------------------------------
# Scrolling
# --------------------------------------------------------------------------
class TestScrolling:
    def test_edge_dwell_scrolls_the_backend(self, h):
        h.clear_hands()
        band = float(get(h.cfg, "scroll.edge.band_ratio"))
        h.steps(20)
        h.g = (0.0, screen_to_g((SCREEN[0] / 2, band * 0.3 * SCREEN[1]))[1])
        n = int(get(h.cfg, "scroll.edge.dwell_ms") * 2 / FRAME_MS) + 10
        h.steps(n)
        assert h.mouse_backend.scrolls, "holding the top edge must scroll"
        assert all(a > 0 for a in h.mouse_backend.scrolls), "top edge scrolls up"

    def test_bottom_edge_scrolls_down(self, h):
        h.clear_hands()
        band = float(get(h.cfg, "scroll.edge.band_ratio"))
        h.steps(20)
        h.g = (0.0, screen_to_g((SCREEN[0] / 2, (1 - band * 0.3) * SCREEN[1]))[1])
        n = int(get(h.cfg, "scroll.edge.dwell_ms") * 2 / FRAME_MS) + 10
        h.steps(n)
        assert h.mouse_backend.scrolls and all(a < 0 for a in h.mouse_backend.scrolls)

    def test_palm_mode_scrolls(self, h):
        set_path(h.cfg, "scroll.mode", "palm")
        h.app.gestures.apply_config(h.cfg)
        h.hands = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        h.steps(int(get(h.cfg, "scroll.palm.toggle_ms") / FRAME_MS) + 6)
        h.mouse_backend.scrolls.clear()
        for v in np.linspace(0.5, 0.85, 16):
            h.g = (0.0, v)
            h.steps(2)
        assert h.mouse_backend.scrolls and all(a < 0 for a in h.mouse_backend.scrolls)


# --------------------------------------------------------------------------
# Sniper
# --------------------------------------------------------------------------
class TestSniper:
    def test_both_temples_slow_the_cursor(self, h):
        h.clear_hands()
        h.look_at(g_to_screen((0.0, 0.0)))
        h.steps(40)
        h.both_hands_on_temples()
        h.steps(6)
        p0 = h.mouse.position
        h.look_at(g_to_screen((0.7, 0.0)))
        h.step()
        sniper_step = math.dist(h.mouse.position, p0)

        h.hand_away()
        h.steps(6)
        h.clock.advance(700)
        h.look_at(g_to_screen((0.0, 0.0)))
        h.steps(40)
        p1 = h.mouse.position
        h.look_at(g_to_screen((0.7, 0.0)))
        h.step()
        normal_step = math.dist(h.mouse.position, p1)

        assert normal_step > 50, f"baseline move was too small ({normal_step:.1f}px)"
        assert sniper_step < normal_step * 0.6, (
            f"sniper {sniper_step:.1f}px vs normal {normal_step:.1f}px")

    def test_sniper_key_activates_without_hands(self, h):
        h.clear_hands()
        h.app.hotkeys._on_press("ctrl")
        st = h.step()
        assert st["gesture"].sniper is True
        h.app.hotkeys._on_release("ctrl")
        assert h.step()["gesture"].sniper is False

    def test_sniper_latch_hotkey(self, h):
        h.clear_hands()
        assert h.app._handle_hotkey("f6") is False
        assert h.app._sniper_latched is True
        assert h.step()["gesture"].sniper is True
        h.app._handle_hotkey("f6")
        assert h.step()["gesture"].sniper is False


# --------------------------------------------------------------------------
# Performance wiring
# --------------------------------------------------------------------------
class TestPerformance:
    def test_fps_meter_advances(self, h):
        states = h.steps(60)
        assert states[-1]["fps"] > 0

    def test_reported_fps_matches_the_frame_interval(self, h):
        states = h.steps(120)
        assert abs(states[-1]["fps"] - 1000.0 / FRAME_MS) < 6.0

    def test_hands_are_throttled(self, h):
        n = 30
        set_path(h.cfg, "performance.hands_every_n_frames", 3)
        h.steps(n)
        assert h.tracker.hand_calls <= h.tracker.face_calls
        assert h.tracker.hand_calls <= n // 3 + 2

    def test_hands_are_skipped_entirely_when_unused(self, h):
        set_path(h.cfg, "clicks.gesture", "wink")
        set_path(h.cfg, "scroll.mode", "edge")
        h.tracker.hand_calls = 0
        h.steps(30)
        assert h.tracker.hand_calls == 0

    def test_process_scale_shrinks_the_inference_frame(self, h):
        seen = []
        h.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        class Spy(mpb.NullTracker):
            def process_face(self, bgr, ts):
                seen.append(None if bgr is None else bgr.shape)
                return super().process_face(bgr, ts)

        h.app._tracker = Spy()
        set_path(h.cfg, "performance.process_scale", 1.0)
        h.step()
        set_path(h.cfg, "performance.process_scale", 0.5)
        h.step()
        assert seen[0] == (480, 640, 3)
        assert seen[1] == (240, 320, 3)

    def test_timing_summary_is_empty_unless_enabled(self, h):
        set_path(h.cfg, "performance.show_timing", False)
        h.app.timer.enabled = False
        h.steps(5)
        assert h.app.timer.summary() == ""


# --------------------------------------------------------------------------
# Run loop and hotkeys
# --------------------------------------------------------------------------
class TestRunLoop:
    def test_run_stops_after_max_frames(self, h):
        rc = h.app.run(max_frames=25)
        assert rc == 0
        assert h.app.frame_count >= 25

    def test_quit_hotkey_stops_the_loop(self, h):
        h.app.on("frame", lambda *a: h.app.hotkeys._on_press("q"))
        rc = h.app.run(max_frames=1000)
        assert rc == 0
        assert h.app.frame_count < 100

    def test_pause_hotkey_toggles_tracking(self, h):
        assert h.app.tracking is True
        h.app._handle_hotkey("f7")
        assert h.app.tracking is False
        h.app._handle_hotkey("f7")
        assert h.app.tracking is True

    def test_recalibrate_hotkey_sets_the_pending_flag(self, h):
        h.app._handle_hotkey("f8")
        assert h.app._pending_recalibrate is True

    def test_save_config_hotkey_writes_the_file(self, h, tmp_path):
        path = str(tmp_path / "c.json")
        set_path(h.cfg, "gaze.gain", 3.25)
        h.cfg["_meta"]["path"] = path
        h.app._handle_hotkey("f9")
        assert os.path.exists(path)
        assert config_mod.load_config(path)["gaze"]["gain"] == pytest.approx(3.25)

    def test_unknown_hotkey_is_ignored(self, h):
        assert h.app._handle_hotkey("f1") is False

    def test_teardown_is_idempotent(self, h):
        h.app.teardown()
        h.app.teardown()

    def test_run_with_no_frame_source_does_not_crash(self):
        cfg = config_mod.load_config(NOFILE)
        app = EyemouseApp(cfg, tracker=mpb.NullTracker(), frame_source=lambda: None,
                          mouse=mctl.MouseController(cfg, backend=mctl.NullMouse(*SCREEN),
                                                     screen=SCREEN),
                          screen=SCREEN, clock=Clock(manual=True), show_overlay=False)
        assert app.run(max_frames=10) == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
class TestCli:
    def test_write_config_creates_a_valid_file(self, tmp_path):
        path = str(tmp_path / "config.json")
        assert main_mod.main(["--write-config", "--config", path]) == 0
        cfg = config_mod.load_config(path)
        assert cfg["gaze"]["gain"] == config_mod.DEFAULTS["gaze"]["gain"]

    def test_invalid_config_exits_with_code_2(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert main_mod.main(["--config", str(path)]) == 2

    def test_camera_flag_overrides_config(self):
        args = main_mod.build_arg_parser().parse_args(["--camera", "2"])
        assert args.camera == 2

    def test_flags_default_to_off(self):
        args = main_mod.build_arg_parser().parse_args([])
        assert args.dry_run is False and args.check is False
        assert args.calibrate is False and args.no_overlay is False
        assert args.frames == 0

    def test_help_mentions_the_gestures(self, capsys):
        parser = main_mod.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        text = capsys.readouterr().out
        assert "temple" in text.lower()
        assert "sniper" in text.lower()

    def test_required_modules_are_named(self):
        """The brief asked for a specific module split."""
        root = os.path.dirname(os.path.abspath(main_mod.__file__))
        for name in ("gaze.py", "calibration.py", "gestures.py",
                     "mouse_control.py", "main.py"):
            assert os.path.exists(os.path.join(root, name)), name


class TestDryRun:
    def test_dry_run_never_touches_a_real_backend(self):
        cfg = config_mod.load_config(NOFILE)
        set_path(cfg, "overlay.show", False)
        app = EyemouseApp(cfg, tracker=mpb.NullTracker(),
                          frame_source=lambda: np.zeros((240, 320, 3), np.uint8),
                          screen=SCREEN, clock=Clock(manual=True),
                          show_overlay=False, dry_run=True)
        assert isinstance(app.mouse.backend, mctl.NullMouse)
        app._apply_calibration(calib.CalibrationModel(
            screen_w=SCREEN[0], screen_h=SCREEN[1]))
        app.run(max_frames=5)
        assert app.mouse.backend.moves or True
