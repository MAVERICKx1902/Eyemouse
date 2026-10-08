"""Gesture state machines: temple touches, winks, dwell, scroll, sniper.

Every test drives a manual clock, so debounce/dwell/scroll timing is asserted
exactly rather than with sleeps and tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import landmarks as LMK
from config import set_path
from gaze import GazeSample, GazeState
from gestures import (CLICK_LEFT, CLICK_RIGHT, SCROLL, SCROLL_MODE_OFF,
                      SCROLL_MODE_ON, SNIPER_ON, Debouncer, GestureEngine,
                      HoldCounter, eye_aspect_ratio, palm_openness,
                      temple_anchors)
from synthetic import FakeHands, hand_at_point, hands_from, make_face, make_hand

SCREEN = (1600, 900)
FRAME_MS = 33


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sample(ear_left=0.30, ear_right=0.30, blink=None, ts=0):
    if blink is None:
        blink = ear_left < 0.19 and ear_right < 0.19
    return GazeSample(valid=not blink, gx=0.0, gy=0.0, hx=0.5, hy=0.5,
                      face_width=0.36, ear_left=ear_left, ear_right=ear_right,
                      blink=blink, quality=0.9, timestamp_ms=ts)


def gaze(nx=0.5, ny=0.5, valid=True, screen=SCREEN):
    return GazeState(valid=valid, x=nx * screen[0], y=ny * screen[1],
                     nx=nx, ny=ny, moved=True)


def face_at(centre=(0.5, 0.42), width=0.36):
    return make_face(centre=centre, face_width=width)


def hand_on(face, side):
    a = temple_anchors(face)[side]["point"]
    return hands_from([(hand_at_point((float(a[0]), float(a[1]))),
                        "Left" if side == "left" else "Right")])


def both_hands_on(face):
    la = temple_anchors(face)["left"]["point"]
    ra = temple_anchors(face)["right"]["point"]
    return hands_from([
        (hand_at_point((float(la[0]), float(la[1]))), "Left"),
        (hand_at_point((float(ra[0]), float(ra[1]))), "Right"),
    ])


HAND_AWAY = hands_from([(make_hand((0.5, 0.93), 0.09, 2), "Right")])
NO_HANDS = FakeHands()


def kinds(states, kind):
    return [e for st in states for e in st.events if e.kind == kind]


def run(engine, clock, n, *, smp=None, gz=None, face=None, hands=NO_HANDS,
        sniper_key=False, advance=FRAME_MS):
    out = []
    for _ in range(n):
        out.append(engine.update(smp if smp is not None else sample(),
                                 gz if gz is not None else gaze(),
                                 face, hands, sniper_key=sniper_key))
        clock.advance(advance)
    return out


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
class TestDebouncer:
    def test_first_call_always_fires(self, clock):
        d = Debouncer(500, clock)
        assert d.fire("left") is True

    def test_blocks_inside_the_window(self, clock):
        d = Debouncer(500, clock)
        d.fire("left")
        clock.advance(499)
        assert d.fire("left") is False

    def test_allows_after_the_window(self, clock):
        d = Debouncer(500, clock)
        d.fire("left")
        clock.advance(500)
        assert d.fire("left") is True

    def test_keys_are_independent(self, clock):
        d = Debouncer(500, clock)
        d.fire("left")
        clock.advance(10)
        assert d.fire("right") is True, "a right click must not wait out a left click"

    def test_zero_ms_never_blocks(self, clock):
        d = Debouncer(0, clock)
        for _ in range(5):
            assert d.fire("left") is True

    def test_remaining_counts_down(self, clock):
        d = Debouncer(500, clock)
        d.fire("left")
        clock.advance(200)
        assert d.remaining("left") == 300

    def test_reset(self, clock):
        d = Debouncer(500, clock)
        d.fire("left")
        d.reset("left")
        assert d.fire("left") is True


class TestHoldCounter:
    def test_needs_consecutive_frames(self):
        h = HoldCounter(3)
        assert h.update(True) is False
        assert h.update(True) is False
        assert h.update(True) is True

    def test_a_gap_resets(self):
        h = HoldCounter(3)
        h.update(True); h.update(True)
        assert h.update(False) is False
        assert h.count == 0
        assert h.update(True) is False

    def test_needed_one_fires_immediately(self):
        assert HoldCounter(1).update(True) is True


class TestGeometry:
    def test_temple_anchors_scale_with_face_width(self):
        small = temple_anchors(make_face(face_width=0.20))
        big = temple_anchors(make_face(face_width=0.50))
        # The anchor separation must track the face width, which is what makes
        # the touch radius distance-invariant.
        d_small = float(np.linalg.norm(small["left"]["point"] - small["right"]["point"]))
        d_big = float(np.linalg.norm(big["left"]["point"] - big["right"]["point"]))
        assert d_big / d_small == pytest.approx(0.50 / 0.20, rel=0.01)

    def test_temple_anchors_are_left_and_right_of_centre(self):
        a = temple_anchors(make_face(centre=(0.5, 0.42)))
        assert a["left"]["point"][0] > 0.5 > a["right"]["point"][0]

    def test_temple_anchors_reject_a_short_array(self):
        assert temple_anchors(np.zeros((100, 3))) == {}
        assert temple_anchors(None) == {}

    @pytest.mark.parametrize("ear", [0.05, 0.15, 0.30, 0.45])
    def test_ear_matches_synthetic_openness(self, ear):
        f = make_face(ear_right=ear)
        assert eye_aspect_ratio(f, LMK.RIGHT_EYE_EAR) == pytest.approx(ear, abs=1e-6)

    def test_ear_zero_on_degenerate_eye(self):
        f = make_face()
        f[LMK.RIGHT_EYE_LATERAL, :2] = f[LMK.RIGHT_EYE_MEDIAL, :2]
        assert eye_aspect_ratio(f, LMK.RIGHT_EYE_EAR) == 0.0

    def test_palm_openness_counts(self):
        assert palm_openness(make_hand((0.5, 0.5), 0.10, 4))[0] == 4
        assert palm_openness(make_hand((0.5, 0.5), 0.10, 2))[0] == 2
        assert palm_openness(make_hand((0.5, 0.5), 0.10, 0))[0] == 0

    def test_open_palm_and_fist_flags(self):
        _, is_open, is_fist = palm_openness(make_hand((0.5, 0.5), 0.10, 4))
        assert is_open and not is_fist
        _, is_open, is_fist = palm_openness(make_hand((0.5, 0.5), 0.10, 0))
        assert is_fist and not is_open

    def test_palm_openness_is_scale_invariant(self):
        """Same hand at twice the distance must read the same."""
        big = palm_openness(make_hand((0.5, 0.5), 0.20, 4))[0]
        small = palm_openness(make_hand((0.5, 0.5), 0.05, 4))[0]
        assert big == small == 4

    def test_palm_openness_rejects_bad_input(self):
        assert palm_openness(None) == (0, False, False)
        assert palm_openness(np.zeros((10, 3))) == (0, False, False)


# --------------------------------------------------------------------------
# temple clicks
# --------------------------------------------------------------------------
class TestTemple:
    def engine(self, cfg, clock):
        return GestureEngine(cfg, clock)

    def test_left_temple_gives_a_left_click(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        assert len(kinds(st, CLICK_LEFT)) == 1
        assert kinds(st, CLICK_RIGHT) == []
        assert kinds(st, CLICK_LEFT)[0].source == "temple"

    def test_right_temple_gives_a_right_click(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 6, face=f, hands=hand_on(f, "right"))
        assert len(kinds(st, CLICK_RIGHT)) == 1
        assert kinds(st, CLICK_LEFT) == []

    def test_hand_away_from_the_temple_does_nothing(self, cfg, clock):
        e = self.engine(cfg, clock)
        st = run(e, clock, 20, face=face_at(), hands=HAND_AWAY)
        assert kinds(st, CLICK_LEFT) == [] and kinds(st, CLICK_RIGHT) == []

    def test_holding_does_not_auto_repeat(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 90, face=f, hands=hand_on(f, "left"))
        assert len(kinds(st, CLICK_LEFT)) == 1, "a held touch must fire once"

    def test_release_and_retouch_fires_again(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        clock.advance(600)
        run(e, clock, 6, face=f, hands=HAND_AWAY)
        st = run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_hold_frames_suppress_single_frame_noise(self, cfg, clock):
        set_path(cfg, "clicks.temple.hold_frames", 3)
        e = self.engine(cfg, clock)
        f = face_at()
        # Two frames on, one off, repeatedly: never reaches 3 consecutive.
        st = []
        for i in range(30):
            hands = hand_on(f, "left") if i % 3 != 2 else HAND_AWAY
            st.append(e.update(sample(), gaze(), f, hands))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_both_temples_give_sniper_and_no_clicks(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 12, face=f, hands=both_hands_on(f))
        assert kinds(st, CLICK_LEFT) == [] and kinds(st, CLICK_RIGHT) == []
        assert len(kinds(st, SNIPER_ON)) == 1
        assert st[-1].sniper is True
        assert st[-1].temple_both is True

    def test_sniper_ends_when_a_hand_is_dropped(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        run(e, clock, 8, face=f, hands=both_hands_on(f))
        st = run(e, clock, 8, face=f, hands=hand_on(f, "left"))
        assert st[-1].sniper is False

    def test_sniper_can_be_disabled(self, cfg, clock):
        set_path(cfg, "clicks.temple.both_temples_sniper", False)
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 12, face=f, hands=both_hands_on(f))
        assert st[-1].sniper is False
        # With sniper off, both temples are treated as two single touches --
        # but the "no simultaneous left+right" rule must still hold.
        assert not (kinds(st, CLICK_LEFT) and kinds(st, CLICK_RIGHT))

    def test_sniper_key_activates_without_hands(self, cfg, clock):
        e = self.engine(cfg, clock)
        st = run(e, clock, 5, face=face_at(), hands=NO_HANDS, sniper_key=True)
        assert st[-1].sniper is True

    def test_touch_radius_scales_with_distance(self, cfg, clock):
        """Far from the camera the face is small; the radius must shrink too."""
        e = self.engine(cfg, clock)
        near = face_at(width=0.50)
        far = face_at(width=0.20)
        run(e, clock, 3, face=near, hands=NO_HANDS)
        r_near = e._temple_radius
        run(e, clock, 3, face=far, hands=NO_HANDS)
        r_far = e._temple_radius
        assert r_far / r_near == pytest.approx(0.20 / 0.50, rel=0.05)

    def test_missing_face_is_handled(self, cfg, clock):
        e = self.engine(cfg, clock)
        st = run(e, clock, 5, face=None, hands=HAND_AWAY)
        assert kinds(st, CLICK_LEFT) == []
        assert st[-1].temple_left is False

    def test_touch_distance_is_reported_for_the_hud(self, cfg, clock):
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 3, face=f, hands=hand_on(f, "left"))
        assert st[-1].touch_distance < st[-1].touch_radius
        st = run(e, clock, 3, face=f, hands=HAND_AWAY)
        assert st[-1].touch_distance > st[-1].touch_radius

    def test_strict_handedness_requires_a_matching_label(self, cfg, clock):
        set_path(cfg, "clicks.temple.handedness_mode", "strict")
        e = self.engine(cfg, clock)
        f = face_at()
        a = temple_anchors(f)["left"]["point"]
        wrong = hands_from([(hand_at_point((float(a[0]), float(a[1]))), "Right")])
        st = run(e, clock, 8, face=f, hands=wrong)
        assert kinds(st, CLICK_LEFT) == []
        st = run(e, clock, 8, face=f, hands=hand_on(f, "left"))
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_swap_handedness_flips_strict_matching(self, cfg, clock):
        set_path(cfg, "clicks.temple.handedness_mode", "strict")
        set_path(cfg, "clicks.temple.swap_handedness", True)
        e = self.engine(cfg, clock)
        f = face_at()
        a = temple_anchors(f)["left"]["point"]
        # Labelled "Right", but swap makes it count as the subject's left hand.
        swapped = hands_from([(hand_at_point((float(a[0]), float(a[1]))), "Right")])
        st = run(e, clock, 8, face=f, hands=swapped)
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_temple_gesture_can_be_disabled(self, cfg, clock):
        set_path(cfg, "clicks.gesture", "wink")
        e = self.engine(cfg, clock)
        f = face_at()
        st = run(e, clock, 10, face=f, hands=hand_on(f, "left"))
        assert kinds(st, CLICK_LEFT) == []


# --------------------------------------------------------------------------
# winks
# --------------------------------------------------------------------------
class TestWink:
    def test_left_wink_gives_a_left_click(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        f = face_at()
        st = []
        for i in range(6):
            s = sample(ear_left=0.05, ear_right=0.30)
            st.append(e.update(s, gaze(), f, NO_HANDS))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_LEFT)) == 1
        assert kinds(st, CLICK_LEFT)[0].source == "wink"

    def test_right_wink_gives_a_right_click(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(6):
            st.append(e.update(sample(0.30, 0.05), gaze(), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_RIGHT)) == 1
        assert kinds(st, CLICK_LEFT) == []

    def test_blink_does_not_click(self, cfg, clock):
        """The single most important wink false-positive test."""
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(8):
            st.append(e.update(sample(0.05, 0.05), gaze(valid=False), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == [] and kinds(st, CLICK_RIGHT) == []
        assert st[-1].blink is True

    def test_repeated_natural_blinking_produces_no_clicks(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(8):
            for _ in range(3):     # eyes shut ~100ms
                st.append(e.update(sample(0.06, 0.06), gaze(valid=False), face_at(), NO_HANDS))
                clock.advance(FRAME_MS)
            for _ in range(12):    # eyes open ~400ms
                st.append(e.update(sample(0.30, 0.30), gaze(), face_at(), NO_HANDS))
                clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == [] and kinds(st, CLICK_RIGHT) == []

    def test_holding_a_wink_does_not_repeat(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(60):
            st.append(e.update(sample(0.05, 0.30), gaze(), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_long_wink_is_treated_as_an_eye_rub(self, cfg, clock):
        """A wink held past max_duration_ms must be swallowed."""
        set_path(cfg, "clicks.wink.max_duration_ms", 300)
        set_path(cfg, "clicks.wink.hold_frames", 30)   # only confirms after 1s
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(40):
            st.append(e.update(sample(0.05, 0.30), gaze(), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_wink_can_be_disabled(self, cfg, clock):
        set_path(cfg, "clicks.gesture", "temple")
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(8):
            st.append(e.update(sample(0.05, 0.30), gaze(), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_both_gestures_enabled(self, cfg, clock):
        set_path(cfg, "clicks.gesture", "both")
        e = GestureEngine(cfg, clock)
        f = face_at()
        st = []
        for _ in range(6):
            st.append(e.update(sample(0.05, 0.30), gaze(), f, NO_HANDS))
            clock.advance(FRAME_MS)
        clock.advance(600)
        for _ in range(6):
            st.append(e.update(sample(0.30, 0.30), gaze(), f, hand_on(f, "left")))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_LEFT)) == 2
        sources = {e_.source for e_ in kinds(st, CLICK_LEFT)}
        assert sources == {"wink", "temple"}

    def test_no_sample_does_not_crash(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = e.update(None, gaze(valid=False), face_at(), NO_HANDS)
        assert st.wink_left is False and st.blink is False


# --------------------------------------------------------------------------
# dwell
# --------------------------------------------------------------------------
class TestDwell:
    def dwell_ms(self, cfg):
        return int(cfg["clicks"]["dwell"]["ms"])

    def test_dwelling_fires_a_left_click(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) / FRAME_MS) + 6
        st = run(e, clock, n, gz=gaze(0.5, 0.5))
        clicks = kinds(st, CLICK_LEFT)
        assert len(clicks) == 1 and clicks[0].source == "dwell"

    def test_no_click_before_the_dwell_time(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) / FRAME_MS) - 6
        st = run(e, clock, max(1, n), gz=gaze(0.5, 0.5))
        assert kinds(st, CLICK_LEFT) == []

    def test_moving_cancels_the_dwell(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for i in range(80):
            # Drift steadily: never inside the radius long enough.
            nx = 0.2 + i * 0.006
            st.append(e.update(sample(), gaze(nx, 0.5), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_jitter_inside_the_radius_still_dwells(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        r = float(cfg["clicks"]["dwell"]["radius_px"])
        rng = np.random.default_rng(11)
        st = []
        n = int(self.dwell_ms(cfg) / FRAME_MS) + 10
        for _ in range(n):
            jx = 0.5 + float(rng.normal(0, r * 0.2)) / SCREEN[0]
            jy = 0.5 + float(rng.normal(0, r * 0.2)) / SCREEN[1]
            st.append(e.update(sample(), gaze(jx, jy), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_dwell_does_not_repeat_in_place(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) / FRAME_MS) + 6
        st = run(e, clock, n * 3, gz=gaze(0.5, 0.5))
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_leaving_the_radius_rearms(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) / FRAME_MS) + 6
        run(e, clock, n, gz=gaze(0.5, 0.5))
        run(e, clock, 20, gz=gaze(0.2, 0.2))       # move away
        st = run(e, clock, n, gz=gaze(0.5, 0.5))    # come back and dwell
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_dwell_progress_ramps_to_one(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = run(e, clock, int(self.dwell_ms(cfg) / FRAME_MS) + 4, gz=gaze(0.5, 0.5))
        progs = [s.dwell_progress for s in st]
        assert progs[0] < progs[-2]
        assert max(progs) == pytest.approx(1.0, abs=0.02)

    def test_long_dwell_right_click(self, cfg, clock):
        set_path(cfg, "clicks.dwell.right_click_ms", 2000)
        e = GestureEngine(cfg, clock)
        st = run(e, clock, int((self.dwell_ms(cfg) + 2000) / FRAME_MS) + 12, gz=gaze(0.5, 0.5))
        assert len(kinds(st, CLICK_LEFT)) == 1
        assert len(kinds(st, CLICK_RIGHT)) == 1

    def test_dwell_can_be_disabled(self, cfg, clock):
        set_path(cfg, "clicks.dwell.enabled", False)
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 120, gz=gaze(0.5, 0.5))
        assert kinds(st, CLICK_LEFT) == []
        assert all(s.dwell_progress == 0.0 for s in st)

    def test_dwell_suppressed_at_the_screen_edge(self, cfg, clock):
        guard = float(cfg["clicks"]["dwell"]["edge_guard_ratio"])
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 120, gz=gaze(0.5, guard * 0.3))
        assert kinds(st, CLICK_LEFT) == []
        assert any("dwell:edge_guard" in s for x in st for s in x.suppressed)

    def test_dwell_suppressed_by_a_temple_touch(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        f = face_at()
        st = run(e, clock, 120, gz=gaze(0.5, 0.5), face=f, hands=hand_on(f, "left"))
        # The temple click happens; the dwell must not stack on top of it.
        assert len(kinds(st, CLICK_LEFT)) == 1
        assert kinds(st, CLICK_LEFT)[0].source == "temple"

    def test_dwell_suppressed_by_a_wink(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(120):
            st.append(e.update(sample(0.05, 0.30), gaze(0.5, 0.5), face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert len(kinds(st, CLICK_LEFT)) == 1
        assert kinds(st, CLICK_LEFT)[0].source == "wink"

    def test_dwell_suppressed_while_blinking(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = []
        for _ in range(120):
            st.append(e.update(sample(0.05, 0.05), gaze(0.5, 0.5, valid=False),
                               face_at(), NO_HANDS))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_invalid_gaze_resets_the_dwell(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        half = int(self.dwell_ms(cfg) / FRAME_MS) - 4
        run(e, clock, half, gz=gaze(0.5, 0.5))
        run(e, clock, 5, gz=gaze(0.5, 0.5, valid=False))
        st = run(e, clock, half + 2, gz=gaze(0.5, 0.5))
        assert kinds(st, CLICK_LEFT) == [], "losing the face must restart the dwell timer"


# --------------------------------------------------------------------------
# edge scroll
# --------------------------------------------------------------------------
class TestEdgeScroll:
    def band(self, cfg):
        return float(cfg["scroll"]["edge"]["band_ratio"])

    def dwell_ms(self, cfg):
        return int(cfg["scroll"]["edge"]["dwell_ms"])

    def test_no_scroll_before_the_dwell(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 0.5 / FRAME_MS)
        st = run(e, clock, n, gz=gaze(0.5, self.band(cfg) * 0.4))
        assert kinds(st, SCROLL) == []

    def test_top_edge_scrolls_up(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 2 / FRAME_MS) + 8
        st = run(e, clock, n, gz=gaze(0.5, self.band(cfg) * 0.3))
        events = kinds(st, SCROLL)
        assert events, "holding the top edge must scroll"
        assert all(ev.amount > 0 for ev in events), "positive = up in pyautogui"
        assert st[-1].edge_scroll_dir == -1
        assert st[-1].edge_scroll_active is True

    def test_bottom_edge_scrolls_down(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 2 / FRAME_MS) + 8
        st = run(e, clock, n, gz=gaze(0.5, 1 - self.band(cfg) * 0.3))
        events = kinds(st, SCROLL)
        assert events and all(ev.amount < 0 for ev in events)
        assert st[-1].edge_scroll_dir == 1

    def test_leaving_the_band_stops_scrolling(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 2 / FRAME_MS) + 8
        run(e, clock, n, gz=gaze(0.5, self.band(cfg) * 0.3))
        st = run(e, clock, 30, gz=gaze(0.5, 0.5))
        assert kinds(st, SCROLL) == []
        assert st[-1].edge_scroll_active is False

    def test_switching_edges_resets_the_dwell(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 0.7 / FRAME_MS)
        run(e, clock, n, gz=gaze(0.5, self.band(cfg) * 0.3))
        st = run(e, clock, 3, gz=gaze(0.5, 1 - self.band(cfg) * 0.3))
        assert kinds(st, SCROLL) == [], "the bottom-edge timer must restart from zero"

    def test_deeper_penetration_scrolls_faster(self, cfg, clock):
        set_path(cfg, "scroll.edge.accelerate", False)
        set_path(cfg, "scroll.edge.repeat_ms", FRAME_MS)
        band = self.band(cfg)

        def total(pen):
            # pen is the fraction of the way *into* the band: pen=1.0 sits
            # exactly on the screen edge, pen=0.0 just barely crosses in.
            from util import Clock as _Clock
            clock2 = _Clock(manual=True, start_ms=10_000)
            e = GestureEngine(cfg, clock2)
            ny = band * (1.0 - pen)
            st = run(e, clock2, int(self.dwell_ms(cfg) / FRAME_MS) + 25, gz=gaze(0.5, ny))
            return sum(abs(ev.amount) for ev in kinds(st, SCROLL))

        deep, shallow = total(0.95), total(0.15)
        assert deep > shallow, f"deep={deep} shallow={shallow}"

    def test_amount_is_bounded(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 200, gz=gaze(0.5, 0.0))
        mx = int(cfg["scroll"]["edge"]["max_amount"]) * 3
        assert all(abs(ev.amount) <= mx for ev in kinds(st, SCROLL))
        assert all(ev.amount != 0 for ev in kinds(st, SCROLL))

    def test_invalid_gaze_does_not_scroll(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 120, gz=gaze(0.5, 0.0, valid=False))
        assert kinds(st, SCROLL) == []

    def test_edge_scroll_suppresses_dwell_click(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        n = int(self.dwell_ms(cfg) * 2 / FRAME_MS) + int(cfg["clicks"]["dwell"]["ms"] / FRAME_MS) + 20
        st = run(e, clock, n, gz=gaze(0.5, self.band(cfg) * 0.3))
        assert kinds(st, SCROLL), "scrolling should have started"
        assert kinds(st, CLICK_LEFT) == [], "and it must not also dwell-click"

    def test_scroll_can_be_disabled(self, cfg, clock):
        set_path(cfg, "scroll.mode", "none")
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 120, gz=gaze(0.5, 0.0))
        assert kinds(st, SCROLL) == []


# --------------------------------------------------------------------------
# palm scroll
# --------------------------------------------------------------------------
class TestPalmScroll:
    @pytest.fixture
    def pcfg(self, cfg):
        set_path(cfg, "scroll.mode", "palm")
        return cfg

    def toggle_frames(self, cfg):
        return int(cfg["scroll"]["palm"]["toggle_ms"] / FRAME_MS) + 4

    def test_open_palm_enters_scroll_mode(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        st = run(e, clock, self.toggle_frames(pcfg), hands=palm)
        assert len(kinds(st, SCROLL_MODE_ON)) == 1
        assert st[-1].scroll_mode is True

    def test_brief_palm_flash_does_not_toggle(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        fist = hands_from([(make_hand((0.5, 0.75), 0.11, 0), "Right")])
        run(e, clock, 2, hands=palm)
        st = run(e, clock, 10, hands=fist)
        assert kinds(st, SCROLL_MODE_ON) == []
        assert st[-1].scroll_mode is False

    def test_fist_exits_scroll_mode(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        fist = hands_from([(make_hand((0.5, 0.75), 0.11, 0), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        clock.advance(600)
        st = run(e, clock, 8, hands=fist)
        assert len(kinds(st, SCROLL_MODE_OFF)) >= 1
        assert st[-1].scroll_mode is False

    def test_looking_down_scrolls_down(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        # Sweep from the gaze position used during the toggle (0.5) downwards,
        # so every delta is genuinely downward.
        st = []
        for v in np.linspace(0.5, 0.85, 16):
            st.append(e.update(sample(), gaze(0.5, float(v)), face_at(), palm))
            clock.advance(FRAME_MS)
        events = kinds(st, SCROLL)
        assert events and all(ev.amount < 0 for ev in events)

    def test_looking_up_scrolls_up(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        st = []
        for v in np.linspace(0.5, 0.15, 16):
            st.append(e.update(sample(), gaze(0.5, float(v)), face_at(), palm))
            clock.advance(FRAME_MS)
        events = kinds(st, SCROLL)
        assert events and all(ev.amount > 0 for ev in events)

    def test_small_eye_movements_do_not_scroll(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        st = run(e, clock, 40, gz=gaze(0.5, 0.5), hands=palm)
        assert kinds(st, SCROLL) == [], "a stationary gaze must not scroll"

    def test_clicks_are_suppressed_in_scroll_mode(self, pcfg, clock):
        """Scroll mode must not be able to race a click."""
        e = GestureEngine(pcfg, clock)
        f = face_at()
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        a = temple_anchors(f)["left"]["point"]
        both = hands_from([(hand_at_point((float(a[0]), float(a[1]))), "Left"),
                           (make_hand((0.5, 0.75), 0.11, 4), "Right")])
        st = run(e, clock, 20, face=f, hands=both)
        assert kinds(st, CLICK_LEFT) == []
        assert "temple" in st[-1].suppressed

    def test_wink_is_suppressed_in_scroll_mode(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        st = []
        for _ in range(20):
            st.append(e.update(sample(0.05, 0.30), gaze(), face_at(), palm))
            clock.advance(FRAME_MS)
        assert kinds(st, CLICK_LEFT) == []

    def test_idle_timeout_exits_scroll_mode(self, pcfg, clock):
        set_path(pcfg, "scroll.palm.exit_after_ms", 500)
        e = GestureEngine(pcfg, clock)
        palm = hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])
        run(e, clock, self.toggle_frames(pcfg), hands=palm)
        st = run(e, clock, int(900 / FRAME_MS), gz=gaze(0.5, 0.5), hands=palm)
        assert st[-1].scroll_mode is False
        assert kinds(st, SCROLL_MODE_OFF)

    def test_edge_scroll_not_used_in_palm_mode(self, pcfg, clock):
        e = GestureEngine(pcfg, clock)
        st = run(e, clock, 120, gz=gaze(0.5, 0.0), hands=NO_HANDS)
        assert kinds(st, SCROLL) == []


# --------------------------------------------------------------------------
# engine plumbing
# --------------------------------------------------------------------------
class TestEngine:
    def test_reset_clears_latches(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        f = face_at()
        run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        e.reset()
        clock.advance(int(cfg["clicks"]["debounce_ms"]) + 50)
        st = run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        # After a reset the still-held touch counts as a fresh rising edge.
        assert len(kinds(st, CLICK_LEFT)) == 1

    def test_reset_preserves_the_debounce_window(self, cfg, clock):
        """Deliberate: reset() runs after calibration, and a user whose hand
        is still on their temple must not get a click for it."""
        e = GestureEngine(cfg, clock)
        f = face_at()
        run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        e.reset()
        st = run(e, clock, 6, face=f, hands=hand_on(f, "left"))
        assert kinds(st, CLICK_LEFT) == []

    def test_apply_config_updates_debounce(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        set_path(cfg, "clicks.debounce_ms", 1234)
        e.apply_config(cfg)
        assert e.debounce.ms == 1234

    def test_apply_config_updates_hold_frames(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        set_path(cfg, "clicks.temple.hold_frames", 7)
        set_path(cfg, "clicks.wink.hold_frames", 5)
        e.apply_config(cfg)
        assert e._hold_temple_l.needed == 7
        assert e._hold_wink_l.needed == 5

    def test_works_with_no_face_and_no_hands(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        st = run(e, clock, 10, face=None, hands=None, gz=gaze(valid=False))
        assert st[-1].hands_visible == 0
        assert st[-1].temple_left is False

    def test_gesture_state_defaults_are_inert(self):
        from gestures import GestureState
        s = GestureState()
        assert s.events == [] and s.suppressed == []
        assert s.sniper is False and s.scroll_mode is False

    def test_event_repr(self):
        from gestures import Event
        assert "click_left" in repr(Event(CLICK_LEFT, source="temple"))
        assert "amount=3" in repr(Event(SCROLL, amount=3))

    def test_hands_visible_count(self, cfg, clock):
        e = GestureEngine(cfg, clock)
        f = face_at()
        two = hands_from([(make_hand((0.2, 0.9), 0.08, 2), "Left"),
                          (make_hand((0.8, 0.9), 0.08, 2), "Right")])
        st = run(e, clock, 3, face=f, hands=two)
        assert st[-1].hands_visible == 2
