"""Gesture recognition: temple clicks, winks, dwell, scroll, sniper.

All inputs are plain numpy arrays and all timing goes through an injectable
:class:`util.Clock`, so every state machine below can be driven by synthetic
data in tests without a webcam, MediaPipe or a display.

Conflict arbitration
--------------------
Several gestures can be true at once, so they are resolved in a fixed order
(see :meth:`GestureEngine.update`):

1. **Both temples** -> sniper mode and *no* click. Without this rule a
   two-handed temple touch would fire a left and a right click at once.
2. **Palm scroll mode** -> suppresses dwell and temple clicks, so scrolling
   can never race a click.
3. **Edge scroll active** -> suppresses dwell (you are deliberately parked at
   the screen edge, not trying to click).
4. **Wink or temple touch active** -> suppresses dwell.

Every click gesture is *edge triggered* and *latched*: it fires once on the
transition into the gesture and cannot fire again until the gesture is
released. The debounce timer sits on top of that as a second line of defence
against a twitchy detector. Debouncing is per-button, so a left click can be
followed immediately by a right click.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import landmarks as LMK
from config import get
from util import Clock, clamp

CLICK_LEFT = "click_left"
CLICK_RIGHT = "click_right"
SCROLL = "scroll"
SCROLL_MODE_ON = "scroll_mode_on"
SCROLL_MODE_OFF = "scroll_mode_off"
SNIPER_ON = "sniper_on"
SNIPER_OFF = "sniper_off"

_EPS = 1e-9

__all__ = [
    "Event", "GestureState", "GestureEngine", "Debouncer", "HoldCounter",
    "temple_anchors", "eye_aspect_ratio", "palm_openness",
    "CLICK_LEFT", "CLICK_RIGHT", "SCROLL", "SCROLL_MODE_ON", "SCROLL_MODE_OFF",
    "SNIPER_ON", "SNIPER_OFF",
]


# ==========================================================================
# Small state helpers
# ==========================================================================
class Debouncer:
    """Rate limiter keyed by action name (per-button click debounce)."""

    def __init__(self, ms: int = 500, clock: Optional[Clock] = None) -> None:
        self.ms = int(ms)
        self._clock = clock or Clock()
        self._last: Dict[str, int] = {}

    def ready(self, key: str) -> bool:
        if self.ms <= 0:
            return True
        last = self._last.get(key)
        return last is None or (self._clock.ms() - last) >= self.ms

    def fire(self, key: str) -> bool:
        """Record an event if the window has elapsed. True when it fired."""
        if not self.ready(key):
            return False
        self._last[key] = self._clock.ms()
        return True

    def remaining(self, key: str) -> int:
        last = self._last.get(key)
        if last is None or self.ms <= 0:
            return 0
        return max(0, self.ms - (self._clock.ms() - last))

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)


class HoldCounter:
    """Requires N consecutive true frames; resets on the first false."""

    def __init__(self, needed: int = 2) -> None:
        self.needed = max(1, int(needed))
        self.count = 0

    def update(self, active: bool) -> bool:
        self.count = self.count + 1 if active else 0
        return self.count >= self.needed

    @property
    def satisfied(self) -> bool:
        return self.count >= self.needed

    def reset(self) -> None:
        self.count = 0


# ==========================================================================
# Geometry primitives
# ==========================================================================
def temple_anchors(face: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """Temple anchor points plus a distance-invariant face width.

    Anchors are the mean of a small landmark cluster rather than a single
    index, so one jittery landmark cannot move the touch target. The radius is
    derived from the 234<->454 face width, which makes the gesture behave the
    same whether the user sits 40 cm or 120 cm from the camera.
    """
    if face is None or face.shape[0] < LMK.BASE_LANDMARKS:
        return {}
    right = face[list(LMK.RIGHT_TEMPLE), :2].mean(axis=0)
    left = face[list(LMK.LEFT_TEMPLE), :2].mean(axis=0)
    a = face[LMK.FACE_WIDTH_PAIR[0], :2]
    b = face[LMK.FACE_WIDTH_PAIR[1], :2]
    width = float(np.linalg.norm(a - b))
    return {
        # Subject's right temple -> right click; subject's left -> left click.
        "right": {"point": right, "width": width},
        "left": {"point": left, "width": width},
    }


def eye_aspect_ratio(face: Optional[np.ndarray], eye_indices: Sequence[int]) -> float:
    """Classic 6-point EAR: ``(|p2-p6| + |p3-p5|) / (2|p1-p4|)``."""
    if face is None or face.shape[0] <= max(eye_indices):
        return 0.0
    p = face[list(eye_indices), :2]
    horiz = float(np.linalg.norm(p[0] - p[3]))
    if horiz < _EPS:
        return 0.0
    vert = 0.5 * (float(np.linalg.norm(p[1] - p[5])) + float(np.linalg.norm(p[2] - p[4])))
    return vert / (2.0 * horiz)


def palm_openness(hand: Optional[np.ndarray], open_fingers: int = 4
                  ) -> Tuple[int, bool, bool]:
    """Count extended fingers -> ``(extended, is_open_palm, is_fist)``.

    A finger counts as extended when its tip is meaningfully further from the
    wrist than its PIP joint. The thumb is excluded: its tip stays near the
    wrist even on an open hand, which would make the test flap. Both terms are
    ratios of the hand span, so the test is scale invariant.
    """
    if hand is None or hand.shape[0] < 21:
        return 0, False, False
    wrist = hand[LMK.WRIST, :2]
    span = float(np.linalg.norm(hand[LMK.MIDDLE_PIP, :2] - wrist))
    if span < _EPS:
        return 0, False, False
    extended = 0
    for tip, pip in LMK.PALM_FINGERS:
        d_tip = float(np.linalg.norm(hand[tip, :2] - wrist))
        d_pip = float(np.linalg.norm(hand[pip, :2] - wrist))
        if d_tip > d_pip * 1.18 and d_tip > span * 0.55:
            extended += 1
    return extended, extended >= int(open_fingers), extended <= 1


# ==========================================================================
# Outputs
# ==========================================================================
@dataclass
class Event:
    kind: str
    amount: int = 0
    source: str = ""
    t: int = 0

    def __repr__(self) -> str:
        extra = f" amount={self.amount}" if self.amount else ""
        return f"<Event {self.kind}{extra} via {self.source}>"


@dataclass
class GestureState:
    """Everything the overlay and the mouse controller need for one frame."""

    # Confirmed detections (after the hold counter).
    temple_left: bool = False
    temple_right: bool = False
    temple_both: bool = False
    # Raw, single-frame touch flags for the HUD.
    temple_left_raw: bool = False
    temple_right_raw: bool = False
    touch_distance: float = float("inf")
    touch_radius: float = 0.0
    wink_left: bool = False
    wink_right: bool = False
    blink: bool = False
    hands_visible: int = 0
    open_palm: bool = False
    fist: bool = False
    extended_fingers: int = 0

    # Modes
    sniper: bool = False
    scroll_mode: bool = False
    edge_scroll_dir: int = 0        # -1 up, 0 idle, +1 down
    edge_scroll_active: bool = False
    edge_dwell_progress: float = 0.0
    dwell_progress: float = 0.0
    dwell_armed: bool = False

    # Emitters
    events: List[Event] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)


# ==========================================================================
# Engine
# ==========================================================================
class GestureEngine:
    """Consumes one frame of landmarks + gaze and emits debounced intents."""

    def __init__(self, cfg: Dict[str, Any], clock: Optional[Clock] = None) -> None:
        self.cfg = cfg
        self.clock = clock or Clock()
        self.debounce = Debouncer(int(get(cfg, "clicks.debounce_ms", 500)), self.clock)

        hold_t = int(get(cfg, "clicks.temple.hold_frames", 2))
        hold_w = int(get(cfg, "clicks.wink.hold_frames", 2))
        self._hold_temple_l = HoldCounter(hold_t)
        self._hold_temple_r = HoldCounter(hold_t)
        self._hold_wink_l = HoldCounter(hold_w)
        self._hold_wink_r = HoldCounter(hold_w)
        self._hold_palm = HoldCounter(2)
        self._hold_fist = HoldCounter(2)

        # Latches: each click gesture fires once and must be released first.
        self._temple_latched = {"left": False, "right": False}
        self._wink_latched = {"left": False, "right": False}
        # When each eye first closed. None rather than 0 so a clock that
        # starts at zero cannot be mistaken for "not started".
        self._wink_raw_since: Dict[str, Optional[int]] = {"left": None, "right": None}

        self._temple_smooth = {"left": np.zeros(2), "right": np.zeros(2)}
        self._temple_init = False
        self._temple_radius = 0.0

        self._dwell_anchor: Optional[Tuple[float, float]] = None
        self._dwell_since = 0
        self._dwell_fired_left = False
        self._dwell_fired_right = False
        self._dwell_refractory = False

        self._edge_dir = 0
        self._edge_in_band_since = 0
        self._edge_active = False
        self._edge_last_emit = 0
        self._edge_accel = 1.0

        self.scroll_mode = False
        self._palm_open_since: Optional[int] = None
        # After an idle-timeout exit the palm must be lowered and raised again
        # before scroll mode re-arms; otherwise a continuously open hand would
        # toggle the mode back on the very next frame and it would flap.
        self._palm_rearm = False
        self._scroll_last_y: Optional[float] = None
        self._scroll_accum = 0.0
        self._scroll_last_event = 0
        self._scroll_last_activity = 0

        self.sniper = False

    # ------------------------------------------------------------------
    def apply_config(self, cfg: Dict[str, Any]) -> None:
        """Live-update tunables (called by the slider overlay)."""
        self.cfg = cfg
        self.debounce.ms = int(get(cfg, "clicks.debounce_ms", 500))
        self._hold_temple_l.needed = max(1, int(get(cfg, "clicks.temple.hold_frames", 2)))
        self._hold_temple_r.needed = self._hold_temple_l.needed
        self._hold_wink_l.needed = max(1, int(get(cfg, "clicks.wink.hold_frames", 2)))
        self._hold_wink_r.needed = self._hold_wink_l.needed

    def reset(self) -> None:
        """Drop all latched state (after calibration, or a tracking gap)."""
        for h in (self._hold_temple_l, self._hold_temple_r, self._hold_wink_l,
                  self._hold_wink_r, self._hold_palm, self._hold_fist):
            h.reset()
        self._temple_latched = {"left": False, "right": False}
        self._wink_latched = {"left": False, "right": False}
        self._wink_raw_since = {"left": None, "right": None}
        self._dwell_anchor = None
        self._reset_dwell_flags()
        self._edge_dir, self._edge_active, self._edge_accel = 0, False, 1.0
        self.scroll_mode = False
        self._palm_open_since = None
        self._palm_rearm = False
        self._scroll_last_y = None
        self._scroll_accum = 0.0
        self.sniper = False

    def _reset_dwell_flags(self) -> None:
        self._dwell_since = 0
        self._dwell_fired_left = False
        self._dwell_fired_right = False
        self._dwell_refractory = False

    # ------------------------------------------------------------------
    def _hand_array(self, hands) -> Optional[np.ndarray]:
        lms = getattr(hands, "landmarks", None) if hands is not None else None
        if lms is None:
            return None
        arr = np.asarray(lms, dtype=np.float64)
        return arr if arr.ndim == 3 and arr.shape[0] > 0 else None

    # ------------------------------------------------------------------
    def _detect_temples(self, face: Optional[np.ndarray], hands, st: GestureState) -> None:
        hands_lms = self._hand_array(hands)
        anchors = temple_anchors(face) if face is not None else {}
        if not anchors:
            self._hold_temple_l.update(False)
            self._hold_temple_r.update(False)
            self._temple_latched = {"left": False, "right": False}
            st.touch_radius = self._temple_radius
            return

        width = max(float(anchors["left"]["width"]), _EPS)
        # EMA-smooth the anchors: they are the reference frame for the whole
        # gesture, and raw face landmarks jitter by a pixel or two.
        ema = 1.0 if not self._temple_init else 0.35
        for side in ("left", "right"):
            self._temple_smooth[side] = (
                (1.0 - ema) * self._temple_smooth[side] + ema * anchors[side]["point"]
            )
        self._temple_init = True

        radius = float(get(self.cfg, "clicks.temple.radius_ratio", 0.34)) * width
        self._temple_radius = radius
        st.touch_radius = radius

        mode = str(get(self.cfg, "clicks.temple.handedness_mode", "temple_side")).lower()
        swap = bool(get(self.cfg, "clicks.temple.swap_handedness", False))
        labels = list(getattr(hands, "handedness", []) or []) if hands is not None else []

        touched = {"left": False, "right": False}
        best_dist = float("inf")

        if hands_lms is not None:
            st.hands_visible = int(hands_lms.shape[0])
            for hi in range(hands_lms.shape[0]):
                hand = hands_lms[hi]
                if hand.shape[0] < 21:
                    continue
                label = labels[hi] if hi < len(labels) else ""
                if swap:
                    label = {"Left": "Right", "Right": "Left"}.get(label, label)
                tips = hand[list(LMK.TOUCH_TIPS), :2]
                for side in ("left", "right"):
                    anchor = self._temple_smooth[side]
                    d = float(np.min(np.linalg.norm(tips - anchor[None, :], axis=1)))
                    if d < best_dist:
                        best_dist = d
                    if d <= radius:
                        if mode == "strict":
                            # Require the MediaPipe handedness label to agree:
                            # the subject's LEFT hand at the LEFT temple.
                            if label and label.lower() == side:
                                touched[side] = True
                        else:
                            touched[side] = True
        st.touch_distance = best_dist

        st.temple_left_raw = touched["left"]
        st.temple_right_raw = touched["right"]
        st.temple_left = self._hold_temple_l.update(touched["left"])
        st.temple_right = self._hold_temple_r.update(touched["right"])
        st.temple_both = st.temple_left and st.temple_right

    # ------------------------------------------------------------------
    def _detect_winks(self, sample, st: GestureState) -> Tuple[bool, bool]:
        cfg = self.cfg
        thr = float(get(cfg, "clicks.wink.ear_threshold", 0.20))
        asym = float(get(cfg, "clicks.wink.asymmetry_ratio", 0.60))
        open_min = float(get(cfg, "clicks.wink.open_eye_min", 0.22))
        max_ms = int(get(cfg, "clicks.wink.max_duration_ms", 900))
        now = self.clock.ms()

        if sample is None:
            self._hold_wink_l.update(False)
            self._hold_wink_r.update(False)
            self._wink_latched = {"left": False, "right": False}
            self._wink_raw_since = {"left": None, "right": None}
            return False, False

        el = float(getattr(sample, "ear_left", 0.0))
        er = float(getattr(sample, "ear_right", 0.0))
        st.blink = bool(getattr(sample, "blink", False)) or (el < thr and er < thr)

        # A wink is one eye closed while the other is verifiably open and
        # clearly more open. The asymmetry test is what separates a wink from
        # a blink, which is the main false-positive source for this gesture.
        raw_l = (el < thr) and (er > open_min) and (el < asym * er)
        raw_r = (er < thr) and (el > open_min) and (er < asym * el)

        # Record when each eye first closed, so the duration guard below has
        # something meaningful to measure against.
        for side, raw in (("left", raw_l), ("right", raw_r)):
            if raw:
                if self._wink_raw_since[side] is None:
                    self._wink_raw_since[side] = now
            else:
                self._wink_raw_since[side] = None

        held_l = self._hold_wink_l.update(raw_l)
        held_r = self._hold_wink_r.update(raw_r)
        st.wink_left, st.wink_right = held_l, held_r

        fired = {"left": False, "right": False}
        for side, held in (("left", held_l), ("right", held_r)):
            if held:
                if not self._wink_latched[side]:
                    # Age is measured from when the eye first *closed*, not
                    # from when the hold counter confirmed the wink. Measuring
                    # from confirmation would always give ~0 and make the
                    # guard a no-op; measuring from closure rejects a wink
                    # that was only confirmed long after the eye shut (e.g.
                    # the face was lost and regained mid-wink), which would
                    # otherwise click from stale state.
                    started = self._wink_raw_since[side]
                    age = (now - started) if started is not None else 0
                    if age <= max_ms:
                        fired[side] = True
                    self._wink_latched[side] = True
            else:
                self._wink_latched[side] = False
        return fired["left"], fired["right"]

    # ------------------------------------------------------------------
    def _detect_palm(self, hands, st: GestureState) -> None:
        need = int(get(self.cfg, "scroll.palm.open_fingers", 4))
        hands_lms = self._hand_array(hands)
        best_ext, any_open, any_fist = 0, False, False
        if hands_lms is not None:
            for hi in range(hands_lms.shape[0]):
                ext, is_open, is_fist = palm_openness(hands_lms[hi], need)
                best_ext = max(best_ext, ext)
                any_open = any_open or is_open
                any_fist = any_fist or is_fist
        st.extended_fingers = best_ext
        st.open_palm = self._hold_palm.update(any_open)
        st.fist = self._hold_fist.update(any_fist)

    # ------------------------------------------------------------------
    def _update_dwell(self, gx: float, gy: float, valid: bool, allowed: bool,
                      st: GestureState) -> None:
        cfg = self.cfg
        if not bool(get(cfg, "clicks.dwell.enabled", True)):
            st.dwell_progress = 0.0
            st.dwell_armed = False
            return
        dwell_ms = max(1, int(get(cfg, "clicks.dwell.ms", 1400)))
        radius = float(get(cfg, "clicks.dwell.radius_px", 24))
        right_ms = int(get(cfg, "clicks.dwell.right_click_ms", 0))
        now = self.clock.ms()

        if not valid or not allowed:
            st.dwell_progress = 0.0
            st.dwell_armed = False
            self._dwell_anchor = None
            self._reset_dwell_flags()
            return

        if self._dwell_anchor is None:
            self._dwell_anchor = (gx, gy)
            self._dwell_since = now
            self._dwell_refractory = False

        ax, ay = self._dwell_anchor
        if math.hypot(gx - ax, gy - ay) > radius:
            # Left the radius: re-anchor. Leaving is also what clears the
            # post-click refractory period, so the same spot cannot re-fire.
            self._dwell_anchor = (gx, gy)
            self._dwell_since = now
            self._dwell_fired_left = False
            self._dwell_fired_right = False
            self._dwell_refractory = False
            st.dwell_progress = 0.0
            st.dwell_armed = True
            return

        if self._dwell_refractory:
            st.dwell_progress = 1.0
            st.dwell_armed = False
            return

        elapsed = now - self._dwell_since
        st.dwell_armed = True

        if not self._dwell_fired_left:
            st.dwell_progress = clamp(elapsed / float(dwell_ms), 0.0, 1.0)
            if elapsed >= dwell_ms:
                self._dwell_fired_left = True
                if self.debounce.fire("dwell_left"):
                    st.events.append(Event(CLICK_LEFT, source="dwell", t=now))
                # Restart the clock so the right-click dwell is measured from
                # the moment the left click landed.
                self._dwell_since = now
                st.dwell_progress = 0.0
                if right_ms <= 0:
                    self._dwell_refractory = True
            return

        if right_ms > 0 and not self._dwell_fired_right:
            st.dwell_progress = clamp(elapsed / float(right_ms), 0.0, 1.0)
            if elapsed >= right_ms:
                self._dwell_fired_right = True
                self._dwell_refractory = True
                if self.debounce.fire("dwell_right"):
                    st.events.append(Event(CLICK_RIGHT, source="dwell", t=now))
        else:
            st.dwell_progress = 1.0
            self._dwell_refractory = True

    # ------------------------------------------------------------------
    def _update_edge_scroll(self, ny: float, valid: bool, allowed: bool,
                            st: GestureState) -> None:
        cfg = self.cfg
        band = float(get(cfg, "scroll.edge.band_ratio", 0.06))
        dwell_ms = max(1, int(get(cfg, "scroll.edge.dwell_ms", 800)))
        repeat_ms = max(1, int(get(cfg, "scroll.edge.repeat_ms", 70)))
        min_amt = int(get(cfg, "scroll.edge.min_amount", 1))
        max_amt = max(min_amt, int(get(cfg, "scroll.edge.max_amount", 8)))
        accel = bool(get(cfg, "scroll.edge.accelerate", True))
        now = self.clock.ms()

        direction, depth = 0, 0.0
        if valid and allowed and band > 0:
            if ny <= band:
                direction, depth = -1, (band - ny) / band
            elif ny >= 1.0 - band:
                direction, depth = 1, (ny - (1.0 - band)) / band
        st.edge_scroll_dir = direction

        if direction != self._edge_dir:
            self._edge_dir = direction
            self._edge_in_band_since = now
            self._edge_active = False
            self._edge_accel = 1.0
        held = now - self._edge_in_band_since if direction else 0
        st.edge_dwell_progress = clamp(held / float(dwell_ms), 0.0, 1.0)

        if direction == 0 or held < dwell_ms:
            self._edge_active = False
            st.edge_scroll_active = False
            return

        self._edge_active = True
        st.edge_scroll_active = True
        if accel:
            self._edge_accel = min(3.0, self._edge_accel + 0.02)
        if now - self._edge_last_emit >= repeat_ms:
            self._edge_last_emit = now
            base = min_amt + depth * (max_amt - min_amt)
            amount = int(round(clamp(base * self._edge_accel, min_amt, max_amt * 3)))
            # pyautogui.scroll(): positive scrolls up, negative scrolls down.
            st.events.append(Event(SCROLL, amount=-amount if direction > 0 else amount,
                                   source="edge", t=now))

    # ------------------------------------------------------------------
    def _update_palm_scroll(self, gy_px: float, valid: bool, st: GestureState) -> None:
        cfg = self.cfg
        toggle_ms = max(1, int(get(cfg, "scroll.palm.toggle_ms", 400)))
        px_per_click = max(1.0, float(get(cfg, "scroll.palm.px_per_click", 55.0)))
        max_clicks = max(1, int(get(cfg, "scroll.palm.max_clicks_per_event", 8)))
        exit_after = int(get(cfg, "scroll.palm.exit_after_ms", 6000))
        now = self.clock.ms()

        # --- enter / exit ------------------------------------------------
        if st.open_palm:
            if self._palm_open_since is None:
                self._palm_open_since = now
        else:
            self._palm_open_since = None
            self._palm_rearm = False      # hand lowered: scroll mode may re-arm

        if not self.scroll_mode and not self._palm_rearm:
            if self._palm_open_since is not None and (now - self._palm_open_since) >= toggle_ms:
                self.scroll_mode = True
                self._scroll_last_y = gy_px if valid else None
                self._scroll_accum = 0.0
                self._scroll_last_activity = now
                st.events.append(Event(SCROLL_MODE_ON, source="palm", t=now))
        elif st.fist:
            self.scroll_mode = False
            self._scroll_last_y = None
            self._scroll_accum = 0.0
            st.events.append(Event(SCROLL_MODE_OFF, source="palm", t=now))

        st.scroll_mode = self.scroll_mode
        if not self.scroll_mode:
            return

        # --- drive scrolling from vertical gaze movement -----------------
        if valid:
            if self._scroll_last_y is not None:
                delta = gy_px - self._scroll_last_y
                if abs(delta) > 0.05:
                    self._scroll_accum += delta
                    self._scroll_last_activity = now
            self._scroll_last_y = gy_px

        if now - self._scroll_last_activity > exit_after:
            self.scroll_mode = False
            st.scroll_mode = False
            self._scroll_last_y = None
            self._scroll_accum = 0.0
            self._palm_rearm = True       # require the hand to drop first
            st.events.append(Event(SCROLL_MODE_OFF, source="palm_timeout", t=now))
            return

        if abs(self._scroll_accum) >= px_per_click:
            clicks = int(clamp(self._scroll_accum / px_per_click, -max_clicks, max_clicks))
            if clicks != 0 and now - self._scroll_last_event >= 30:
                self._scroll_last_event = now
                # Looking down (accum > 0) scrolls the page down.
                st.events.append(Event(SCROLL, amount=-clicks, source="palm", t=now))
                self._scroll_accum -= clicks * px_per_click

    # ------------------------------------------------------------------
    def update(self, sample, gaze, face: Optional[np.ndarray], hands,
               sniper_key: bool = False) -> GestureState:
        """Process one frame.

        ``sample`` is a :class:`gaze.GazeSample` (or None), ``gaze`` a
        :class:`gaze.GazeState` (or None), ``face`` an ``(N,3)`` landmark
        array and ``hands`` a hand-observation object (or None).
        """
        cfg = self.cfg
        st = GestureState()
        now = self.clock.ms()
        click_mode = str(get(cfg, "clicks.gesture", "both")).lower()
        scroll_cfg = str(get(cfg, "scroll.mode", "edge")).lower()
        use_temple = click_mode in ("temple", "both")
        use_wink = click_mode in ("wink", "both")
        both_temples_sniper = bool(get(cfg, "clicks.temple.both_temples_sniper", True))

        # ---- detection -------------------------------------------------
        if use_temple:
            self._detect_temples(face, hands, st)
        if scroll_cfg == "palm":
            self._detect_palm(hands, st)
        fired_wink_l = fired_wink_r = False
        if use_wink:
            fired_wink_l, fired_wink_r = self._detect_winks(sample, st)

        # ---- sniper (both temples, or the hold key) ---------------------
        want_sniper = bool(sniper_key) or (both_temples_sniper and st.temple_both)
        if want_sniper != self.sniper:
            self.sniper = want_sniper
            st.events.append(Event(SNIPER_ON if want_sniper else SNIPER_OFF,
                                   source="sniper", t=now))
        st.sniper = self.sniper

        # ---- scrolling -------------------------------------------------
        gaze_valid = bool(gaze is not None and getattr(gaze, "valid", False))
        gx = float(gaze.x) if gaze_valid else 0.0
        gy = float(gaze.y) if gaze_valid else 0.0
        ny = float(gaze.ny) if gaze_valid else 0.5

        if scroll_cfg == "edge":
            # Palm scroll mode (if it was left on) takes precedence.
            self._update_edge_scroll(ny, gaze_valid, allowed=not self.scroll_mode, st=st)
        if scroll_cfg == "palm":
            self._update_palm_scroll(gy, gaze_valid, st)

        # ---- clicks ----------------------------------------------------
        if self.scroll_mode:
            st.suppressed.extend(["temple", "wink", "dwell"])
            click_ok = False
        else:
            click_ok = True

        # A two-handed temple touch means sniper, never left+right at once.
        temple_click_ok = use_temple and click_ok and not st.temple_both
        if temple_click_ok:
            for side, held, kind in (("left", st.temple_left, CLICK_LEFT),
                                     ("right", st.temple_right, CLICK_RIGHT)):
                if held and not self._temple_latched[side]:
                    self._temple_latched[side] = True
                    if self.debounce.fire(kind):
                        st.events.append(Event(kind, source="temple", t=now))
                elif not held:
                    self._temple_latched[side] = False

        if use_wink and click_ok:
            if fired_wink_l and self.debounce.fire(CLICK_LEFT):
                st.events.append(Event(CLICK_LEFT, source="wink", t=now))
            if fired_wink_r and self.debounce.fire(CLICK_RIGHT):
                st.events.append(Event(CLICK_RIGHT, source="wink", t=now))

        # ---- dwell -----------------------------------------------------
        guard = float(get(cfg, "clicks.dwell.edge_guard_ratio", 0.08))
        in_edge_band = ny <= guard or ny >= 1.0 - guard
        dwell_allowed = (
            gaze_valid
            and click_ok
            and not st.edge_scroll_active
            and not in_edge_band
            and not st.temple_left and not st.temple_right
            and not st.wink_left and not st.wink_right
            and not st.blink
        )
        if st.edge_scroll_active:
            st.suppressed.append("dwell:edge_scroll")
        elif in_edge_band and gaze_valid:
            st.suppressed.append("dwell:edge_guard")
        self._update_dwell(gx, gy, gaze_valid, dwell_allowed, st)

        return st
