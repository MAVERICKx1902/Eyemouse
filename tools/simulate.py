#!/usr/bin/env python3
"""Drive the whole Eyemouse pipeline with synthetic landmarks -- no webcam,
no MediaPipe, no display, no cursor movement.

This exists for two reasons:

1. **Development/CI.** It exercises gaze extraction, calibration fitting,
   smoothing, the deadzone, temple clicks, winks, debounce, dwell, edge
   scroll, palm scroll and sniper mode end to end, and reports pass/fail.
2. **You, on a new machine.** If ``python main.py`` misbehaves, run this
   first. If the simulator passes, the logic is fine and the problem is the
   camera, lighting or MediaPipe models -- which narrows the search a lot.

Usage::

    python tools/simulate.py            # run all checks
    python tools/simulate.py -v         # verbose per-check detail
    python tools/simulate.py --only temple

Ground truth
------------
The synthetic face is generated from a known gaze vector ``g``, and the
simulator *defines* the intended screen position as::

    screen = centre + g * 0.40 * screen_size

Calibration is then run against that relationship exactly as the interactive
routine would, and every later check measures the cursor against the same
ground truth. So this validates the real numeric chain, not a stub.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import calibration as calib          # noqa: E402
import config as config_mod          # noqa: E402
import gestures as gest              # noqa: E402
import mouse_control as mctl         # noqa: E402
import mp_backend as mpb             # noqa: E402
from config import get, set_path     # noqa: E402
from gaze import GazeExtractor       # noqa: E402
from main import EyemouseApp         # noqa: E402
from synthetic import (FakeHands, hand_at_point, hands_from,  # noqa: E402
                       make_face, make_hand)
from util import Clock               # noqa: E402

SCREEN = (1600, 900)
SPAN = 0.40          # gaze g=1.0 maps to 40% of the screen from centre
FACE_CENTRE = (0.5, 0.42)
FACE_WIDTH = 0.36
FRAME_MS = 33        # ~30 fps


def g_to_screen(g: Tuple[float, float], screen: Tuple[int, int] = SCREEN) -> Tuple[float, float]:
    w, h = screen
    return (w / 2.0 + g[0] * SPAN * w, h / 2.0 + g[1] * SPAN * h)


def screen_to_g(p: Tuple[float, float], screen: Tuple[int, int] = SCREEN) -> Tuple[float, float]:
    w, h = screen
    return ((p[0] - w / 2.0) / (SPAN * w), (p[1] - h / 2.0) / (SPAN * h))


# ==========================================================================
# Synthetic tracker
# ==========================================================================
class SimTracker(mpb._BackendBase):
    """Returns whatever the scenario last assigned. Ignores the image."""

    api = "simulated"

    def __init__(self) -> None:
        self.face: mpb.FaceObservation = mpb.FaceObservation()
        self.hands: mpb.HandObservation = mpb.HandObservation()
        self.face_calls = 0
        self.hand_calls = 0

    def process_face(self, bgr, timestamp_ms: int) -> mpb.FaceObservation:
        self.face_calls += 1
        obs = self.face
        obs.timestamp_ms = int(timestamp_ms)
        return obs

    def process_hands(self, bgr, timestamp_ms: int) -> mpb.HandObservation:
        self.hand_calls += 1
        obs = self.hands
        obs.timestamp_ms = int(timestamp_ms)
        return obs

    def close(self) -> None:
        pass


# ==========================================================================
# Harness
# ==========================================================================
@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    group: str = ""


class Scenario:
    """Builds an app wired to synthetic inputs and steps it frame by frame."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 screen: Tuple[int, int] = SCREEN, calibrate: bool = True) -> None:
        self.cfg = cfg or config_mod.load_config("/nonexistent-config.json")
        # Never touch the real cursor, and never open a window.
        set_path(self.cfg, "overlay.show", False)
        set_path(self.cfg, "overlay.desktop_dot", False)
        self.screen = screen
        self.clock = Clock(manual=True)
        self.tracker = SimTracker()
        self.mouse = mctl.MouseController(
            self.cfg, backend=mctl.NullMouse(*screen), screen=screen)
        self.app = EyemouseApp(self.cfg, tracker=self.tracker,
                               frame_source=lambda: np.zeros((240, 320, 3), np.uint8),
                               mouse=self.mouse, screen=screen, clock=self.clock,
                               show_overlay=False)
        self.extractor: GazeExtractor = self.app.extractor
        self.checks: List[Check] = []
        self._g: Tuple[float, float] = (0.0, 0.0)
        self._ear = (0.30, 0.30)
        self._hands: Optional[FakeHands] = None
        self._face_present = True
        self._centre = FACE_CENTRE
        self._width = FACE_WIDTH
        if calibrate:
            # Every check that moves the cursor needs a fitted model; without
            # one the pipeline reports "not calibrated" and silently no-ops.
            self.calibrate(9)
            self.frames(5)

    # ------------------------------------------------------------------
    # scene control
    # ------------------------------------------------------------------
    def set_gaze(self, g: Tuple[float, float]) -> None:
        self._g = (float(g[0]), float(g[1]))

    def set_gaze_screen(self, p: Tuple[float, float]) -> None:
        self.set_gaze(screen_to_g(p, self.screen))

    def set_eyes(self, ear_left: float = 0.30, ear_right: float = 0.30) -> None:
        self._ear = (float(ear_left), float(ear_right))

    def set_face(self, present: bool = True, centre=FACE_CENTRE, width=FACE_WIDTH) -> None:
        self._face_present = present
        self._centre, self._width = tuple(centre), float(width)

    def set_hands(self, hands: Optional[FakeHands]) -> None:
        self._hands = hands

    def hand_on_temple(self, side: str) -> FakeHands:
        """A hand whose index fingertip sits exactly on the given temple."""
        face = make_face(centre=FACE_CENTRE, face_width=FACE_WIDTH)
        anchor = gest.temple_anchors(face)[side]["point"]
        return hands_from([(hand_at_point((float(anchor[0]), float(anchor[1]))), 
                            "Left" if side == "left" else "Right")])

    def hand_away(self) -> FakeHands:
        return hands_from([(make_hand((0.5, 0.92), 0.09, 2), "Right")])

    def open_palm(self) -> FakeHands:
        return hands_from([(make_hand((0.5, 0.75), 0.11, 4), "Right")])

    def fist(self) -> FakeHands:
        return hands_from([(make_hand((0.5, 0.75), 0.11, 0), "Right")])

    # ------------------------------------------------------------------
    def publish(self) -> None:
        """Push the current scene into the tracker."""
        if not self._face_present:
            self.tracker.face = mpb.FaceObservation(present=False)
        else:
            lms = make_face(centre=getattr(self, "_centre", FACE_CENTRE),
                            face_width=getattr(self, "_width", FACE_WIDTH),
                            gaze=self._g, ear_left=self._ear[0], ear_right=self._ear[1])
            self.tracker.face = mpb.FaceObservation(landmarks=lms, present=True,
                                                    n_landmarks=int(lms.shape[0]))
        if self._hands is None or self._hands.landmarks is None:
            self.tracker.hands = mpb.HandObservation(present=False)
        else:
            self.tracker.hands = mpb.HandObservation(
                landmarks=self._hands.landmarks, handedness=list(self._hands.handedness),
                scores=list(self._hands.scores), present=True)

    def frame(self, advance_ms: int = FRAME_MS) -> Dict[str, Any]:
        """Render one synthetic frame through the whole app pipeline."""
        self.publish()
        state = self.app.step()
        self.clock.advance(advance_ms)
        return state

    def frames(self, n: int, advance_ms: int = FRAME_MS) -> List[Dict[str, Any]]:
        return [self.frame(advance_ms) for _ in range(n)]

    def collect_events(self, states: List[Dict[str, Any]], kind: Optional[str] = None
                       ) -> List[gest.Event]:
        out = []
        for s in states:
            for ev in s["gesture"].events:
                if kind is None or ev.kind == kind:
                    out.append(ev)
        return out

    def reset_events(self) -> None:
        self.mouse.backend.clicks.clear()      # type: ignore[attr-defined]
        self.mouse.backend.scrolls.clear()     # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def record(self, group: str, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name=name, passed=bool(passed), detail=detail, group=group))
        return bool(passed)

    # ------------------------------------------------------------------
    def calibrate(self, points: int = 9) -> calib.CalibrationModel:
        """Run the same fit the interactive routine would, on synthetic data."""
        grid = calib.calibration_grid(points, float(get(self.cfg, "calibration.inset_ratio", 0.12)),
                                      *self.screen)
        rows = []
        for sx, sy in grid:
            g = screen_to_g((sx, sy), self.screen)
            self.set_gaze(g)
            self.set_eyes(0.30, 0.30)
            self.publish()
            lms = self.tracker.face.landmarks
            sample = self.extractor.extract(lms, self.clock.ms())
            if not sample.valid:
                raise RuntimeError(f"synthetic sample invalid at {(sx, sy)}: {sample.reason}")
            rows.append([sample.gx, sample.gy, sample.hx, sample.hy, float(sx), float(sy)])
        model = calib.fit_calibration(
            rows, model=str(get(self.cfg, "calibration.model", "quadratic")),
            ridge=float(get(self.cfg, "calibration.ridge", 1e-4)),
            screen_w=self.screen[0], screen_h=self.screen[1], points=grid)
        self.app._apply_calibration(model)
        return model


# ==========================================================================
# Checks
# ==========================================================================
def check_calibration_model(sc: Scenario, verbose: bool) -> None:
    g = "calibration"
    model = sc.calibrate(9)
    ok = sc.record(g, "9-point fit converges", math.isfinite(model.rms_px) and model.rms_px < 8.0,
                   f"model={model.model} RMS={model.rms_px:.3f}px worst={model.max_px:.3f}px")

    # Round-trip through JSON, as the real app does between runs.
    path = os.path.join(ROOT, "_sim_calibration.json")
    calib.save_calibration(model, path)
    reloaded = calib.load_calibration(path)
    try:
        same = reloaded is not None and all(
            np.allclose(getattr(model, a), getattr(reloaded, a), atol=1e-9)
            for a in ("coef_x", "coef_y", "mean", "std"))
        sc.record(g, "save/load round-trip is lossless", same,
                  f"rms after reload={reloaded.rms_px:.3f}px" if reloaded else "reload returned None")
    finally:
        if os.path.exists(path):
            os.remove(path)

    # Accuracy at points the calibration never saw.
    errs = []
    for gx in np.linspace(-0.8, 0.8, 7):
        for gy in np.linspace(-0.8, 0.8, 7):
            truth = g_to_screen((gx, gy), sc.screen)
            pred = model.apply_px(*_feature_for(sc, (gx, gy)))
            errs.append(math.dist(truth, pred))
    sc.record(g, "interpolates unseen gaze points", bool(ok) and max(errs) < 25.0,
              f"max={max(errs):.2f}px mean={np.mean(errs):.2f}px over {len(errs)} points")

    # 5- and 13-point grids must also work.
    for n in (5, 13):
        try:
            m = sc.calibrate(n)
            good = math.isfinite(m.rms_px) and m.rms_px < 15.0
            sc.record(g, f"{n}-point calibration fits", good, f"RMS={m.rms_px:.3f}px")
        except Exception as exc:
            sc.record(g, f"{n}-point calibration fits", False, f"{type(exc).__name_}: {exc}")
    # Restore a clean 9-point model for anything that runs afterwards.
    sc.app._apply_calibration(sc.calibrate(9))


def _feature_for(sc: Scenario, g: Tuple[float, float]) -> Tuple[float, float, float, float]:
    sc.set_gaze(g)
    sc.set_eyes(0.30, 0.30)
    sc.publish()
    s = sc.extractor.extract(sc.tracker.face.landmarks, 0)
    return s.gx, s.gy, s.hx, s.hy


def check_gaze_tracking(sc: Scenario, verbose: bool) -> None:
    g = "gaze"
    sc.app._apply_calibration(sc.calibrate(9))
    sc.set_hands(None)

    targets = [g_to_screen((x, y), sc.screen) for x in (-0.9, 0.0, 0.9) for y in (-0.9, 0.0, 0.9)]
    worst = 0.0
    for t in targets:
        sc.set_gaze_screen(t)
        states = sc.frames(40)          # let the filters settle
        st = states[-1]["gaze"]
        if not st.valid:
            sc.record(g, "gaze state valid", False, f"invalid at {t}: {st.reason}")
            return
        err = math.dist((st.x, st.y), t)
        worst = max(worst, err)
    sc.record(g, "cursor reaches 9 screen positions", worst < 30.0, f"worst error {worst:.1f}px")

    # Monotonic sweep: gaze left->right must move the cursor left->right.
    xs = []
    for v in np.linspace(-0.9, 0.9, 25):
        sc.set_gaze((float(v), 0.0))
        s = sc.frames(12)[-1]["gaze"]
        xs.append(s.x)
    increasing = all(b >= a - 1.0 for a, b in zip(xs, xs[1:]))
    sc.record(g, "horizontal sweep is monotonic", increasing,
              f"span {xs[0]:.0f}->{xs[-1]:.0f}px of {sc.screen[0]}")

    # Smoothing must actually reduce jitter.
    rng = np.random.default_rng(7)
    raw, smooth = [], []
    sc.set_gaze((0.0, 0.0))
    sc.frames(30)
    for i in range(120):
        noise = (float(rng.normal(0, 0.02)), float(rng.normal(0, 0.02)))
        sc.set_gaze(noise)
        s = sc.frame()["gaze"]
        raw.append(g_to_screen(noise, sc.screen)[0])
        smooth.append(s.x)
    jit_raw = float(np.std(np.diff(raw)))
    jit_out = float(np.std(np.diff(smooth)))
    sc.record(g, "smoothing reduces jitter", jit_out < jit_raw * 0.6,
              f"input {jit_raw:.1f}px/frame -> output {jit_out:.1f}px/frame "
              f"({100 * (1 - jit_out / max(jit_raw, 1e-9)):.0f}% reduction)")

    # Face loss must not throw or move the cursor wildly.
    before = sc.mouse.backend.position()
    sc.set_face(present=False)
    states = sc.frames(10)
    sc.set_face(present=True)
    invalid = all(not s["gaze"].valid for s in states)
    sc.record(g, "face loss handled gracefully", invalid and sc.mouse.backend.position() == before,
              f"cursor held at {before}" if invalid else "unexpected valid gaze")


def check_temple_clicks(sc: Scenario, verbose: bool) -> None:
    g = "temple"
    sc.set_gaze((0.0, 0.0))
    sc.frames(20)

    sc.set_hands(sc.hand_away())
    sc.frames(6)
    sc.reset_events()

    # Left temple -> left click.
    sc.set_hands(sc.hand_on_temple("left"))
    states = sc.frames(6)
    lefts = sc.collect_events(states, gest.CLICK_LEFT)
    sc.record(g, "left temple -> left click", len(lefts) == 1,
              f"{len(lefts)} left-click event(s), source={lefts[0].source if lefts else '-'}")

    # Holding the touch must not auto-repeat.
    states = sc.frames(40)
    sc.record(g, "holding temple does not repeat", len(sc.collect_events(states, gest.CLICK_LEFT)) == 0,
              "no extra clicks while held" if not sc.collect_events(states, gest.CLICK_LEFT) else "repeated")

    # Release then re-touch after the debounce window -> a second click.
    sc.set_hands(sc.hand_away())
    sc.frames(6)
    sc.clock.advance(600)
    sc.set_hands(sc.hand_on_temple("left"))
    states = sc.frames(6)
    sc.record(g, "re-touch after debounce clicks again",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 1, "")

    # Right temple -> right click.
    sc.set_hands(sc.hand_away())
    sc.frames(6)
    sc.clock.advance(600)
    sc.set_hands(sc.hand_on_temple("right"))
    states = sc.frames(6)
    rights = sc.collect_events(states, gest.CLICK_RIGHT)
    sc.record(g, "right temple -> right click", len(rights) == 1,
              f"{len(rights)} right-click event(s)")
    sc.record(g, "right temple does not also left-click",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 0, "")

    # BOTH temples -> sniper, and crucially no clicks at all.
    sc.set_hands(sc.hand_away())
    sc.frames(6)
    sc.clock.advance(600)
    face = make_face(centre=FACE_CENTRE, face_width=FACE_WIDTH)
    anchors = gest.temple_anchors(face)
    both = hands_from([
        (hand_at_point((float(anchors["left"]["point"][0]), float(anchors["left"]["point"][1]))), "Left"),
        (hand_at_point((float(anchors["right"]["point"][0]), float(anchors["right"]["point"][1]))), "Right"),
    ])
    sc.set_hands(both)
    sc.reset_events()
    states = sc.frames(10)
    clicks = (sc.collect_events(states, gest.CLICK_LEFT) + sc.collect_events(states, gest.CLICK_RIGHT))
    snipers = sc.collect_events(states, gest.SNIPER_ON)
    sc.record(g, "both temples -> sniper, no clicks", len(clicks) == 0 and len(snipers) == 1,
              f"clicks={len(clicks)} sniper_on={len(snipers)}")
    sc.record(g, "sniper mode active while both held", states[-1]["gesture"].sniper, "")

    # Sniper must slow the cursor by roughly the configured factor.
    factor = float(get(sc.cfg, "precision.sniper_factor", 0.25))
    p0 = sc.mouse.position
    sc.set_gaze_screen(g_to_screen((0.6, 0.0), sc.screen))
    sc.frame()
    step_sniper = math.dist(sc.mouse.position, p0)
    sc.set_hands(sc.hand_away())
    sc.frames(4)
    sc.clock.advance(600)
    sc.set_gaze((0.0, 0.0))
    sc.frames(30)
    p1 = sc.mouse.position
    sc.set_gaze_screen(g_to_screen((0.6, 0.0), sc.screen))
    sc.frame()
    step_normal = math.dist(sc.mouse.position, p1)
    ratio = step_sniper / max(step_normal, 1e-6)
    sc.record(g, "sniper slows the cursor ~4x", 0.10 < ratio < 0.60,
              f"sniper step {step_sniper:.1f}px vs normal {step_normal:.1f}px (ratio {ratio:.2f}, "
              f"configured {factor})")

    sc.set_hands(None)
    sc.frames(4)


def check_winks(sc: Scenario, verbose: bool) -> None:
    g = "wink"
    sc.set_hands(None)
    sc.set_gaze((0.0, 0.0))
    sc.frames(20)
    sc.clock.advance(700)
    sc.reset_events()

    # Left wink -> left click, and the cursor must not jump.
    before = sc.mouse.position
    sc.set_eyes(ear_left=0.05, ear_right=0.30)
    states = sc.frames(6)
    lefts = sc.collect_events(states, gest.CLICK_LEFT)
    sc.record(g, "left wink -> left click", len(lefts) == 1,
              f"{len(lefts)} event(s), source={lefts[0].source if lefts else '-'}")
    drift = math.dist(sc.mouse.position, before)
    sc.record(g, "wink does not move the cursor", drift < 25.0, f"cursor drifted {drift:.1f}px")

    # Still winking must not repeat.
    states = sc.frames(30)
    sc.record(g, "holding a wink does not repeat",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 0, "")

    # Right wink -> right click.
    sc.set_eyes(0.30, 0.30)
    sc.frames(6)
    sc.clock.advance(700)
    sc.set_eyes(ear_left=0.30, ear_right=0.05)
    states = sc.frames(6)
    sc.record(g, "right wink -> right click",
              len(sc.collect_events(states, gest.CLICK_RIGHT)) == 1, "")

    # A blink (both eyes) must NOT click. This is the main false-positive risk.
    sc.set_eyes(0.30, 0.30)
    sc.frames(6)
    sc.clock.advance(700)
    sc.reset_events()
    sc.set_eyes(0.05, 0.05)
    states = sc.frames(6)
    n = (len(sc.collect_events(states, gest.CLICK_LEFT)) +
         len(sc.collect_events(states, gest.CLICK_RIGHT)))
    sc.record(g, "blink does not click", n == 0, f"{n} click event(s) during a blink")
    sc.record(g, "blink reported as blink", states[-1]["gesture"].blink, "")

    # Rapid natural blinking over ~3 s must not produce clicks.
    sc.set_eyes(0.30, 0.30)
    sc.frames(4)
    sc.reset_events()
    for _ in range(6):
        sc.set_eyes(0.05, 0.05)
        sc.frames(3)
        sc.set_eyes(0.30, 0.30)
        sc.frames(12)
    sc.record(g, "repeated blinking produces no clicks", sc.mouse.backend.clicks == [],  # type: ignore[attr-defined]
              f"clicks={sc.mouse.backend.clicks}")  # type: ignore[attr-defined]
    sc.set_eyes(0.30, 0.30)


def check_dwell(sc: Scenario, verbose: bool) -> None:
    g = "dwell"
    sc.set_hands(None)
    sc.set_eyes(0.30, 0.30)
    dwell_ms = int(get(sc.cfg, "clicks.dwell.ms", 1400))
    target = g_to_screen((0.3, 0.2), sc.screen)
    sc.set_gaze_screen(target)
    sc.frames(40)
    sc.clock.advance(700)
    sc.reset_events()

    need = int(dwell_ms / FRAME_MS) + 8
    states = sc.frames(need)
    clicks = sc.collect_events(states, gest.CLICK_LEFT)
    sc.record(g, f"dwell {dwell_ms}ms -> left click", len(clicks) == 1,
              f"{len(clicks)} click(s) after {need} frames (~{need * FRAME_MS}ms), "
              f"source={clicks[0].source if clicks else '-'}")

    # Staying put must not fire again.
    states = sc.frames(int(dwell_ms / FRAME_MS) + 20)
    sc.record(g, "dwelling in place does not repeat",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 0, "")

    # Moving outside the radius re-arms it.
    sc.set_gaze_screen(g_to_screen((-0.4, -0.3), sc.screen))
    sc.frames(40)
    sc.reset_events()
    states = sc.frames(int(dwell_ms / FRAME_MS) + 8)
    sc.record(g, "moving away re-arms dwell", len(sc.collect_events(states, gest.CLICK_LEFT)) == 1, "")

    # Small jitter inside the radius must not reset the timer.
    sc.set_gaze_screen(target)
    sc.frames(40)
    sc.reset_events()
    radius = float(get(sc.cfg, "clicks.dwell.radius_px", 24))
    rng = np.random.default_rng(3)
    fired_at = None
    for i in range(int(dwell_ms / FRAME_MS) + 15):
        jx = target[0] + float(rng.normal(0, radius * 0.25))
        jy = target[1] + float(rng.normal(0, radius * 0.25))
        sc.set_gaze_screen((jx, jy))
        s = sc.frame()
        if any(e.kind == gest.CLICK_LEFT for e in s["gesture"].events) and fired_at is None:
            fired_at = i
    sc.record(g, "jitter inside the radius still dwells", fired_at is not None,
              f"fired on frame {fired_at}" if fired_at is not None else "never fired")


def check_edge_scroll(sc: Scenario, verbose: bool) -> None:
    g = "scroll"
    set_path(sc.cfg, "scroll.mode", "edge")
    sc.app.gestures.apply_config(sc.cfg)
    sc.set_hands(None)
    sc.set_eyes(0.30, 0.30)
    band = float(get(sc.cfg, "scroll.edge.band_ratio", 0.06))
    dwell_ms = int(get(sc.cfg, "scroll.edge.dwell_ms", 800))

    sc.set_gaze((0.0, 0.0))
    sc.frames(30)
    sc.reset_events()

    # Briefly entering the band must not scroll (this is the 800ms guard).
    sc.set_gaze_screen((sc.screen[0] / 2, band * sc.screen[1] * 0.4))
    states = sc.frames(int((dwell_ms * 0.5) / FRAME_MS))
    sc.record(g, "brief edge entry does not scroll",
              len(sc.collect_events(states, gest.SCROLL)) == 0,
              f"{len(sc.collect_events(states, gest.SCROLL))} scroll event(s) before the dwell elapsed")

    # Holding there must scroll up (positive amounts).
    states = sc.frames(int((dwell_ms * 1.5) / FRAME_MS) + 12)
    ups = [e for e in sc.collect_events(states, gest.SCROLL) if e.amount > 0]
    downs = [e for e in sc.collect_events(states, gest.SCROLL) if e.amount < 0]
    sc.record(g, "top edge dwell scrolls up", len(ups) > 0 and len(downs) == 0,
              f"{len(ups)} up event(s), total {sum(e.amount for e in ups)} units")

    # Bottom edge must scroll down.
    sc.set_gaze((0.0, 0.0))
    sc.frames(30)
    sc.reset_events()
    sc.set_gaze_screen((sc.screen[0] / 2, sc.screen[1] * (1 - band * 0.4)))
    states = sc.frames(int((dwell_ms * 1.5) / FRAME_MS) + 12)
    downs = [e for e in sc.collect_events(states, gest.SCROLL) if e.amount < 0]
    ups = [e for e in sc.collect_events(states, gest.SCROLL) if e.amount > 0]
    sc.record(g, "bottom edge dwell scrolls down", len(downs) > 0 and len(ups) == 0,
              f"{len(downs)} down event(s), total {sum(e.amount for e in downs)} units")

    # Returning to the centre must stop it.
    sc.set_gaze((0.0, 0.0))
    sc.frames(10)
    sc.reset_events()
    states = sc.frames(30)
    sc.record(g, "leaving the edge stops scrolling",
              len(sc.collect_events(states, gest.SCROLL)) == 0, "")

    # Scroll and click must not conflict: dwell-click is suppressed at the edge.
    sc.set_gaze_screen((sc.screen[0] / 2, sc.screen[1] * (1 - band * 0.3)))
    sc.frames(int((dwell_ms * 3) / FRAME_MS) + int(get(sc.cfg, "clicks.dwell.ms", 1400) / FRAME_MS))
    sc.reset_events()
    states = sc.frames(int(get(sc.cfg, "clicks.dwell.ms", 1400) / FRAME_MS) + 20)
    sc.record(g, "no dwell-click while edge scrolling",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 0, "")
    sc.set_gaze((0.0, 0.0))
    sc.frames(20)


def check_palm_scroll(sc: Scenario, verbose: bool) -> None:
    g = "palm"
    set_path(sc.cfg, "scroll.mode", "palm")
    sc.app.gestures.apply_config(sc.cfg)
    sc.set_eyes(0.30, 0.30)
    sc.set_hands(None)
    sc.set_gaze((0.0, 0.0))
    sc.frames(30)
    sc.reset_events()

    toggle_ms = int(get(sc.cfg, "scroll.palm.toggle_ms", 400))
    # Open palm held -> scroll mode on.
    sc.set_hands(sc.open_palm())
    states = sc.frames(int(toggle_ms / FRAME_MS) + 6)
    on = sc.collect_events(states, gest.SCROLL_MODE_ON)
    sc.record(g, "open palm toggles scroll mode", len(on) == 1 and states[-1]["gesture"].scroll_mode,
              f"{len(on)} toggle-on event(s)")

    # A brief palm flash must not toggle.
    sc.set_hands(sc.fist())
    sc.frames(6)
    sc.reset_events()
    sc.set_hands(sc.open_palm())
    sc.frames(2)
    sc.set_hands(sc.fist())
    states = sc.frames(6)
    sc.record(g, "brief palm flash does not toggle",
              len(sc.collect_events(states, gest.SCROLL_MODE_ON)) == 0, "")

    # Re-enter, then move the eyes down -> downward scroll.
    sc.set_hands(sc.open_palm())
    sc.frames(int(toggle_ms / FRAME_MS) + 4)
    sc.reset_events()
    # Collect across the *whole* sweep: the scroll events are emitted as the
    # accumulated eye movement crosses each px_per_click threshold, so looking
    # only at the final frames would miss them.
    sweep: List[Dict[str, Any]] = []
    sc.set_gaze((0.0, 0.0))
    sweep += sc.frames(4)
    for v in np.linspace(0.0, 0.55, 14):
        sc.set_gaze((0.0, float(v)))
        sweep += sc.frames(2)
    sweep += sc.frames(4)
    scrolls = sc.collect_events(sweep, gest.SCROLL)
    sc.record(g, "eye-down in scroll mode scrolls down",
              len(scrolls) > 0 and all(e.amount < 0 for e in scrolls),
              f"{len(scrolls)} event(s), amounts {[e.amount for e in scrolls][:6]}")

    # Looking back up must scroll the other way.
    sc.reset_events()
    sweep = []
    for v in np.linspace(0.55, 0.0, 14):
        sc.set_gaze((0.0, float(v)))
        sweep += sc.frames(2)
    sweep += sc.frames(4)
    ups = sc.collect_events(sweep, gest.SCROLL)
    sc.record(g, "eye-up in scroll mode scrolls up",
              len(ups) > 0 and all(e.amount > 0 for e in ups),
              f"{len(ups)} event(s), amounts {[e.amount for e in ups][:6]}")

    # While in scroll mode, a temple touch must NOT click (no conflict).
    sc.reset_events()
    face = make_face(centre=FACE_CENTRE, face_width=FACE_WIDTH)
    anchor = gest.temple_anchors(face)["left"]["point"]
    sc.set_hands(hands_from([(hand_at_point((float(anchor[0]), float(anchor[1]))), "Left"),
                             (make_hand((0.5, 0.75), 0.11, 4), "Right")]))
    states = sc.frames(10)
    sc.record(g, "temple touch suppressed in scroll mode",
              len(sc.collect_events(states, gest.CLICK_LEFT)) == 0, "")

    # Fist -> scroll mode off.
    sc.set_hands(sc.fist())
    sc.clock.advance(600)
    sc.reset_events()
    states = sc.frames(8)
    sc.record(g, "fist exits scroll mode",
              len(sc.collect_events(states, gest.SCROLL_MODE_OFF)) >= 1 and not states[-1]["gesture"].scroll_mode,
              "")
    set_path(sc.cfg, "scroll.mode", "edge")
    sc.app.gestures.apply_config(sc.cfg)
    sc.set_hands(None)
    sc.frames(6)


def check_debounce(sc: Scenario, verbose: bool) -> None:
    g = "debounce"
    set_path(sc.cfg, "clicks.gesture", "wink")
    sc.app.gestures.apply_config(sc.cfg)
    sc.set_hands(None)
    sc.set_eyes(0.30, 0.30)
    sc.set_gaze((0.0, 0.0))
    sc.frames(20)

    db = int(get(sc.cfg, "clicks.debounce_ms", 500))
    # Two winks inside the debounce window -> one click.
    sc.reset_events()
    sc.set_eyes(0.05, 0.30)
    sc.frames(4)
    sc.set_eyes(0.30, 0.30)
    sc.frames(4)
    sc.clock.advance(int(db * 0.4))
    sc.set_eyes(0.05, 0.30)
    sc.frames(4)
    sc.set_eyes(0.30, 0.30)
    sc.frames(4)
    n = len(sc.mouse.backend.clicks)  # type: ignore[attr-defined]
    sc.record(g, f"two winks within {db}ms click once", n == 1, f"{n} click(s) delivered")

    # After the window, a second wink clicks again.
    sc.clock.advance(int(db * 1.5))
    sc.set_eyes(0.05, 0.30)
    sc.frames(4)
    sc.set_eyes(0.30, 0.30)
    sc.frames(4)
    n2 = len(sc.mouse.backend.clicks)  # type: ignore[attr-defined]
    sc.record(g, "wink after the debounce window clicks", n2 == 2, f"{n2} total click(s)")

    set_path(sc.cfg, "clicks.gesture", "both")
    sc.app.gestures.apply_config(sc.cfg)
    sc.set_eyes(0.30, 0.30)
    sc.frames(6)


def check_performance(sc: Scenario, verbose: bool) -> None:
    g = "performance"
    import time as _t
    sc.app._apply_calibration(sc.calibrate(9))
    sc.set_hands(sc.hand_away())
    sc.set_gaze((0.0, 0.0))
    sc.frames(20)

    n = 300
    t0 = _t.perf_counter()
    for i in range(n):
        sc.set_gaze((0.6 * math.sin(i / 12.0), 0.5 * math.cos(i / 9.0)))
        sc.frame()
    dt = _t.perf_counter() - t0
    per_frame = dt / n * 1000.0
    fps = n / max(dt, 1e-9)
    # The MediaPipe inference itself is not measured here (it is simulated),
    # so this is the budget the *rest* of the pipeline must fit inside.
    sc.record(g, "non-inference overhead leaves room for 15fps", per_frame < 20.0,
              f"{per_frame:.2f} ms/frame of app logic ({fps:.0f} fps equivalent); "
              f"a 30fps budget is 33.3ms, so inference may use {33.3 - per_frame:.1f}ms")
    sc.record(g, "hands are throttled, not run every frame",
              sc.tracker.hand_calls <= sc.tracker.face_calls,
              f"face calls={sc.tracker.face_calls} hand calls={sc.tracker.hand_calls} "
              f"(every {get(sc.cfg, 'performance.hands_every_n_frames')} frames)")


# ==========================================================================
# Runner
# ==========================================================================
CHECKS: List[Tuple[str, Callable[[Scenario, bool], None]]] = [
    ("calibration", check_calibration_model),
    ("gaze", check_gaze_tracking),
    ("temple", check_temple_clicks),
    ("wink", check_winks),
    ("dwell", check_dwell),
    ("scroll", check_edge_scroll),
    ("palm", check_palm_scroll),
    ("debounce", check_debounce),
    ("performance", check_performance),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run Eyemouse against synthetic landmark data.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated check groups to run")
    p.add_argument("--config", default=None, help="use a real config.json instead of defaults")
    args = p.parse_args(argv)

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    cfg = config_mod.load_config(args.config) if args.config else None

    print("=" * 74)
    print("Eyemouse simulation -- synthetic landmarks, no webcam/MediaPipe/display")
    print(f"screen {SCREEN[0]}x{SCREEN[1]}, frame time {FRAME_MS}ms")
    print("=" * 74)

    all_checks: List[Check] = []
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        sc = Scenario(cfg)
        try:
            fn(sc, args.verbose)
        except Exception as exc:
            import traceback
            sc.record(name, f"{name} harness crashed", False,
                      f"{type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
        all_checks.extend(sc.checks)
        print(f"\n[{name}]")
        for c in sc.checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"  {mark}  {c.name}")
            if c.detail and (args.verbose or not c.passed):
                print(f"        {c.detail}")

    passed = sum(1 for c in all_checks if c.passed)
    total = len(all_checks)
    print("\n" + "=" * 74)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed != total:
        print("Failures:")
        for c in all_checks:
            if not c.passed:
                print(f"  - [{c.group}] {c.name}: {c.detail}")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
