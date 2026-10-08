"""Gaze extraction and smoothing.

Pipeline
--------
::

    face landmarks (N,3) normalized
        -> GazeExtractor      : geometry -> head-invariant gaze features
        -> CalibrationModel   : features  -> normalized screen coords (0..1)
        -> gain / offset      : user tuning
        -> filter chain       : moving average -> One Euro / Kalman
        -> deadzone           : swallow sub-threshold jitter
        -> screen pixels

Design notes
~~~~~~~~~~~~
*Everything downstream of :class:`GazeExtractor` works on plain numpy arrays
and floats.* MediaPipe is never imported here, which is what makes the whole
gaze chain unit-testable without a webcam or the model bundles.

The gaze feature is deliberately **head-invariant**: each iris centre is
projected into its own eye's corner-to-corner frame and divided by the eye
width, so leaning toward or away from the camera barely changes the signal.
The two eyes are given a common sign convention (both axes point along image
+x for a frontal face) before being averaged, which is the detail most
webcam eye-trackers get wrong -- MediaPipe's ``lateral -> medial`` vector
points in *opposite* image directions for the left and right eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # only needed for the iris-less fallback pupil detector
except Exception:  # pragma: no cover - OpenCV is a hard requirement in practice
    cv2 = None

import landmarks as LMK
from config import get
from util import clamp

__all__ = [
    "GazeSample",
    "GazeState",
    "MovingAverage",
    "OneEuroFilter2D",
    "KalmanCV2D",
    "Deadzone",
    "GazeExtractor",
    "GazePipeline",
    "build_filter_chain",
]

# --------------------------------------------------------------------------
# Per-eye geometry spec
# --------------------------------------------------------------------------
EYE_SPECS: Dict[str, Dict[str, Any]] = {
    # axis_sign makes both eyes' horizontal axis point along image +x.
    "right": {
        "lateral": LMK.RIGHT_EYE_LATERAL,   # 33
        "medial": LMK.RIGHT_EYE_MEDIAL,     # 133
        "axis_sign": +1,
        "iris": LMK.RIGHT_IRIS_CENTER,      # 468
        "ring": LMK.RIGHT_IRIS_RING,
        "ear": LMK.RIGHT_EYE_EAR,
        "lids": LMK.RIGHT_EYE_LIDS,
    },
    "left": {
        "lateral": LMK.LEFT_EYE_LATERAL,    # 263
        "medial": LMK.LEFT_EYE_MEDIAL,      # 362
        "axis_sign": -1,
        "iris": LMK.LEFT_IRIS_CENTER,       # 473
        "ring": LMK.LEFT_IRIS_RING,
        "ear": LMK.LEFT_EYE_EAR,
        "lids": LMK.LEFT_EYE_LIDS,
    },
}

_EPS = 1e-9


# ==========================================================================
# Data containers
# ==========================================================================
@dataclass
class GazeSample:
    """One frame of raw, *unmapped* gaze evidence."""

    valid: bool = False
    gx: float = 0.0            # horizontal gaze feature, ~0 when centred
    gy: float = 0.0            # vertical gaze feature, ~0 when centred
    hx: float = 0.5            # head position (mid-eye) in frame, 0..1
    hy: float = 0.5
    face_width: float = 0.0    # normalized 234<->454 distance
    ear_left: float = 0.3      # eye aspect ratios
    ear_right: float = 0.3
    lid_left: float = 0.3      # simple vertical lid opening (robustness check)
    lid_right: float = 0.3
    blink: bool = False
    iris_used: bool = True
    quality: float = 0.0       # 0..1 heuristic confidence
    timestamp_ms: int = 0
    # Per-eye debug detail, used by the overlay and by tests.
    eyes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reason: str = ""


@dataclass
class GazeState:
    """One frame of post-processing, screen-space output."""

    valid: bool = False
    x: float = 0.0             # screen pixels
    y: float = 0.0
    nx: float = 0.0            # normalized 0..1
    ny: float = 0.0
    speed: float = 0.0         # px/s, used to auto-tune One Euro beta
    moved: bool = False        # False when the deadzone swallowed the update
    sample: Optional[GazeSample] = None
    reason: str = ""


# ==========================================================================
# Filters
# ==========================================================================
class MovingAverage:
    """Fixed-window moving average over 2D points.

    Uses a ring buffer so the cost is O(1) per sample regardless of window
    size, and it seeds itself on the first sample instead of ramping up from
    the origin (which would otherwise drag the cursor across the screen at
    startup).
    """

    def __init__(self, window: int = 5) -> None:
        self._window = -1
        self._buf = np.zeros((1, 2), dtype=np.float64)
        self._n = 0
        self._i = 0
        self.set_window(window)

    def set_window(self, window: int) -> None:
        """Resize the ring buffer. A no-op when the window is unchanged, so
        live slider updates do not wipe accumulated history."""
        window = max(1, int(window))
        if window == self._window:
            return
        old = self._buf[: self._n] if self._n else np.empty((0, 2))
        self._window = window
        self._buf = np.zeros((window, 2), dtype=np.float64)
        keep = old[-window:] if len(old) else old
        if len(keep):
            self._buf[: len(keep)] = keep
        self._n = len(keep)
        self._i = self._n % window

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self._n = 0
        self._i = 0
        if x is not None and y is not None:
            self._buf[:] = (x, y)
            self._n = self._window

    def __call__(self, x: float, y: float, dt: float = 0.0) -> Tuple[float, float]:
        self._buf[self._i] = (x, y)
        self._i = (self._i + 1) % self._window
        self._n = min(self._n + 1, self._window)
        mean = self._buf[: self._n].mean(axis=0)
        return float(mean[0]), float(mean[1])

    @property
    def window(self) -> int:
        return self._window


class _LowPass:
    """Exponential low-pass with an initialisation fast-path."""

    def __init__(self) -> None:
        self._y: Optional[float] = None
        self._init = True

    def reset(self, value: Optional[float] = None) -> None:
        self._y = value
        self._init = value is None

    def __call__(self, x: float, alpha: float) -> float:
        if self._init or self._y is None:
            self._y = x
            self._init = False
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return float(self._y)


class OneEuroFilter2D:
    """One Euro filter (Casiez et al., CHI 2012) for 2D cursor positions.

    The adaptive cutoff is the reason this feels better than a plain moving
    average: at low speed it smooths hard (kills jitter), and as speed rises
    the cutoff opens up so lag does not build. ``beta`` trades jitter against
    lag; ``min_cutoff`` sets the smoothing floor.
    """

    def __init__(self, min_cutoff: float = 1.1, beta: float = 0.012, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = [_LowPass(), _LowPass()]
        self._dx = [_LowPass(), _LowPass()]
        self._last_t: Optional[float] = None

    def set_params(self, min_cutoff: Optional[float] = None, beta: Optional[float] = None,
                   d_cutoff: Optional[float] = None) -> None:
        if min_cutoff is not None:
            self.min_cutoff = max(1e-4, float(min_cutoff))
        if beta is not None:
            self.beta = max(0.0, float(beta))
        if d_cutoff is not None:
            self.d_cutoff = max(1e-4, float(d_cutoff))

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        for i, lp in enumerate(self._x):
            lp.reset(x if i == 0 else y if (x is not None and y is not None) else None)
        for lp in self._dx:
            lp.reset(0.0)
        self._last_t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, y: float, dt: float) -> Tuple[float, float]:
        dt = float(dt) if dt and dt > 0 else 1.0 / 60.0
        dt = clamp(dt, 1.0 / 1000.0, 1.0)
        out = []
        for i, value in enumerate((x, y)):
            if self._last_t is None:
                dx = 0.0
            else:
                prev = self._x[i]._y
                dx = 0.0 if prev is None else (value - prev) / dt
            edx = self._dx[i](dx, self._alpha(self.d_cutoff, dt))
            cutoff = self.min_cutoff + self.beta * abs(edx)
            out.append(self._x[i](value, self._alpha(cutoff, dt)))
        self._last_t = dt
        return float(out[0]), float(out[1])


class KalmanCV2D:
    """Constant-velocity Kalman filter on a 4-state vector ``[x, vx, y, vy]``.

    A compact alternative to One Euro when the user prefers velocity-aware
    prediction. The two axes are independent, so the 4x4 system decomposes
    into two 2x2 solves; we still write it as one matrix update for clarity
    (it is cheap enough at 30 Hz).
    """

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 2.5e-2) -> None:
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1.0
        self._init = False

    def set_params(self, process_noise: Optional[float] = None,
                   measurement_noise: Optional[float] = None) -> None:
        if process_noise is not None:
            self.q = max(1e-9, float(process_noise))
        if measurement_noise is not None:
            self.r = max(1e-9, float(measurement_noise))

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self._init = False
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1.0
        if x is not None and y is not None:
            self._x = np.array([x, 0.0, y, 0.0], dtype=np.float64)
            self._init = True

    def __call__(self, x: float, y: float, dt: float) -> Tuple[float, float]:
        dt = clamp(float(dt) if dt else 1.0 / 60.0, 1.0 / 1000.0, 1.0)
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        q = self.q
        Q = np.diag([q * dt * dt, q * dt, q * dt * dt, q * dt]) * 1000.0
        z = np.array([x, y], dtype=np.float64)

        if not self._init:
            self._x = np.array([x, 0.0, y, 0.0], dtype=np.float64)
            self._P = np.eye(4, dtype=np.float64) * 1.0
            self._init = True
            return float(x), float(y)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        R = np.eye(2, dtype=np.float64) * self.r
        S = H @ self._P @ H.T + R
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically guarded
            return float(self._x[0]), float(self._x[2])
        self._x = self._x + K @ (z - H @ self._x)
        self._P = (np.eye(4) - K @ H) @ self._P
        return float(self._x[0]), float(self._x[2])


class Deadzone:
    """Spatial deadzone that never loses accumulated motion.

    Because the comparison is against the last *emitted absolute* position
    (not against the previous input), slow drift is preserved: a gaze that
    creeps 2 px at a time with a 3 px deadzone still eventually moves the
    cursor, and it lands exactly where the gaze is rather than one deadzone
    behind. Optional ``soft`` mode attenuates sub-threshold motion instead of
    zeroing it, for users who find a hard deadzone "sticky".
    """

    def __init__(self, radius_px: float = 2.5, soft: float = 0.0) -> None:
        self.set_params(radius_px, soft)
        self._out: Optional[Tuple[float, float]] = None

    def set_params(self, radius_px: Optional[float] = None, soft: Optional[float] = None) -> None:
        if radius_px is not None:
            self.radius = max(0.0, float(radius_px))
        if soft is not None:
            self.soft = clamp(float(soft), 0.0, 1.0)

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self._out = None if (x is None or y is None) else (float(x), float(y))

    def __call__(self, x: float, y: float) -> Tuple[Tuple[float, float], bool]:
        """Returns ``((out_x, out_y), moved)``.

        ``moved`` is False when the deadzone held the previous position, which
        lets callers skip a (surprisingly expensive) OS cursor call.
        """
        if self.radius <= 0.0:
            self._out = (float(x), float(y))
            return self._out, True
        if self._out is None:
            self._out = (float(x), float(y))
            return self._out, True
        ox, oy = self._out
        dx, dy = x - ox, y - oy
        dist = math.hypot(dx, dy)
        if dist < self.radius:
            if self.soft > 0.0 and dist > _EPS:
                self._out = (ox + dx * self.soft, oy + dy * self.soft)
                return self._out, False
            return (ox, oy), False
        self._out = (float(x), float(y))
        return self._out, True


def build_filter_chain(cfg: Dict[str, Any]):
    """Instantiate the smoothing chain named by ``gaze.filter``."""
    name = str(get(cfg, "gaze.filter", "ma_one_euro")).lower()
    ma = MovingAverage(int(get(cfg, "gaze.moving_average_window", 4)))
    oe_cfg = get(cfg, "gaze.one_euro", {}) or {}
    one_euro = OneEuroFilter2D(
        min_cutoff=float(oe_cfg.get("min_cutoff", 1.1)),
        beta=float(oe_cfg.get("beta", 0.012)),
        d_cutoff=float(oe_cfg.get("d_cutoff", 1.0)),
    )
    k_cfg = get(cfg, "gaze.kalman", {}) or {}
    kalman = KalmanCV2D(
        process_noise=float(k_cfg.get("process_noise", 1e-3)),
        measurement_noise=float(k_cfg.get("measurement_noise", 2.5e-2)),
    )
    if name == "none":
        return []
    if name == "moving_average":
        return [ma]
    if name == "one_euro":
        return [one_euro]
    if name == "kalman":
        return [kalman]
    return [ma, one_euro]  # ma_one_euro (default)


# ==========================================================================
# Extraction
# ==========================================================================
class GazeExtractor:
    """Face landmarks -> :class:`GazeSample`.

    Parameters
    ----------
    blink_ear:
        Both-eyes-closed threshold on the eye aspect ratio.
    min_face_width:
        Reject frames where the face is smaller than this fraction of the
        frame width -- the iris signal is pure noise at that scale.
    """

    def __init__(self, blink_ear: float = 0.19, min_face_width: float = 0.10,
                 iris_fallback_threshold: bool = True,
                 min_ear_for_gaze: float = 0.145) -> None:
        self.blink_ear = float(blink_ear)
        self.min_face_width = float(min_face_width)
        self.iris_fallback_threshold = iris_fallback_threshold
        # An eye must be at least this open for its iris to be trusted.
        self.min_ear_for_gaze = float(min_ear_for_gaze)

    # ------------------------------------------------------------------
    @staticmethod
    def _eye_axis(lms: np.ndarray, spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        lat = lms[spec["lateral"], :2]
        med = lms[spec["medial"], :2]
        a = (med - lat) * spec["axis_sign"]
        w = float(np.linalg.norm(a))
        if w < _EPS:
            return np.zeros(2), np.zeros(2), 0.0
        u = a / w
        # Rotate +90 deg in image coords (y down) so v points "down" the face.
        v = np.array([-u[1], u[0]], dtype=np.float64)
        return u, v, w

    @staticmethod
    def _pupil_from_blob(rgb: np.ndarray, lms: np.ndarray, spec: Dict[str, Any]
                         ) -> Optional[Tuple[float, float]]:
        """Fallback pupil finder for 468-landmark models (no iris points).

        Crops the eye quad, and takes the intensity-weighted centroid of the
        darkest pixels. Crude, but it recovers a usable horizontal signal when
        a face-landmarker bundle without iris refinement is in use.
        """
        if cv2 is None or rgb is None or rgb.size == 0:
            return None
        h, w = rgb.shape[:2]
        idx = list(spec["ear"])
        pts = np.stack([lms[i, :2] for i in idx], axis=0)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad = 0.35 * max(x1 - x0, y1 - y0)
        x0 = int(clamp((x0 - pad) * w, 0, w - 1)); x1 = int(clamp((x1 + pad) * w, 1, w))
        y0 = int(clamp((y0 - pad) * h, 0, h - 1)); y1 = int(clamp((y1 + pad) * h, 1, h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = float(np.percentile(gray, 20))
        mask = gray <= max(thresh, 1.0)
        if mask.sum() < 4:
            return None
        weight = (thresh + 1.0 - gray) * mask
        total = float(weight.sum())
        if total < _EPS:
            return None
        yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        cx = float((xx * weight).sum() / total) / max(1, gray.shape[1])
        cy = float((yy * weight).sum() / total) / max(1, gray.shape[0])
        # Crop-normalized -> frame-normalized.
        return ((x0 + cx * (x1 - x0)) / w, (y0 + cy * (y1 - y0)) / h)

    # ------------------------------------------------------------------
    def extract(self, lms: Optional[np.ndarray], timestamp_ms: int = 0,
                rgb: Optional[np.ndarray] = None) -> GazeSample:
        """``lms`` is an ``(N,3)`` array of normalized landmarks."""
        if lms is None:
            return GazeSample(valid=False, timestamp_ms=timestamp_ms, reason="no face")
        n = int(lms.shape[0])
        if n < LMK.BASE_LANDMARKS:
            return GazeSample(valid=False, timestamp_ms=timestamp_ms,
                              reason=f"only {n} landmarks (need >= {LMK.BASE_LANDMARKS})")
        use_iris = LMK.has_iris(n)

        face_w = float(np.linalg.norm(lms[LMK.FACE_WIDTH_PAIR[0], :2] - lms[LMK.FACE_WIDTH_PAIR[1], :2]))
        if face_w < self.min_face_width:
            return GazeSample(valid=False, face_width=face_w, timestamp_ms=timestamp_ms,
                              reason=f"face too small ({face_w:.3f} < {self.min_face_width:.3f})")

        ears: Dict[str, float] = {}
        lids: Dict[str, float] = {}
        per_eye: Dict[str, Dict[str, float]] = {}
        geom: Dict[str, Dict[str, Any]] = {}

        # ---- pass 1: eye geometry + openness ---------------------------
        # Done before any pupil work so that a full blink still reports
        # blink=True. Callers need to tell "eyes shut" apart from "face
        # lost": the first is a momentary pause, the second is a reason to
        # stop trusting everything.
        for name, spec in EYE_SPECS.items():
            u, v, w = self._eye_axis(lms, spec)
            lat = lms[spec["lateral"], :2]
            med = lms[spec["medial"], :2]
            centre = 0.5 * (lat + med)

            p = np.stack([lms[i, :2] for i in spec["ear"]], axis=0)
            horiz = float(np.linalg.norm(p[0] - p[3]))
            vert = 0.5 * (float(np.linalg.norm(p[1] - p[5])) + float(np.linalg.norm(p[2] - p[4])))
            ear = vert / (2.0 * horiz) if horiz > _EPS else 0.0
            ears[name] = ear

            up, lo = spec["lids"]
            lid_dist = float(np.linalg.norm(lms[up, :2] - lms[lo, :2]))
            lids[name] = lid_dist / w if w > _EPS else 0.0

            geom[name] = {"spec": spec, "u": u, "v": v, "w": w, "centre": centre, "ear": ear}

        ear_left = ears.get("left", 0.0)
        ear_right = ears.get("right", 0.0)
        blink = (ear_left < self.blink_ear) and (ear_right < self.blink_ear)

        # ---- pass 2: pupil -> head-invariant gaze feature --------------
        fx: list = []
        fy: list = []
        for name, g in geom.items():
            spec, u, v, w, centre, ear = g["spec"], g["u"], g["v"], g["w"], g["centre"], g["ear"]
            pupil: Optional[Tuple[float, float]] = None
            if use_iris:
                c = lms[spec["iris"], :2]
                if self.iris_fallback_threshold and w > _EPS:
                    # Sanity check: the iris centre must sit inside the eye
                    # box. When the model loses an eye it can park the iris on
                    # the nose, which would swing the gaze estimate wildly.
                    ring = np.stack([lms[i, :2] for i in spec["ring"]], axis=0)
                    ring_r = float(np.mean(np.linalg.norm(ring - c, axis=1)))
                    if ring_r > 0.9 * w or abs(float(np.dot(c - centre, u))) > 0.75 * w:
                        c = None
                if c is not None:
                    pupil = (float(c[0]), float(c[1]))
            if pupil is None and rgb is not None:
                pupil = self._pupil_from_blob(rgb, lms, spec)
            if pupil is None and use_iris:
                c = lms[spec["iris"], :2]   # last resort: ungated iris centre
                pupil = (float(c[0]), float(c[1]))
            if pupil is None or w < _EPS:
                per_eye[name] = {"fx": 0.0, "fy": 0.0, "ear": ear, "ok": 0.0}
                continue

            pnt = np.asarray(pupil, dtype=np.float64)
            d = pnt - centre
            ex = float(np.dot(d, u)) / w
            ey = float(np.dot(d, v)) / w
            # A closed or nearly-closed eye gives a meaningless iris centre
            # (the model parks it wherever the lid crease is). Excluding it is
            # what makes wink-to-click usable: without this, every wink yanks
            # the averaged gaze estimate sideways and the cursor jumps.
            usable = ear > self.min_ear_for_gaze
            per_eye[name] = {"fx": ex, "fy": ey, "ear": ear, "ok": 1.0 if usable else 0.0,
                             "px": float(pnt[0]), "py": float(pnt[1]),
                             "cx": float(centre[0]), "cy": float(centre[1]),
                             "w": w}
            if usable:
                fx.append(ex)
                fy.append(ey)

        if not fx:
            return GazeSample(valid=False, face_width=face_w, timestamp_ms=timestamp_ms,
                              ear_left=ear_left, ear_right=ear_right,
                              lid_left=lids.get("left", 0.0), lid_right=lids.get("right", 0.0),
                              blink=blink, iris_used=use_iris, eyes=per_eye,
                              reason="blink" if blink else "no pupil resolved")

        gx = float(np.mean(fx))
        gy = float(np.mean(fy))

        mid_eye = 0.5 * (lms[LMK.RIGHT_EYE_MEDIAL, :2] + lms[LMK.LEFT_EYE_MEDIAL, :2])
        hx = float(mid_eye[0])
        hy = float(mid_eye[1])

        # Heuristic quality: reward a big, stable face and open eyes.
        size_q = clamp((face_w - self.min_face_width) / 0.20, 0.0, 1.0)
        open_q = clamp((min(ear_left, ear_right) - 0.12) / 0.14, 0.0, 1.0)
        both_q = float(np.mean([e.get("ok", 0.0) for e in per_eye.values()]))
        quality = clamp(0.5 * size_q + 0.3 * open_q + 0.2 * both_q, 0.0, 1.0)
        if blink:
            quality *= 0.35

        return GazeSample(
            valid=True and not blink,
            gx=gx, gy=gy, hx=hx, hy=hy,
            face_width=face_w,
            ear_left=ear_left, ear_right=ear_right,
            lid_left=lids.get("left", 0.0), lid_right=lids.get("right", 0.0),
            blink=blink,
            iris_used=use_iris,
            quality=quality,
            timestamp_ms=timestamp_ms,
            eyes=per_eye,
            reason="blink" if blink else "",
        )


# ==========================================================================
# Full pipeline
# ==========================================================================
class GazePipeline:
    """Turns :class:`GazeSample` into a stable screen-space cursor target."""

    def __init__(self, cfg: Dict[str, Any], screen_w: int, screen_h: int) -> None:
        self.cfg = cfg
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.calibration = None  # set via set_calibration()
        self.chain = build_filter_chain(cfg)
        self.deadzone = Deadzone(
            radius_px=float(get(cfg, "gaze.deadzone_px", 2.5)),
            soft=float(get(cfg, "gaze.deadzone_soft", 0.0)),
        )
        self._last_t: Optional[int] = None
        self._last_out: Optional[Tuple[float, float]] = None
        self._smoothing_scale = 1.0

    # ------------------------------------------------------------------
    def set_calibration(self, model) -> None:
        self.calibration = model
        self.reset()

    def set_screen_size(self, w: int, h: int) -> None:
        if (w, h) != (self.screen_w, self.screen_h):
            self.screen_w, self.screen_h = int(w), int(h)
            self.reset()

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Drop filter memory -- called after calibration or a tracking gap."""
        for f in self.chain:
            f.reset(x, y)
        self.deadzone.reset(x, y)
        self._last_t = None
        self._last_out = (x, y) if (x is not None and y is not None) else None

    def apply_config(self, cfg: Dict[str, Any]) -> None:
        """Live-update tunables (used by the slider overlay)."""
        self.cfg = cfg
        self.deadzone.set_params(
            radius_px=float(get(cfg, "gaze.deadzone_px", 2.5)),
            soft=float(get(cfg, "gaze.deadzone_soft", 0.0)),
        )
        for f in self.chain:
            if isinstance(f, MovingAverage):
                f.set_window(int(get(cfg, "gaze.moving_average_window", 4)))
            elif isinstance(f, OneEuroFilter2D):
                oe = get(cfg, "gaze.one_euro", {}) or {}
                f.set_params(
                    min_cutoff=float(oe.get("min_cutoff", 1.1)) / self._smoothing_scale,
                    beta=float(oe.get("beta", 0.012)),
                    d_cutoff=float(oe.get("d_cutoff", 1.0)),
                )
            elif isinstance(f, KalmanCV2D):
                k = get(cfg, "gaze.kalman", {}) or {}
                f.set_params(
                    process_noise=float(k.get("process_noise", 1e-3)) / (self._smoothing_scale ** 2),
                    measurement_noise=float(k.get("measurement_noise", 2.5e-2)) * self._smoothing_scale,
                )

    def set_smoothing_boost(self, factor: float) -> None:
        """Sniper mode leans on extra smoothing as well as a slower cursor."""
        factor = clamp(float(factor), 0.1, 20.0)
        if abs(factor - self._smoothing_scale) < 1e-6:
            return
        self._smoothing_scale = factor
        self.apply_config(self.cfg)

    # ------------------------------------------------------------------
    def process(self, sample: GazeSample) -> GazeState:
        if not sample.valid:
            return GazeState(valid=False, sample=sample, reason=sample.reason or "invalid")
        if self.calibration is None:
            return GazeState(valid=False, sample=sample, reason="not calibrated")

        nx, ny = self.calibration.apply(sample.gx, sample.gy, sample.hx, sample.hy)

        gain = float(get(self.cfg, "gaze.gain", 1.0))
        nx = 0.5 + (nx - 0.5) * gain + float(get(self.cfg, "gaze.offset_x", 0.0)) / max(1, self.screen_w)
        ny = 0.5 + (ny - 0.5) * gain + float(get(self.cfg, "gaze.offset_y", 0.0)) / max(1, self.screen_h)
        nx = clamp(nx, 0.0, 1.0)
        ny = clamp(ny, 0.0, 1.0)

        px, py = nx * self.screen_w, ny * self.screen_h

        t = sample.timestamp_ms
        if self._last_t is None or t <= self._last_t:
            dt = 1.0 / 60.0
        else:
            dt = clamp((t - self._last_t) / 1000.0, 1.0 / 240.0, 0.25)
        self._last_t = t

        speed = 0.0
        if self._last_out is not None:
            speed = math.hypot(px - self._last_out[0], py - self._last_out[1]) / max(dt, 1e-6)

        for f in self.chain:
            px, py = f(px, py, dt)

        inset = float(get(self.cfg, "mouse.screen_inset_px", 6))
        px = clamp(px, inset, max(inset, self.screen_w - inset))
        py = clamp(py, inset, max(inset, self.screen_h - inset))

        (ox, oy), moved = self.deadzone(px, py)
        self._last_out = (ox, oy)

        return GazeState(
            valid=True, x=ox, y=oy,
            nx=ox / max(1.0, float(self.screen_w)), ny=oy / max(1.0, float(self.screen_h)),
            speed=speed, moved=moved, sample=sample,
        )
