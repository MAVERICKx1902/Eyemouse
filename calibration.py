"""9-point calibration: collection, polynomial fit, persistence.

Why a polynomial map?
---------------------
A webcam gives no metric depth, so the relationship between "where the iris
sits in its socket" and "where the user is looking on screen" is nonlinear
(eye rotation, head sway, lens distortion and the fact that the user does not
sit perfectly square to the camera all contribute). Fitting a low-order
polynomial to 9 known points absorbs all of that without needing a 3D eye
model.

The fit is deliberately modest:

* ``linear``         -- 3 terms  (fast, robust, less accurate at the edges)
* ``quadratic``      -- 6 terms  (default; 9 points leaves 3 DOF)
* ``quadratic_head`` -- 8 terms  (adds head position as linear correction)

Inputs are standardized before the polynomial is built and clamped to
+/-3 sigma at apply time, so the map cannot explode when the user looks
further off-centre than they did during calibration. Outputs are clamped to
the screen.

Everything here is pure numpy + json. The interactive procedure is split into
a testable state machine (:class:`CalibrationSession`) and thin OpenCV
rendering (:func:`draw_calibration_screen`), so calibration logic can be
exercised in CI without a display.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import get
from util import Clock, clamp

CALIBRATION_VERSION = 3

__all__ = [
    "CalibrationModel",
    "CalibrationSession",
    "calibration_grid",
    "fit_calibration",
    "load_calibration",
    "save_calibration",
    "draw_calibration_screen",
]


# ==========================================================================
# Target grid
# ==========================================================================
def calibration_grid(points: int = 9, inset_ratio: float = 0.12,
                     screen_w: int = 1920, screen_h: int = 1080
                     ) -> List[Tuple[int, int]]:
    """Screen-pixel targets, ordered for a natural left-right, top-bottom sweep.

    ``points`` may be 5 (corners + centre), 9 (a 3x3 grid) or 13 (the 3x3 plus
    four quarter points, which sample the diagonal regions a 3x3 misses). Any
    other count falls back to the closest square grid.

    The inset keeps targets clear of the scroll bands at the very top and
    bottom of the screen -- without it, the outermost calibration points would
    trigger edge scrolling mid-calibration.
    """
    pts = max(3, int(points))
    ix = clamp(float(inset_ratio), 0.0, 0.4)

    def axis(frac: float) -> float:
        return ix + (1.0 - 2.0 * ix) * frac

    if pts == 5:
        fracs = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)]
    elif pts == 9:
        fracs = [(c, r) for r in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)]
    elif pts == 13:
        fracs = [(c, r) for r in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)]
        fracs += [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    else:
        side = max(2, int(math.ceil(math.sqrt(pts))))
        levels = [i / (side - 1) for i in range(side)]
        fracs = [(c, r) for r in levels for c in levels][:pts]

    out: List[Tuple[int, int]] = []
    seen = set()
    for fx, fy in fracs:
        p = (int(round(axis(fx) * screen_w)), int(round(axis(fy) * screen_h)))
        if p in seen:      # can only happen on very small screens
            continue
        seen.add(p)
        out.append(p)
    return out


# ==========================================================================
# Model
# ==========================================================================
TERM_SETS: Dict[str, Tuple[str, ...]] = {
    "linear": ("1", "gx", "gy"),
    "quadratic": ("1", "gx", "gy", "gx2", "gxgy", "gy2"),
    "quadratic_head": ("1", "gx", "gy", "gx2", "gxgy", "gy2", "hx", "hy"),
}


def _design(features: np.ndarray, model: str) -> np.ndarray:
    """``features`` is ``(N,4)`` = [gx, gy, hx, hy] -> design matrix ``(N,T)``.

    ``features`` must already be standardized.
    """
    gx, gy, hx, hy = features[:, 0], features[:, 1], features[:, 2], features[:, 3]
    cols = {
        "1": np.ones_like(gx),
        "gx": gx,
        "gy": gy,
        "gx2": gx * gx,
        "gy2": gy * gy,
        "gxgy": gx * gy,
        "hx": hx,
        "hy": hy,
    }
    names = TERM_SETS[model]
    return np.stack([cols[n] for n in names], axis=1)


@dataclass
class CalibrationModel:
    """Fitted gaze -> screen mapping. Serializable to plain JSON."""

    model: str = "quadratic"
    # Standardization stats for [gx, gy, hx, hy].
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4))
    std: np.ndarray = field(default_factory=lambda: np.ones(4))
    # Coefficient vectors producing normalized screen coords in 0..1.
    coef_x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    coef_y: np.ndarray = field(default_factory=lambda: np.zeros(6))
    screen_w: int = 1920
    screen_h: int = 1080
    n_points: int = 9
    rms_norm: float = float("nan")
    rms_px: float = float("nan")
    max_px: float = float("nan")
    created: float = 0.0
    points: List[Tuple[int, int]] = field(default_factory=list)
    per_point_px: List[float] = field(default_factory=list)
    samples_per_point: List[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _standardize(self, gx: float, gy: float, hx: float, hy: float) -> np.ndarray:
        raw = np.array([gx, gy, hx, hy], dtype=np.float64)
        z = (raw - self.mean) / self.std
        # Bound extrapolation: beyond 3 sigma the polynomial is meaningless
        # and a quadratic would shoot the cursor to the moon.
        return np.clip(z, -3.0, 3.0)

    def apply(self, gx: float, gy: float, hx: float = 0.5, hy: float = 0.5
              ) -> Tuple[float, float]:
        """Map gaze features to normalized screen coordinates in ``[0, 1]``."""
        z = self._standardize(gx, gy, hx, hy)
        row = _design(z[None, :], self.model)[0]
        nx = float(np.dot(row, self.coef_x))
        ny = float(np.dot(row, self.coef_y))
        if not math.isfinite(nx):
            nx = 0.5
        if not math.isfinite(ny):
            ny = 0.5
        return clamp(nx, 0.0, 1.0), clamp(ny, 0.0, 1.0)

    def apply_px(self, gx: float, gy: float, hx: float = 0.5, hy: float = 0.5
                 ) -> Tuple[float, float]:
        nx, ny = self.apply(gx, gy, hx, hy)
        return nx * self.screen_w, ny * self.screen_h

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": CALIBRATION_VERSION,
            "model": self.model,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "coef_x": [float(v) for v in self.coef_x],
            "coef_y": [float(v) for v in self.coef_y],
            "screen_w": int(self.screen_w),
            "screen_h": int(self.screen_h),
            "n_points": int(self.n_points),
            "rms_norm": None if not math.isfinite(self.rms_norm) else float(self.rms_norm),
            "rms_px": None if not math.isfinite(self.rms_px) else float(self.rms_px),
            "max_px": None if not math.isfinite(self.max_px) else float(self.max_px),
            "created": float(self.created or time.time()),
            "points": [[int(a), int(b)] for a, b in self.points],
            "per_point_px": [float(v) for v in self.per_point_px],
            "samples_per_point": [int(v) for v in self.samples_per_point],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationModel":
        version = int(data.get("version", 1))
        if version > CALIBRATION_VERSION:
            raise ValueError(
                f"Calibration file version {version} is newer than this build "
                f"supports ({CALIBRATION_VERSION}). Re-run calibration."
            )
        model = str(data.get("model", "quadratic"))
        if model not in TERM_SETS:
            model = "quadratic"
        n_terms = len(TERM_SETS[model])

        def vec(key: str, default_len: int) -> np.ndarray:
            raw = data.get(key) or []
            arr = np.asarray(raw, dtype=np.float64).ravel()
            if arr.size != default_len:
                out = np.zeros(default_len, dtype=np.float64)
                out[: min(default_len, arr.size)] = arr[: min(default_len, arr.size)]
                return out
            return arr

        m = cls(
            model=model,
            mean=vec("mean", 4),
            std=np.maximum(vec("std", 4), 1e-6),
            coef_x=vec("coef_x", n_terms),
            coef_y=vec("coef_y", n_terms),
            screen_w=int(data.get("screen_w", 1920)),
            screen_h=int(data.get("screen_h", 1080)),
            n_points=int(data.get("n_points", 9)),
            created=float(data.get("created", 0.0)),
            points=[(int(p[0]), int(p[1])) for p in (data.get("points") or []) if len(p) >= 2],
            per_point_px=[float(v) for v in (data.get("per_point_px") or [])],
            samples_per_point=[int(v) for v in (data.get("samples_per_point") or [])],
        )
        for key, attr in (("rms_norm", "rms_norm"), ("rms_px", "rms_px"), ("max_px", "max_px")):
            val = data.get(key)
            if val is not None:
                setattr(m, attr, float(val))
        return m

    # ------------------------------------------------------------------
    def report(self) -> str:
        """Human-readable accuracy summary, printed after calibration."""
        lines = [
            f"calibration: {self.model} model, {self.n_points} points, "
            f"{int(self.screen_w)}x{int(self.screen_h)}"
        ]
        if math.isfinite(self.rms_px):
            grade = ("excellent" if self.rms_px < 30 else
                     "good" if self.rms_px < 60 else
                     "usable" if self.rms_px < 110 else "poor")
            lines.append(f"  RMS error : {self.rms_px:6.1f} px  ({grade})")
            lines.append(f"  worst point: {self.max_px:6.1f} px")
            if self.rms_px >= 60:
                lines.append("  -> Improve lighting (even, front-facing, no backlight),")
                lines.append("     sit square to the camera ~50-70cm away, keep your")
                lines.append("     head still, and look AT each dot, not near it.")
        if self.per_point_px:
            worst = int(np.argmax(self.per_point_px))
            if worst < len(self.points):
                lines.append(f"  worst was point #{worst + 1} at {self.points[worst]}")
        return "\n".join(lines)


# ==========================================================================
# Fitting
# ==========================================================================
def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Least squares with a small L2 penalty (intercept left unpenalized)."""
    XtX = X.T @ X
    Xty = X.T @ y
    scale = float(np.trace(XtX)) / max(1, XtX.shape[0])
    pen = np.eye(XtX.shape[0]) * (ridge * max(scale, 1e-9))
    pen[0, 0] = 0.0  # never penalize the intercept
    try:
        return np.linalg.solve(XtX + pen, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtX + pen, Xty, rcond=None)[0]


def fit_calibration(rows: Sequence[Sequence[float]], model: str = "quadratic",
                    ridge: float = 1e-4, screen_w: int = 1920, screen_h: int = 1080,
                    points: Optional[Sequence[Tuple[int, int]]] = None,
                    ) -> CalibrationModel:
    """Fit gaze features to normalized screen coordinates.

    ``rows`` is a sequence of ``[gx, gy, hx, hy, sx_px, sy_px]``.

    If there are fewer distinct rows than the chosen model has terms, the
    model is automatically downgraded -- an under-determined fit produces
    wild extrapolation, and a slightly less accurate but stable map is far
    more useful than one that throws the cursor off-screen.
    """
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 6:
        raise ValueError(
            f"calibration needs at least 3 rows of [gx,gy,hx,hy,sx,sy]; got shape "
            f"{getattr(data, 'shape', None)}"
        )
    if model not in TERM_SETS:
        model = "quadratic"

    feats = data[:, :4]
    targets_px = data[:, 4:6]
    targets = np.stack([targets_px[:, 0] / max(1.0, float(screen_w)),
                        targets_px[:, 1] / max(1.0, float(screen_h))], axis=1)

    # Standardize. A zero-variance axis (e.g. a user who never moved their
    # head) gets std=1 so the term simply contributes nothing.
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    # Auto-downgrade when under-determined.
    n_distinct = len(np.unique(feats, axis=0))
    order = ["quadratic_head", "quadratic", "linear"]
    if model in order:
        idx = order.index(model)
        candidates = order[idx:]
    else:
        candidates = [model]
    for cand in candidates:
        if n_distinct >= len(TERM_SETS[cand]):
            model = cand
            break
    else:
        model = "linear"

    Z = np.clip((feats - mean) / std, -3.0, 3.0)
    X = _design(Z, model)

    coef_x = _ridge_solve(X, targets[:, 0], ridge)
    coef_y = _ridge_solve(X, targets[:, 1], ridge)

    m = CalibrationModel(
        model=model, mean=mean, std=std, coef_x=coef_x, coef_y=coef_y,
        screen_w=int(screen_w), screen_h=int(screen_h),
        n_points=len(points) if points else data.shape[0],
        created=time.time(),
        points=list(points) if points else [],
    )

    pred = np.stack([X @ coef_x, X @ coef_y], axis=1)
    err = (pred - targets) * np.array([screen_w, screen_h], dtype=np.float64)
    dist = np.linalg.norm(err, axis=1)
    m.rms_norm = float(np.sqrt(np.mean(((pred - targets) ** 2).sum(axis=1))))
    m.rms_px = float(np.sqrt(np.mean(dist ** 2)))
    m.max_px = float(dist.max()) if dist.size else float("nan")

    if points is not None and len(points) == data.shape[0]:
        m.per_point_px = [float(v) for v in dist]
    return m


# ==========================================================================
# Persistence
# ==========================================================================
def save_calibration(model: CalibrationModel, path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def load_calibration(path: str) -> Optional[CalibrationModel]:
    """Return ``None`` when the file is missing or unusable (never raises)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return CalibrationModel.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[calibration] ignoring unreadable {path}: {exc}")
        return None


# ==========================================================================
# Interactive session (rendering-free state machine)
# ==========================================================================
class CalibrationSession:
    """Collects gaze samples for each calibration target.

    The caller drives it::

        session = CalibrationSession(grid, samples_per_point=30, discard_first=8)
        while not session.complete:
            sample = extractor.extract(landmarks)
            if sample.valid:
                session.add_sample(sample.gx, sample.gy, sample.hx, sample.hy)
            canvas = draw_calibration_screen(session, thumb=webcam_frame)
            cv2.imshow("Eyemouse calibration", canvas)

    Per point, the first ``discard_first`` samples are dropped (the eye is
    still settling after the saccade), then remaining samples are
    median-filtered to remove blinks and single-frame tracking glitches
    before being averaged into one observation.
    """

    def __init__(self, grid: Sequence[Tuple[int, int]], samples_per_point: int = 30,
                 discard_first: int = 8, clock: Optional[Clock] = None,
                 settle_ms: int = 250, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self.grid = [(int(x), int(y)) for x, y in grid]
        if len(self.grid) < 3:
            raise ValueError("calibration needs at least 3 target points")
        self.samples_per_point = max(5, int(samples_per_point))
        self.discard_first = max(0, int(discard_first))
        self.screen_w = max(1, int(screen_w))
        self.screen_h = max(1, int(screen_h))
        self._clock = clock or Clock()
        self.settle_ms = int(settle_ms)

        self.index = 0
        self.complete = False
        self._buf: List[Tuple[float, float, float, float]] = []
        self._observations: List[Optional[Tuple[float, float, float, float]]] = [None] * len(self.grid)
        self._counts: List[int] = [0] * len(self.grid)
        self._point_started = self._clock.ms()
        self.history: List[str] = []

    # ------------------------------------------------------------------
    @property
    def current_target(self) -> Tuple[int, int]:
        return self.grid[min(self.index, len(self.grid) - 1)]

    @property
    def progress(self) -> float:
        done = sum(1 for o in self._observations if o is not None)
        return done / float(len(self.grid))

    @property
    def settling(self) -> bool:
        return (self._clock.ms() - self._point_started) < self.settle_ms

    def point_progress(self) -> float:
        n = max(0, len(self._buf) - self.discard_first)
        need = max(1, self.samples_per_point - self.discard_first)
        return clamp(n / float(need), 0.0, 1.0)

    # ------------------------------------------------------------------
    def add_sample(self, gx: float, gy: float, hx: float, hy: float) -> bool:
        """Record one gaze observation for the current target.

        Returns True when the point has just been completed.
        """
        if self.complete:
            return False
        if not all(math.isfinite(v) for v in (gx, gy, hx, hy)):
            return False
        if self.settling:
            return False
        self._buf.append((float(gx), float(gy), float(hx), float(hy)))
        if len(self._buf) >= self.samples_per_point:
            return self._finalize_point()
        return False

    def skip_point(self) -> bool:
        """Abandon the current target (e.g. the tracker lost the face)."""
        self._buf.clear()
        self._observations[self.index] = None
        return self._advance()

    def _finalize_point(self) -> bool:
        arr = np.asarray(self._buf[self.discard_first:], dtype=np.float64)
        self._counts[self.index] = int(arr.shape[0])
        if arr.shape[0] < 3:
            obs = arr.mean(axis=0) if arr.shape[0] else np.zeros(4)
        else:
            # Reject samples further than 3 * MAD from the per-axis median.
            med = np.median(arr, axis=0)
            mad = np.median(np.abs(arr - med), axis=0)
            scale = np.where(mad < 1e-9, np.std(arr, axis=0), mad * 1.4826)
            scale = np.where(scale < 1e-9, 1.0, scale)
            z = np.abs(arr - med) / scale
            keep = np.all(z <= 3.0, axis=1)
            if keep.sum() < max(3, int(0.3 * arr.shape[0])):
                keep = np.ones(arr.shape[0], dtype=bool)  # too aggressive; keep all
            obs = arr[keep].mean(axis=0)
            dropped = int((~keep).sum())
            if dropped:
                self.history.append(f"point {self.index + 1}: dropped {dropped} outlier sample(s)")
        self._observations[self.index] = tuple(float(v) for v in obs)
        self._buf.clear()
        return self._advance()

    def _advance(self) -> bool:
        self.index += 1
        self._point_started = self._clock.ms()
        if self.index >= len(self.grid):
            self.complete = True
            return True
        return False

    # ------------------------------------------------------------------
    def is_point_done(self, i: int) -> bool:
        """True once target ``i`` has a recorded observation."""
        return 0 <= i < len(self._observations) and self._observations[i] is not None

    def observations(self) -> List[Tuple[Tuple[float, float, float, float], Tuple[int, int]]]:
        """Valid (feature, target) pairs, skipping abandoned points."""
        out = []
        for obs, target in zip(self._observations, self.grid):
            if obs is not None:
                out.append((obs, target))
        return out

    def build(self, model: str = "quadratic", ridge: float = 1e-4,
              screen_w: int = 1920, screen_h: int = 1080) -> CalibrationModel:
        obs = self.observations()
        if len(obs) < 3:
            raise ValueError(
                f"only {len(obs)} calibration point(s) collected; need at least 3. "
                f"Re-run calibration and keep your face in frame."
            )
        rows, points = [], []
        for feat, (sx, sy) in obs:
            rows.append([feat[0], feat[1], feat[2], feat[3], float(sx), float(sy)])
            points.append((sx, sy))
        counts = [self._counts[i] for i, o in enumerate(self._observations) if o is not None]
        m = fit_calibration(rows, model=model, ridge=ridge,
                            screen_w=screen_w, screen_h=screen_h, points=points)
        m.samples_per_point = counts
        return m


# ==========================================================================
# Rendering helpers (OpenCV; guarded so headless tests never import it)
# ==========================================================================
def draw_calibration_screen(session: CalibrationSession, thumb: Optional[np.ndarray] = None,
                            radius: int = 14, canvas_w: int = 0, canvas_h: int = 0
                            ) -> Optional[np.ndarray]:
    """Build the full-screen calibration canvas.

    The target is drawn at its **exact screen pixel** on a canvas the size of
    the display, which is why calibration runs in a full-screen window: any
    proportional rescaling would put the dot somewhere other than where the
    user is told to look. ``thumb`` (a webcam frame) is inset in a corner so
    the user can confirm their face is still tracked.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        return None
    W = int(canvas_w) if canvas_w > 0 else session.screen_w
    H = int(canvas_h) if canvas_h > 0 else session.screen_h
    canvas = np.full((H, W, 3), 18, dtype=np.uint8)

    sx = session.screen_w / max(1.0, float(W))
    sy = session.screen_h / max(1.0, float(H))
    tx, ty = session.current_target
    px = int(clamp(tx / sx, 0, W - 1))
    py = int(clamp(ty / sy, 0, H - 1))

    # Non-active targets, so the user can see the whole path ahead of time.
    for i, (gx_, gy_) in enumerate(session.grid):
        if i == session.index:
            continue
        qx = int(clamp(gx_ / sx, 0, W - 1))
        qy = int(clamp(gy_ / sy, 0, H - 1))
        done = session.is_point_done(i)
        cv2.circle(canvas, (qx, qy), 5, (90, 90, 90) if not done else (60, 130, 70), -1, cv2.LINE_AA)

    title = f"Calibration {session.index + 1} / {len(session.grid)}"
    hint = ("Look AT the orange dot and hold still" if session.settling
            else "Keep looking at the dot - collecting...")
    cv2.putText(canvas, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, hint, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1, cv2.LINE_AA)
    bar_w = int(W * 0.30)
    cv2.rectangle(canvas, (16, 74), (16 + bar_w, 82), (60, 60, 60), -1)
    cv2.rectangle(canvas, (16, 74), (16 + int(bar_w * session.progress), 82), (80, 200, 120), -1)

    if thumb is not None and thumb.size:
        th = int(H * 0.22)
        scale = th / float(max(1, thumb.shape[0]))
        tw = max(1, int(thumb.shape[1] * scale))
        small = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_AREA)
        x0, y0 = W - tw - 16, H - th - 16
        canvas[y0:y0 + th, x0:x0 + tw] = small
        cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + th), (120, 120, 120), 1)

    # The active target is drawn LAST, on top of the HUD and the thumbnail.
    # On a short screen the top row of targets sits right where the progress
    # bar is, and a partially hidden dot silently costs calibration accuracy
    # -- the user looks at the visible part, not the true centre.
    pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)
    r = int(radius * (1.0 + 0.22 * pulse))
    prog = session.point_progress()
    if prog > 0:
        cv2.ellipse(canvas, (px, py), (r + 20, r + 20), -90, 0, 360.0 * prog,
                    (80, 255, 120), 3, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), r + 12, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), r, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), max(2, r // 3), (25, 25, 25), -1, cv2.LINE_AA)
    return canvas
