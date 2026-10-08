"""Synthetic MediaPipe-shaped landmark generators.

Used by the test suite and by ``tools/simulate.py`` to drive the *entire*
gaze/gesture/mouse pipeline without a webcam, without MediaPipe and without
the model bundles. This is also the fastest way to sanity-check a tuning
change: the geometry is exact, so if the state machines are wrong it shows up
immediately.

The generators produce arrays with exactly MediaPipe's layout:

* face -- ``(478, 3)`` normalized ``x, y, z``
* hand -- ``(21, 3)`` normalized ``x, y, z``

and a small :class:`FakeHands` wrapper matching the neutral hand container in
:mod:`mp_backend`.

Coordinate conventions match a **non-mirrored** webcam frame: the subject's
right eye (landmarks 33/133, iris 468) is on the image-left, their left eye
(263/362, iris 473) is on the image-right. Positive ``gaze_x`` means "looking
towards image-right", which is the same convention :mod:`gaze` uses
internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import landmarks as LMK

__all__ = ["FakeHands", "make_face", "make_hand", "hand_at_point"]


@dataclass
class FakeHands:
    """Duck-types :class:`mp_backend.HandObservation`."""

    landmarks: Optional[np.ndarray] = None
    handedness: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# Face
# --------------------------------------------------------------------------
def make_face(centre: Tuple[float, float] = (0.5, 0.42),
              face_width: float = 0.36,
              gaze: Tuple[float, float] = (0.0, 0.0),
              ear_left: float = 0.30,
              ear_right: float = 0.30,
              n: int = LMK.IRIS_LANDMARKS,
              yaw: float = 0.0) -> np.ndarray:
    """Build an internally-consistent face landmark array.

    Parameters
    ----------
    centre:
        Mid-eye position in normalized frame coordinates.
    face_width:
        Distance between landmarks 234 and 454 (the pre-auricular pair), which
        is what every proximity threshold in :mod:`gestures` scales from.
    gaze:
        ``(gx, gy)`` in roughly -1..1. ``+gx`` looks towards image-right,
        ``+gy`` looks down.
    ear_left / ear_right:
        Target eye aspect ratios. ``0.30`` is a relaxed open eye, ``< 0.15``
        reads as closed, and setting one low while leaving the other open
        produces a wink.
    yaw:
        Rigid horizontal head rotation, applied about ``centre``. Lets tests
        verify that head turn is absorbed rather than mistaken for gaze.
    """
    lms = np.zeros((max(n, LMK.IRIS_LANDMARKS), 3), dtype=np.float64)
    cx, cy = float(centre[0]), float(centre[1])
    W = float(face_width)
    ew = 0.13 * W                      # eye half-width
    eye_dx = 0.21 * W                  # eye centre offset from face centre

    def put(idx, x, y):
        lms[idx, 0], lms[idx, 1] = x, y

    # --- eye centres (subject right = image left) ----------------------
    er = (cx - eye_dx, cy)
    el = (cx + eye_dx, cy)

    # --- corners -------------------------------------------------------
    put(LMK.RIGHT_EYE_LATERAL, er[0] - ew, er[1])   # 33
    put(LMK.RIGHT_EYE_MEDIAL, er[0] + ew, er[1])    # 133
    put(LMK.LEFT_EYE_MEDIAL, el[0] - ew, el[1])     # 362
    put(LMK.LEFT_EYE_LATERAL, el[0] + ew, el[1])    # 263

    # --- lids / EAR points --------------------------------------------
    # EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). With the corners 2*ew apart and
    # the lids placed +/-h about the eye centre, that works out to h/(2*ew),
    # so h has to be 2*ear*ew for the measured EAR to equal the requested one.
    def eye_lids(centre_eye, ear, spec, mirror=False):
        h = 2.0 * float(ear) * ew
        s = -1.0 if mirror else 1.0
        p = list(spec)
        # p1, p4 are the corners (already set); place p2,p3 (upper) and
        # p6,p5 (lower) symmetrically about the eye centre.
        put(p[1], centre_eye[0] + s * 0.4 * ew, centre_eye[1] - h)
        put(p[2], centre_eye[0] - s * 0.4 * ew, centre_eye[1] - h)
        put(p[4], centre_eye[0] - s * 0.4 * ew, centre_eye[1] + h)
        put(p[5], centre_eye[0] + s * 0.4 * ew, centre_eye[1] + h)

    eye_lids(er, ear_right, LMK.RIGHT_EYE_EAR, mirror=False)
    eye_lids(el, ear_left, LMK.LEFT_EYE_EAR, mirror=True)
    hr = 2.0 * float(ear_right) * ew
    hl = 2.0 * float(ear_left) * ew
    put(LMK.RIGHT_EYE_LIDS[0], er[0], er[1] - hr)   # 159 upper
    put(LMK.RIGHT_EYE_LIDS[1], er[0], er[1] + hr)   # 145 lower
    put(LMK.LEFT_EYE_LIDS[0], el[0], el[1] - hl)    # 386 upper
    put(LMK.LEFT_EYE_LIDS[1], el[0], el[1] + hl)    # 374 lower

    # --- iris ----------------------------------------------------------
    # Both eyes translate the same way in image space for a given gaze.
    ix = float(gaze[0]) * ew * 0.55
    iy = float(gaze[1]) * ew * 0.45
    for centre_eye, c_idx, ring in ((er, LMK.RIGHT_IRIS_CENTER, LMK.RIGHT_IRIS_RING),
                                    (el, LMK.LEFT_IRIS_CENTER, LMK.LEFT_IRIS_RING)):
        px, py = centre_eye[0] + ix, centre_eye[1] + iy
        put(c_idx, px, py)
        r = 0.42 * ew
        for k, idx in enumerate(ring):
            a = k * (np.pi / 2.0)
            put(idx, px + r * np.cos(a), py + r * np.sin(a))

    # --- temples & face-width pair ------------------------------------
    put(LMK.FACE_WIDTH_PAIR[0], cx - 0.5 * W, cy)   # 234
    put(LMK.FACE_WIDTH_PAIR[1], cx + 0.5 * W, cy)   # 454
    right_cluster = [(-0.50, 0.00), (-0.45, -0.12), (-0.42, 0.10), (-0.36, 0.02)]
    left_cluster = [(0.50, 0.00), (0.45, -0.12), (0.42, 0.10), (0.36, 0.02)]
    for idx, (dx, dy) in zip(LMK.RIGHT_TEMPLE, right_cluster):
        put(idx, cx + dx * W, cy + dy * W)
    for idx, (dx, dy) in zip(LMK.LEFT_TEMPLE, left_cluster):
        put(idx, cx + dx * W, cy + dy * W)

    # --- a few structural anchors -------------------------------------
    put(LMK.NOSE_TIP, cx, cy + 0.18 * W)
    put(LMK.NOSE_BRIDGE, cx, cy - 0.05 * W)
    put(LMK.CHIN, cx, cy + 0.45 * W)
    put(LMK.FOREHEAD, cx, cy - 0.30 * W)

    # Fill every unused landmark with a plausible value inside the face oval
    # so nothing downstream can trip over a stray origin point.
    unfilled = np.all(lms[:, :2] == 0.0, axis=1)
    if unfilled.any():
        rng = np.random.default_rng(1234)
        ang = rng.uniform(0, 2 * np.pi, size=int(unfilled.sum()))
        rad = rng.uniform(0.0, 0.45, size=int(unfilled.sum()))
        lms[unfilled, 0] = cx + rad * np.cos(ang) * W
        lms[unfilled, 1] = cy + rad * np.sin(ang) * W * 1.3

    if abs(yaw) > 1e-9:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        xy = lms[:, :2] - np.array([cx, cy])
        lms[:, :2] = xy @ R.T + np.array([cx, cy])
    return lms[:n]


# --------------------------------------------------------------------------
# Hands
# --------------------------------------------------------------------------
def make_hand(wrist: Tuple[float, float], span: float = 0.10,
              extended: int = 4, point: Optional[Tuple[float, float]] = None,
              direction: Tuple[float, float] = (0.0, -1.0)) -> np.ndarray:
    """Build a 21-landmark hand.

    ``extended`` is how many of the four fingers are straight (4 = open palm,
    0 = fist). When ``point`` is given, the index fingertip is placed exactly
    there, which is what the temple-touch tests need.
    """
    lms = np.zeros((21, 3), dtype=np.float64)
    wx, wy = float(wrist[0]), float(wrist[1])
    d = np.asarray(direction, dtype=np.float64)
    dn = np.linalg.norm(d)
    if dn < 1e-9:
        d = np.array([0.0, -1.0])
    else:
        d = d / dn
    lateral = np.array([-d[1], d[0]], dtype=np.float64)
    span = float(span)

    lms[LMK.WRIST, :2] = (wx, wy)

    # (lateral, along) offsets in units of the hand span, for the four fingers
    # that palm_openness() inspects. Curling multiplies the "along" term,
    # which is what moves the tip closer to the wrist than its PIP joint.
    fingers = [
        (LMK.INDEX_TIP, LMK.INDEX_PIP, (0.34, 1.00), (0.30, 0.55)),
        (LMK.MIDDLE_TIP, LMK.MIDDLE_PIP, (0.06, 1.06), (0.05, 0.58)),
        (LMK.RING_TIP, LMK.RING_PIP, (-0.20, 0.98), (-0.18, 0.55)),
        (LMK.PINKY_TIP, LMK.PINKY_PIP, (-0.44, 0.86), (-0.40, 0.48)),
    ]
    for i, (tip_idx, pip_idx, (t_lat, t_along), (p_lat, p_along)) in enumerate(fingers):
        ext = 1.0 if i < int(extended) else 0.42
        lms[tip_idx, :2] = ((wx, wy) + d * (t_along * ext * span)
                            + lateral * (t_lat * span))
        lms[pip_idx, :2] = ((wx, wy) + d * (p_along * span)
                            + lateral * (p_lat * span))

    # Thumb: tip stays near the wrist even when open, which is exactly why
    # palm_openness() excludes it.
    lms[LMK.THUMB_TIP, :2] = (wx, wy) + d * (0.35 * span) + lateral * (0.55 * span)

    if point is not None:
        shift = np.asarray(point, dtype=np.float64) - lms[LMK.INDEX_TIP, :2]
        lms[:, :2] += shift
    return lms


def hand_at_point(point: Tuple[float, float], span: float = 0.10,
                  extended: int = 2) -> np.ndarray:
    """A hand whose index fingertip lands exactly on ``point``."""
    return make_hand(wrist=(point[0], point[1] + span * 1.4), span=span,
                     extended=extended, point=point, direction=(0.0, -1.0))


def hands_from(items: Sequence[Tuple[np.ndarray, str]]
               ) -> FakeHands:
    """Stack ``(landmarks, label)`` pairs into a :class:`FakeHands`."""
    if not items:
        return FakeHands(landmarks=None, handedness=[], scores=[])
    lms = np.stack([np.asarray(a, dtype=np.float64) for a, _ in items], axis=0)
    return FakeHands(landmarks=lms, handedness=[b for _, b in items],
                     scores=[1.0] * len(items))
