"""Canonical MediaPipe Face Mesh landmark indices used by Eyemouse.

Every index below was verified against ``mediapipe.tasks.python.vision
.FaceLandmarksConnections`` rather than copied from a blog post, so the
left/right grouping is MediaPipe's own (anatomical, i.e. the *subject's*
left and right -- not the image's).

MediaPipe ships two landmark counts:

* ``468`` -- base Face Mesh (no iris refinement).
* ``478`` -- Face Mesh + Iris. The ``face_landmarker.task`` bundle used by
  the Tasks API emits 478, and the legacy ``refine_landmarks=True``
  solutions API does too.

Iris indices (468..477) only exist in the 478-landmark model, so
:func:`has_iris` is checked at runtime and :mod:`gaze` falls back to a
lid/corner-based pupil estimate when they are missing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Landmark counts
# --------------------------------------------------------------------------
BASE_LANDMARKS = 468
IRIS_LANDMARKS = 478

# --------------------------------------------------------------------------
# Iris (present only with the 478-landmark model)
# --------------------------------------------------------------------------
# 468 / 473 are the iris *centres*; the four following indices are the ring.
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_CENTER = 473
LEFT_IRIS_RING = (474, 475, 476, 477)

# --------------------------------------------------------------------------
# Eye corners. "lateral" = towards the temple, "medial" = towards the nose.
# --------------------------------------------------------------------------
RIGHT_EYE_LATERAL = 33
RIGHT_EYE_MEDIAL = 133
LEFT_EYE_LATERAL = 263
LEFT_EYE_MEDIAL = 362

# --------------------------------------------------------------------------
# Eye aspect ratio (EAR) landmark sets, ordered p1..p6 so that
#     EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
# p1/p4 are the outer/inner corners and p2,p3 / p5,p6 are upper/lower lids.
# --------------------------------------------------------------------------
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 380, 373)

# Simple vertical lid pairs, used as a robustness cross-check on EAR.
RIGHT_EYE_LIDS = (159, 145)   # upper, lower
LEFT_EYE_LIDS = (386, 374)

# --------------------------------------------------------------------------
# Temple regions. The temple is the soft area between the outer eye corner,
# the eyebrow tail and the ear. We average a small cluster so a single noisy
# landmark cannot move the anchor.
# --------------------------------------------------------------------------
RIGHT_TEMPLE = (234, 162, 93, 127)
LEFT_TEMPLE = (454, 389, 323, 356)

# --------------------------------------------------------------------------
# Head / face geometry anchors
# --------------------------------------------------------------------------
NOSE_TIP = 1
NOSE_BRIDGE = 168
CHIN = 152
FOREHEAD = 10

# Pre-auricular points: a stable, near-constant measure of face width that
# is used to make every proximity threshold distance-invariant.
FACE_WIDTH_PAIR = (234, 454)

# --------------------------------------------------------------------------
# Hand landmarks (MediaPipe Hands, 21 points)
# --------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
# The fingers used for the open-palm test (the thumb is excluded because its
# tip sits close to the wrist even on an open hand).
PALM_FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)
# Fingertips considered when touching the temple -- the index finger and
# thumb are what people actually reach up with.
TOUCH_TIPS = (INDEX_TIP, MIDDLE_TIP, THUMB_TIP)


def has_iris(n_landmarks: int) -> bool:
    """True when the landmark array carries the 10 iris points."""
    return n_landmarks >= IRIS_LANDMARKS
