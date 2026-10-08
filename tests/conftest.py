"""Shared pytest fixtures.

Everything here runs without a webcam, without a display and without the
MediaPipe model bundles. ``cv2`` may be the headless build, so no test calls
``imshow``/``waitKey``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as config_mod          # noqa: E402
import mp_backend as mpb             # noqa: E402
from synthetic import make_face      # noqa: E402
from util import Clock               # noqa: E402

SCREEN = (1600, 900)


@pytest.fixture
def cfg():
    """Validated default config (no file on disk is read or written)."""
    c = config_mod.load_config(os.path.join(ROOT, "__does_not_exist__.json"))
    c["_meta"]["notes"] = []
    return c


@pytest.fixture
def clock():
    """Deterministic clock: time only advances when a test says so."""
    return Clock(manual=True)


@pytest.fixture
def face():
    """A neutral, forward-facing synthetic face with both eyes open."""
    return make_face(centre=(0.5, 0.42), face_width=0.36, gaze=(0.0, 0.0),
                     ear_left=0.30, ear_right=0.30)


class FakeTracker(mpb._BackendBase):
    """Returns the observation a test assigned last; ignores the image."""

    api = "fake"

    def __init__(self) -> None:
        self.face_obs = mpb.FaceObservation()
        self.hand_obs = mpb.HandObservation()
        self.face_calls = 0
        self.hand_calls = 0
        self.closed = False

    def set_face(self, landmarks=None, present=True):
        self.face_obs = mpb.FaceObservation(
            landmarks=landmarks, present=present and landmarks is not None,
            n_landmarks=0 if landmarks is None else int(landmarks.shape[0]))

    def set_hands(self, hands=None):
        if hands is None or getattr(hands, "landmarks", None) is None:
            self.hand_obs = mpb.HandObservation(present=False)
        else:
            self.hand_obs = mpb.HandObservation(
                landmarks=hands.landmarks, handedness=list(hands.handedness),
                scores=list(hands.scores), present=True)

    def process_face(self, bgr, timestamp_ms):
        self.face_calls += 1
        self.face_obs.timestamp_ms = int(timestamp_ms)
        return self.face_obs

    def process_hands(self, bgr, timestamp_ms):
        self.hand_calls += 1
        self.hand_obs.timestamp_ms = int(timestamp_ms)
        return self.hand_obs

    def close(self):
        self.closed = True


@pytest.fixture
def tracker():
    return FakeTracker()


@pytest.fixture
def frame():
    """A small dummy BGR image; contents are irrelevant to the fakes."""
    return np.zeros((240, 320, 3), dtype=np.uint8)
