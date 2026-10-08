"""MediaPipe access layer.

MediaPipe removed the legacy ``mp.solutions`` API in the 1.x line: the
``py3-none`` wheels (0.10.30 and later, and all of 1.x) expose only the Tasks
API, while ``cp39``-``cp312`` wheels up to 0.10.21 still expose
``mp.solutions.face_mesh`` / ``mp.solutions.hands``. Rather than force one,
this module detects what is installed and adapts:

===========================  ===================================  ============================
Installed mediapipe          API used                             Face model
===========================  ===================================  ============================
>= 0.10.30 (incl. 1.x)       Tasks ``FaceLandmarker`` /           ``models/face_landmarker.task``
                             ``HandLandmarker``                   ``models/hand_landmarker.task``
<= 0.10.21                   legacy ``solutions.face_mesh`` /     bundled in the wheel
                             ``solutions.hands``
===========================  ===================================  ============================

Both paths emit the same neutral containers (:class:`FaceObservation`,
:class:`HandObservation`) holding plain numpy arrays, so nothing else in the
project imports MediaPipe and everything downstream stays unit-testable.

MediaPipe is imported *lazily* (inside :func:`create_tracker`) so this module
can be imported -- and its dataclasses reused -- on a machine with no
MediaPipe at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import get

MODEL_DIRNAME = "models"
FACE_MODEL_NAME = "face_landmarker.task"
HAND_MODEL_NAME = "hand_landmarker.task"

MODEL_URLS = {
    FACE_MODEL_NAME: (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    HAND_MODEL_NAME: (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


class TrackerSetupError(RuntimeError):
    """Raised when MediaPipe or its models cannot be initialised."""


# ==========================================================================
# Neutral containers
# ==========================================================================
@dataclass
class FaceObservation:
    """One frame of face landmarks, normalized to ``0..1`` frame coordinates."""

    landmarks: Optional[np.ndarray] = None   # (N,3) float64
    present: bool = False
    timestamp_ms: int = 0
    n_landmarks: int = 0
    score: float = 0.0

    @property
    def has_iris(self) -> bool:
        return self.n_landmarks >= 478


@dataclass
class HandObservation:
    """One frame of hand landmarks.

    ``handedness`` holds MediaPipe's own labels (``"Left"`` / ``"Right"``).
    Those labels assume a mirrored selfie-style input; whether that matches a
    given webcam is a per-machine question, which is why the temple gesture
    defaults to keying off *which temple was touched* instead of the label
    (``clicks.temple.handedness_mode``).
    """

    landmarks: Optional[np.ndarray] = None   # (H,21,3) float64
    handedness: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    present: bool = False
    timestamp_ms: int = 0

    @property
    def count(self) -> int:
        return 0 if self.landmarks is None else int(self.landmarks.shape[0])


# ==========================================================================
# Environment probing
# ==========================================================================
def probe() -> Dict[str, Any]:
    """Report what MediaPipe offers here, without constructing anything."""
    info: Dict[str, Any] = {"installed": False, "version": None, "api": None,
                            "errors": []}
    try:
        import mediapipe as mp  # noqa: F401
    except Exception as exc:
        info["errors"].append(f"import mediapipe failed: {type(exc).__name__}: {exc}")
        return info
    info["installed"] = True
    info["version"] = getattr(mp, "__version__", "unknown")
    has_tasks = False
    try:
        from mediapipe.tasks.python import vision as _vision  # noqa: F401
        has_tasks = hasattr(_vision, "FaceLandmarker") and hasattr(_vision, "HandLandmarker")
    except Exception as exc:
        info["errors"].append(f"tasks API unavailable: {type(exc).__name__}: {exc}")
    has_solutions = hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions", None), "face_mesh")
    info["has_tasks"] = has_tasks
    info["has_solutions"] = has_solutions
    info["api"] = "tasks" if has_tasks else ("solutions" if has_solutions else None)
    if info["api"] is None:
        info["errors"].append(
            "neither the Tasks API (mediapipe>=0.10.30) nor the legacy "
            "solutions API (mediapipe<=0.10.21) is usable"
        )
    return info


def model_dir(root: Optional[str] = None) -> str:
    root = root or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, MODEL_DIRNAME)


def find_models(root: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Locate the model bundles, checking ``models/`` then the project root."""
    base = root or os.path.dirname(os.path.abspath(__file__))
    out: Dict[str, Optional[str]] = {}
    for name in (FACE_MODEL_NAME, HAND_MODEL_NAME):
        found = None
        for d in (os.path.join(base, MODEL_DIRNAME), base):
            cand = os.path.join(d, name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 1024:
                found = cand
                break
        out[name] = found
    return out


# ==========================================================================
# Backends
# ==========================================================================
def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR -> RGB as a contiguous uint8 array (what MediaPipe expects).

    ``cv2.cvtColor`` is SIMD-optimised; the slicing fallback keeps this module
    importable if OpenCV is missing. A non-contiguous negative-stride view can
    upset MediaPipe's native layer, so contiguity is enforced either way.
    """
    try:
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.ascontiguousarray(bgr[:, :, ::-1])


class _BackendBase:
    api = "none"

    def process_face(self, bgr: np.ndarray, timestamp_ms: int) -> FaceObservation:
        raise NotImplementedError

    def process_hands(self, bgr: np.ndarray, timestamp_ms: int) -> HandObservation:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TasksBackend(_BackendBase):
    """MediaPipe Tasks API (VIDEO running mode)."""

    api = "tasks"

    def __init__(self, face_model: str, hand_model: str, cfg: Dict[str, Any]) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self._vision = vision
        self._last_ts = -1
        self._hand_last_ts = -1

        min_det = float(get(cfg, "mediapipe.min_detection_confidence", 0.5))
        min_trk = float(get(cfg, "mediapipe.min_tracking_confidence", 0.5))
        delegate = get(cfg, "mediapipe.delegate", "cpu")

        def opts(path: str) -> Any:
            bo = BaseOptions(model_asset_path=path)
            if str(delegate).lower() == "gpu":
                try:
                    bo = BaseOptions(model_asset_path=path,
                                     delegate=BaseOptions.Delegate.GPU)
                except Exception:
                    bo = BaseOptions(model_asset_path=path)
            return bo

        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=opts(face_model),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_det,
                min_face_presence_confidence=min_det,
                min_tracking_confidence=min_trk,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=opts(hand_model),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_det,
                min_hand_presence_confidence=min_det,
                min_tracking_confidence=min_trk,
            )
        )

    # ------------------------------------------------------------------
    def _image(self, bgr: np.ndarray) -> Any:
        mp = self._mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_to_rgb(bgr))

    @staticmethod
    def _mono(last: int, ts: int) -> int:
        """VIDEO mode rejects non-increasing timestamps."""
        return ts if ts > last else last + 1

    def process_face(self, bgr: np.ndarray, timestamp_ms: int) -> FaceObservation:
        ts = self._mono(self._last_ts, int(timestamp_ms))
        self._last_ts = ts
        result = self._face.detect_for_video(self._image(bgr), ts)
        lms = result.face_landmarks
        if not lms:
            return FaceObservation(present=False, timestamp_ms=ts)
        arr = np.asarray([[p.x, p.y, p.z] for p in lms[0]], dtype=np.float64)
        return FaceObservation(landmarks=arr, present=True, timestamp_ms=ts,
                               n_landmarks=int(arr.shape[0]), score=1.0)

    def process_hands(self, bgr: np.ndarray, timestamp_ms: int) -> HandObservation:
        ts = self._mono(self._hand_last_ts, int(timestamp_ms))
        self._hand_last_ts = ts
        result = self._hands.detect_for_video(self._image(bgr), ts)
        lms = result.hand_landmarks
        if not lms:
            return HandObservation(present=False, timestamp_ms=ts)
        stacked = np.asarray(
            [[[p.x, p.y, p.z] for p in hand] for hand in lms], dtype=np.float64
        )
        labels: List[str] = []
        scores: List[float] = []
        for group in (result.handedness or []):
            if group:
                labels.append(str(group[0].category_name or group[0].display_name or ""))
                scores.append(float(group[0].score))
            else:
                labels.append("")
                scores.append(0.0)
        while len(labels) < stacked.shape[0]:
            labels.append("")
            scores.append(0.0)
        return HandObservation(landmarks=stacked, handedness=labels, scores=scores,
                               present=True, timestamp_ms=ts)

    def close(self) -> None:
        for obj in (getattr(self, "_face", None), getattr(self, "_hands", None)):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


class SolutionsBackend(_BackendBase):
    """Legacy ``mp.solutions`` API (mediapipe <= 0.10.21)."""

    api = "solutions"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        import mediapipe as mp
        self._mp = mp
        min_det = float(get(cfg, "mediapipe.min_detection_confidence", 0.5))
        min_trk = float(get(cfg, "mediapipe.min_tracking_confidence", 0.5))
        self._face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,        # <- this is what adds the 10 iris points
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk,
        )
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk,
        )
        self._ts = 0

    def process_face(self, bgr: np.ndarray, timestamp_ms: int) -> FaceObservation:
        self._ts = max(int(timestamp_ms), self._ts + 1)
        res = self._face.process(bgr_to_rgb(bgr))
        faces = getattr(res, "multi_face_landmarks", None)
        if not faces:
            return FaceObservation(present=False, timestamp_ms=self._ts)
        arr = np.asarray([[p.x, p.y, p.z] for p in faces[0].landmark], dtype=np.float64)
        return FaceObservation(landmarks=arr, present=True, timestamp_ms=self._ts,
                               n_landmarks=int(arr.shape[0]), score=1.0)

    def process_hands(self, bgr: np.ndarray, timestamp_ms: int) -> HandObservation:
        self._ts = max(int(timestamp_ms), self._ts + 1)
        res = self._hands.process(bgr_to_rgb(bgr))
        hands = getattr(res, "multi_hand_landmarks", None)
        if not hands:
            return HandObservation(present=False, timestamp_ms=self._ts)
        stacked = np.asarray(
            [[[p.x, p.y, p.z] for p in h.landmark] for h in hands], dtype=np.float64
        )
        labels, scores = [], []
        for hd in (getattr(res, "multi_handedness", None) or []):
            if hd.classification:
                labels.append(str(hd.classification[0].label))
                scores.append(float(hd.classification[0].score))
        while len(labels) < stacked.shape[0]:
            labels.append("")
            scores.append(0.0)
        return HandObservation(landmarks=stacked, handedness=labels, scores=scores,
                               present=True, timestamp_ms=self._ts)

    def close(self) -> None:
        for name in ("_face", "_hands"):
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    obj.close()
            except Exception:
                pass


# ==========================================================================
# Factory
# ==========================================================================
def create_tracker(cfg: Dict[str, Any], root: Optional[str] = None
                   ) -> Tuple[_BackendBase, List[str]]:
    """Build the best available backend. Raises :class:`TrackerSetupError`.

    The error messages are deliberately actionable -- a missing model bundle
    is the single most common first-run failure, and "mediapipe didn't work"
    is not something a user can act on.
    """
    notes: List[str] = []
    info = probe()
    if not info["installed"]:
        raise TrackerSetupError(
            "MediaPipe is not installed or failed to import:\n  "
            + "\n  ".join(info["errors"])
            + "\nFix: pip install -U mediapipe   (Windows: also see README)"
        )
    notes.append(f"mediapipe {info['version']} ({info['api']} API)")
    if info["api"] is None:
        raise TrackerSetupError(
            "MediaPipe is installed but exposes no usable landmark API.\n  "
            + "\n  ".join(info["errors"])
        )

    if info["api"] == "tasks":
        models = find_models(root)
        missing = [n for n, p in models.items() if p is None]
        if missing:
            d = model_dir(root)
            raise TrackerSetupError(
                "The MediaPipe Tasks API needs model bundles that are not "
                "downloaded yet:\n"
                + "".join(f"  missing: {m}\n" for m in missing)
                + f"\nExpected in: {d}\n"
                "Fix: python tools/download_models.py\n"
                "(or download them by hand -- URLs are in mp_backend.MODEL_URLS)"
            )
        notes.append(f"face model: {models[FACE_MODEL_NAME]}")
        notes.append(f"hand model: {models[HAND_MODEL_NAME]}")
        try:
            return TasksBackend(models[FACE_MODEL_NAME], models[HAND_MODEL_NAME], cfg), notes
        except Exception as exc:
            raise TrackerSetupError(
                f"Failed to initialise MediaPipe Tasks: {type(exc).__name__}: {exc}\n"
                "Common causes: a truncated model file (re-run "
                "tools/download_models.py), or a mediapipe/numpy mismatch."
            ) from exc

    notes.append("using legacy solutions API (no model download needed)")
    try:
        return SolutionsBackend(cfg), notes
    except Exception as exc:
        raise TrackerSetupError(
            f"Failed to initialise legacy MediaPipe solutions: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class NullTracker(_BackendBase):
    """Returns a supplied observation sequence; used by ``tools/simulate.py``."""

    api = "null"

    def __init__(self, faces: Optional[List[FaceObservation]] = None,
                 hands: Optional[List[HandObservation]] = None) -> None:
        self.faces = faces or []
        self.hands = hands or []
        self._i = 0
        self.face_calls = 0
        self.hand_calls = 0

    def process_face(self, bgr: np.ndarray, timestamp_ms: int) -> FaceObservation:
        self.face_calls += 1
        if not self.faces:
            return FaceObservation(present=False, timestamp_ms=timestamp_ms)
        obs = self.faces[self._i % len(self.faces)]
        self._i += 1
        obs.timestamp_ms = timestamp_ms
        return obs

    def process_hands(self, bgr: np.ndarray, timestamp_ms: int) -> HandObservation:
        self.hand_calls += 1
        if not self.hands:
            return HandObservation(present=False, timestamp_ms=timestamp_ms)
        return self.hands[(self._i - 1) % len(self.hands)]
