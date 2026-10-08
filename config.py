"""Configuration handling: defaults, JSON load/save, validation and clamping.

The whole app is driven by a single JSON file (``config.json`` next to
``main.py`` by default). Every key in :data:`DEFAULTS` is guaranteed to exist
after :func:`load_config`, so the rest of the code never has to defend
against a missing entry, and a user can hand-edit a partial file without
breaking anything (missing keys fall back to defaults, unknown keys are
reported but tolerated).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Tuple

CONFIG_FILENAME = "config.json"
CALIBRATION_FILENAME = "calibration.json"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        # Mirror the preview so it behaves like a bathroom mirror. Landmarks
        # are still computed on the *unmirrored* frame, which is what
        # MediaPipe's handedness labels assume for a selfie camera.
        "mirror_preview": True,
        # DirectShow avoids the slow MSMF backend on many Windows webcams.
        "backend_windows": "dshow",
    },
    "performance": {
        # Downscale factor applied before inference. 1.0 = capture size.
        "process_scale": 1.0,
        # Hands only need to be located a few times a second; running the
        # hand model every Nth frame is the single biggest CPU win.
        "hands_every_n_frames": 3,
        "show_timing": False,
        "max_fps": 0,          # 0 = uncapped
        "fps_smoothing": 0.1,  # EMA factor for the reported FPS
    },
    "mediapipe": {
        # Raise these when the face is dropped in uneven lighting; lower
        # them to stop a false detection from snapping the cursor. Both
        # apply to the face landmarker and the hand landmarker.
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        # "gpu" is attempted and falls back to CPU if the delegate cannot be
        # created -- the stock Windows wheels are CPU-only.
        "delegate": "cpu",
    },
    "gaze": {
        # Multiplies the calibrated gaze offset around the screen centre.
        # >1 = smaller eye movement needed to cross the screen.
        "gain": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        # one of: none | moving_average | one_euro | kalman | ma_one_euro
        "filter": "ma_one_euro",
        "moving_average_window": 4,
        "one_euro": {"min_cutoff": 1.1, "beta": 0.012, "d_cutoff": 1.0},
        "kalman": {"process_noise": 1.0e-3, "measurement_noise": 2.5e-2},
        # Faces narrower than this fraction of the frame width are rejected:
        # the iris signal is pure noise at that scale.
        "min_face_width": 0.10,
        # Screen pixels of movement that are swallowed to stop micro-jitter.
        "deadzone_px": 2.5,
        # 0 = hard deadzone (motion inside the radius is frozen).
        # >0 = attenuate it instead, for users who find a hard zone "sticky".
        "deadzone_soft": 0.0,
    },
    "calibration": {
        "points": 9,
        "samples_per_point": 30,
        "discard_first": 8,
        # one of: linear | quadratic | quadratic_head
        "model": "quadratic",
        "ridge": 1.0e-4,
        "file": CALIBRATION_FILENAME,
        "auto_load": True,
        # Keep points this far in from the screen edge so the calibration
        # does not fight the scroll bands.
        "inset_ratio": 0.12,
        "point_radius_px": 14,
        "countdown_sec": 0.6,
    },
    "clicks": {
        # one of: temple | wink | both | none
        "gesture": "both",
        "debounce_ms": 500,
        "temple": {
            # Touch radius as a fraction of face width (234<->454).
            "radius_ratio": 0.34,
            # Consecutive frames the touch must be held: kills single-frame
            # false positives without adding perceptible latency.
            "hold_frames": 2,
            # temple_side = whichever temple is touched decides the button
            # (robust). strict = the hand's MediaPipe handedness label must
            # also agree. off = ignore handedness entirely.
            "handedness_mode": "temple_side",
            "swap_handedness": False,
            "both_temples_sniper": True,
        },
        "wink": {
            "ear_threshold": 0.20,
            # The winking eye must be this much more closed than the other.
            "asymmetry_ratio": 0.60,
            "open_eye_min": 0.22,
            "hold_frames": 2,
            # Reject a wink that is only *confirmed* this long after the eye
            # first closed (e.g. the face was lost and regained mid-wink), so
            # a click cannot fire from stale state.
            "max_duration_ms": 900,
        },
        "dwell": {
            "enabled": True,
            "ms": 1400,
            "radius_px": 24,
            # 0 disables the long-dwell right click.
            "right_click_ms": 0,
            "show_progress": True,
            # Dwell is suppressed while a temple/wink click is armed, while
            # edge-scrolling, and in this band around the screen edge.
            "edge_guard_ratio": 0.08,
        },
    },
    "scroll": {
        # one of: edge | palm | none
        "mode": "edge",
        "edge": {
            "band_ratio": 0.06,
            "dwell_ms": 800,
            "repeat_ms": 70,
            "min_amount": 1,
            "max_amount": 8,
            "accelerate": True,
        },
        "palm": {
            "toggle_ms": 400,
            "open_fingers": 4,
            "exit_after_ms": 6000,
            "px_per_click": 55.0,
            "max_clicks_per_event": 8,
        },
    },
    "precision": {
        # Cursor delta multiplier while sniper mode is held.
        "sniper_factor": 0.25,
        "sniper_key": "ctrl",
        "smoothing_boost": 2.2,
    },
    "mouse": {
        # one of: auto | pyautogui | pynput | none
        "backend": "auto",
        "screen_inset_px": 6,
        "dpi_aware": True,
        "clamp_to_screen": True,
    },
    "hotkeys": {
        "recalibrate": "f8",
        "toggle_tracking": "f7",
        "toggle_sniper": "f6",
        "save_config": "f9",
        "quit": "esc",
        "quit_alt": "q",
    },
    "overlay": {
        "show": True,
        "width": 980,
        "draw_iris": True,
        "draw_face": False,
        "draw_hands": True,
        "draw_temple_zones": True,
        "sliders": True,
        "flash_ms": 320,
        "hud": True,
        # Optional click-through gaze dot drawn on the real desktop (tkinter
        # + Win32 layered window). Off by default: it needs a display and an
        # extra thread, and the preview HUD already shows the gaze point.
        "desktop_dot": False,
        "desktop_dot_size": 26,
    },
}

# --------------------------------------------------------------------------
# Validation tables
# --------------------------------------------------------------------------
CHOICES: Dict[str, Tuple[str, ...]] = {
    "gaze.filter": ("none", "moving_average", "one_euro", "kalman", "ma_one_euro"),
    "calibration.model": ("linear", "quadratic", "quadratic_head"),
    "calibration.points": (5, 9, 13),
    "clicks.gesture": ("temple", "wink", "both", "none"),
    "clicks.temple.handedness_mode": ("temple_side", "strict", "off"),
    "scroll.mode": ("edge", "palm", "none"),
    "mouse.backend": ("auto", "pyautogui", "pynput", "none"),
    "mediapipe.delegate": ("cpu", "gpu"),
}

# path -> (lo, hi). Values are clamped, never rejected, so a typo in the JSON
# degrades gracefully instead of refusing to start.
RANGES: Dict[str, Tuple[float, float]] = {
    "camera.index": (0, 10),
    "camera.width": (160, 4096),
    "camera.height": (120, 2160),
    "camera.fps": (5, 240),
    "performance.process_scale": (0.25, 1.0),
    "performance.hands_every_n_frames": (1, 15),
    "performance.max_fps": (0, 240),
    "performance.fps_smoothing": (0.01, 1.0),
    "mediapipe.min_detection_confidence": (0.05, 1.0),
    "mediapipe.min_tracking_confidence": (0.05, 1.0),
    "gaze.gain": (0.2, 5.0),
    "gaze.offset_x": (-2000, 2000),
    "gaze.offset_y": (-2000, 2000),
    "gaze.moving_average_window": (1, 30),
    "gaze.one_euro.min_cutoff": (0.005, 20.0),
    "gaze.one_euro.beta": (0.0, 5.0),
    "gaze.one_euro.d_cutoff": (0.005, 20.0),
    "gaze.kalman.process_noise": (1e-7, 10.0),
    "gaze.kalman.measurement_noise": (1e-6, 10.0),
    "gaze.min_face_width": (0.03, 0.6),
    "gaze.deadzone_px": (0.0, 60.0),
    "gaze.deadzone_soft": (0.0, 1.0),
    "calibration.samples_per_point": (5, 300),
    "calibration.discard_first": (0, 100),
    "calibration.ridge": (1e-8, 10.0),
    "calibration.inset_ratio": (0.0, 0.3),
    "calibration.point_radius_px": (4, 80),
    "calibration.countdown_sec": (0.0, 5.0),
    "clicks.debounce_ms": (0, 5000),
    "clicks.temple.radius_ratio": (0.05, 1.5),
    "clicks.temple.hold_frames": (1, 15),
    "clicks.wink.ear_threshold": (0.05, 0.6),
    "clicks.wink.asymmetry_ratio": (0.05, 1.0),
    "clicks.wink.open_eye_min": (0.05, 0.8),
    "clicks.wink.hold_frames": (1, 15),
    "clicks.wink.max_duration_ms": (100, 5000),
    "clicks.dwell.ms": (200, 10000),
    "clicks.dwell.radius_px": (2, 200),
    "clicks.dwell.right_click_ms": (0, 30000),
    "clicks.dwell.edge_guard_ratio": (0.0, 0.4),
    "scroll.edge.band_ratio": (0.01, 0.3),
    "scroll.edge.dwell_ms": (100, 5000),
    "scroll.edge.repeat_ms": (10, 1000),
    "scroll.edge.min_amount": (1, 50),
    "scroll.edge.max_amount": (1, 100),
    "scroll.palm.toggle_ms": (50, 5000),
    "scroll.palm.open_fingers": (2, 4),
    "scroll.palm.exit_after_ms": (500, 60000),
    "scroll.palm.px_per_click": (5.0, 1000.0),
    "scroll.palm.max_clicks_per_event": (1, 50),
    "precision.sniper_factor": (0.05, 1.0),
    "precision.smoothing_boost": (1.0, 10.0),
    "mouse.screen_inset_px": (0, 200),
    "overlay.width": (320, 3840),
    "overlay.flash_ms": (0, 3000),
    "overlay.desktop_dot_size": (6, 120),
}

INT_KEYS = {
    "camera.index", "camera.width", "camera.height", "camera.fps",
    "performance.hands_every_n_frames", "performance.max_fps",
    "gaze.moving_average_window", "calibration.points",
    "calibration.samples_per_point", "calibration.discard_first",
    "calibration.point_radius_px", "clicks.debounce_ms",
    "clicks.temple.hold_frames", "clicks.wink.hold_frames",
    "clicks.wink.max_duration_ms", "clicks.dwell.ms", "clicks.dwell.radius_px",
    "clicks.dwell.right_click_ms", "scroll.edge.dwell_ms", "scroll.edge.repeat_ms",
    "scroll.edge.min_amount", "scroll.edge.max_amount", "scroll.palm.toggle_ms",
    "scroll.palm.open_fingers", "scroll.palm.exit_after_ms",
    "overlay.width", "overlay.flash_ms", "overlay.desktop_dot_size",
}

BOOL_KEYS = {
    "camera.mirror_preview", "performance.show_timing", "calibration.auto_load",
    "clicks.temple.both_temples_sniper", "clicks.temple.swap_handedness",
    "clicks.dwell.enabled", "clicks.dwell.show_progress", "scroll.edge.accelerate",
    "mouse.dpi_aware", "mouse.clamp_to_screen", "overlay.show", "overlay.draw_iris",
    "overlay.draw_face", "overlay.draw_hands", "overlay.draw_temple_zones",
    "overlay.sliders", "overlay.hud", "overlay.desktop_dot",
}


class ConfigError(Exception):
    """Raised when the config file cannot be read at all."""


def get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Dotted-path lookup, e.g. ``get(cfg, "gaze.one_euro.beta")``."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """Dotted-path write, creating intermediate dicts as needed."""
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_types(cfg: Dict[str, Any]) -> List[str]:
    """Cast int/bool keys to their declared type. Returns notes."""
    notes: List[str] = []
    for path in sorted(BOOL_KEYS):
        val = get(cfg, path)
        if val is None:
            continue
        if not isinstance(val, bool):
            try:
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    val = bool(val)
            except Exception:
                continue
            set_path(cfg, path, val)
            notes.append(f"{path}: coerced to bool -> {val}")
    for path in sorted(INT_KEYS):
        val = get(cfg, path)
        if val is None or isinstance(val, bool):
            continue
        try:
            ival = int(round(float(val)))
        except (TypeError, ValueError):
            notes.append(f"{path}: not a number, left as-is")
            continue
        if ival != val:
            notes.append(f"{path}: {val!r} -> {ival}")
        set_path(cfg, path, ival)
    return notes


def _clamp(cfg: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for path, (lo, hi) in RANGES.items():
        val = get(cfg, path)
        if val is None or isinstance(val, bool):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            notes.append(f"{path}: {val!r} is not numeric -> default kept")
            set_path(cfg, path, get(DEFAULTS, path))
            continue
        clamped = min(hi, max(lo, fval))
        if path in INT_KEYS:
            clamped = int(round(clamped))
        else:
            clamped = float(clamped)
        if clamped != fval:
            notes.append(f"{path}: {fval} clamped to {clamped} (range {lo}..{hi})")
        set_path(cfg, path, clamped)
    return notes


def _check_choices(cfg: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for path, allowed in CHOICES.items():
        val = get(cfg, path)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (str, int)):
            notes.append(f"{path}: invalid type, reset to default")
            set_path(cfg, path, get(DEFAULTS, path))
            continue
        sval = str(val).strip().lower()
        if sval not in {str(a).lower() for a in allowed}:
            notes.append(
                f"{path}: {val!r} not in {allowed} -> {get(DEFAULTS, path)!r}"
            )
            set_path(cfg, path, get(DEFAULTS, path))
        else:
            # Normalise case/int-ness so downstream comparisons are simple.
            target = get(DEFAULTS, path)
            if isinstance(target, int):
                sval = int(sval)
            set_path(cfg, path, sval)
    return notes


def _find_unknown(cfg: Dict[str, Any], base: Dict[str, Any], prefix: str = "") -> List[str]:
    unknown: List[str] = []
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if not isinstance(base, dict) or key not in base:
            unknown.append(path)
            continue
        if isinstance(value, dict):
            unknown.extend(_find_unknown(value, base[key], prefix=f"{path}."))
    return unknown


def validate(cfg: Dict[str, Any]) -> List[str]:
    """Normalise ``cfg`` in place and return human-readable notes."""
    notes: List[str] = []
    notes.extend(_coerce_types(cfg))
    notes.extend(_check_choices(cfg))
    notes.extend(_clamp(cfg))

    # Cross-field sanity: min must not exceed max, dwell right > dwell left.
    lo = get(cfg, "scroll.edge.min_amount")
    hi = get(cfg, "scroll.edge.max_amount")
    if lo is not None and hi is not None and lo > hi:
        set_path(cfg, "scroll.edge.max_amount", lo)
        notes.append(f"scroll.edge.max_amount ({hi}) < min_amount ({lo}); raised to {lo}")

    dw = get(cfg, "clicks.dwell.ms") or 0
    rdw = get(cfg, "clicks.dwell.right_click_ms") or 0
    if rdw and rdw <= dw:
        set_path(cfg, "clicks.dwell.right_click_ms", int(dw * 2))
        notes.append(
            f"clicks.dwell.right_click_ms ({rdw}) must exceed dwell.ms ({dw}); "
            f"set to {int(dw * 2)}"
        )

    # The scroll band and the calibration inset must not overlap, otherwise
    # the outermost calibration points trigger scrolling mid-calibration.
    band = get(cfg, "scroll.edge.band_ratio") or 0.0
    inset = get(cfg, "calibration.inset_ratio") or 0.0
    if inset < band + 0.02:
        new_inset = round(band + 0.04, 3)
        set_path(cfg, "calibration.inset_ratio", new_inset)
        notes.append(
            f"calibration.inset_ratio ({inset}) overlaps the scroll band ({band}); "
            f"raised to {new_inset}"
        )
    return notes


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and validate config. Missing file -> defaults (and no error)."""
    cfg = copy.deepcopy(DEFAULTS)
    used_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)
    notes: List[str] = []

    if os.path.exists(used_path):
        try:
            with open(used_path, "r", encoding="utf-8") as fh:
                user = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{used_path} is not valid JSON ({exc}). Fix it or delete the file "
                f"to fall back to defaults."
            ) from exc
        except OSError as exc:
            raise ConfigError(f"Could not read {used_path}: {exc}") from exc
        if not isinstance(user, dict):
            raise ConfigError(f"{used_path} must contain a JSON object.")
        unknown = _find_unknown(user, DEFAULTS)
        for key in unknown:
            notes.append(f"unknown config key ignored: {key}")
        cfg = deep_merge(cfg, user)
    else:
        notes.append(f"no config file at {used_path}; using built-in defaults")

    notes.extend(validate(cfg))
    cfg["_meta"] = {"path": used_path, "notes": notes, "existed": os.path.exists(used_path)}
    return cfg


def save_config(cfg: Dict[str, Any], path: str | None = None) -> str:
    """Write the user-visible config (internal ``_meta`` is stripped)."""
    target = path or cfg.get("_meta", {}).get("path") or CONFIG_FILENAME
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, target)
    return target


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)


def write_default_config(path: str | None = None) -> str:
    """Materialise a fully-populated config.json (used by ``--write-config``)."""
    target = path or default_config_path()
    cfg = copy.deepcopy(DEFAULTS)
    validate(cfg)
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    return target
