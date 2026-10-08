"""Cursor and button control, with a pluggable backend.

Three backends share one interface:

* :class:`PynputMouse` -- ``pynput`` ``Controller``. Preferred when
  available: ``SetCursorPos`` per frame with no pyautogui argument
  validation in the way, which matters at 30-60 Hz.
* :class:`PyAutoGUIMouse` -- ``pyautogui``. Always present per the project
  requirements, and used as the fallback.
* :class:`NullMouse` -- records every call and moves nothing. This is what
  the test suite and ``--dry-run`` use.

Two Windows details that bite eye trackers specifically
--------------------------------------------------------
**pyautogui's FAILSAFE must be turned off.** By default, moving the cursor
into the top-left corner raises ``FailSafeException``. An eye tracker parks
the cursor in screen corners *by design* (that is where the scroll bands
are), so leaving failsafe on guarantees a crash within the first minute.
``PAUSE`` is also set to 0 -- pyautogui otherwise sleeps 0.1 s after every
call, capping the app at ~10 Hz.

**DPI scaling.** At 125%/150% display scaling, a DPI-unaware process sees a
virtualized screen size while ``SetCursorPos`` works in physical pixels, so
the cursor lands in the wrong place and cannot reach the right/bottom edges.
:func:`ensure_dpi_aware` opts the process into per-monitor awareness before
the screen size is ever read.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import get
from util import clamp

__all__ = [
    "MouseController", "NullMouse", "PyAutoGUIMouse", "PynputMouse",
    "ensure_dpi_aware", "screen_size", "create_mouse",
]

_IS_WINDOWS = sys.platform.startswith("win")


# ==========================================================================
# Windows plumbing
# ==========================================================================
def ensure_dpi_aware() -> str:
    """Make the process DPI aware so screen and cursor coordinates agree.

    Returns a short string describing what happened (used in the startup log).
    """
    if not _IS_WINDOWS:
        return "n/a (not Windows)"
    try:
        import ctypes
    except Exception:  # pragma: no cover
        return "ctypes unavailable"
    # Per-monitor v2 (2) is the modern answer; SetProcessDpiAwareness (1) and
    # the legacy SetProcessDPIAware are fallbacks for older builds.
    for label, fn in (
        ("per-monitor-v2", lambda: ctypes.windll.shcore.SetProcessDpiAwarenessContext(-4)),
        ("per-monitor", lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2)),
        ("system", lambda: ctypes.windll.user32.SetProcessDPIAware()),
    ):
        try:
            fn()
            return label
        except Exception:
            continue
    # Usually means awareness was already set by the host process.
    return "already-set/unavailable"


def screen_size() -> Tuple[int, int]:
    """Primary monitor size in pixels (physical, once DPI aware)."""
    if _IS_WINDOWS:
        try:
            import ctypes
            u = ctypes.windll.user32
            w, h = int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    for reader in (_size_via_pyautogui,):
        try:
            wh = reader()
            if wh and wh[0] > 0 and wh[1] > 0:
                return int(wh[0]), int(wh[1])
        except Exception:
            continue
    return 1920, 1080


def _size_via_pyautogui() -> Tuple[int, int]:
    import pyautogui
    pyautogui.FAILSAFE = False
    w, h = pyautogui.size()
    return int(w), int(h)


# ==========================================================================
# Backends
# ==========================================================================
class _BaseMouse:
    name = "base"

    def move(self, x: float, y: float) -> None:
        raise NotImplementedError

    def position(self) -> Tuple[int, int]:
        raise NotImplementedError

    def click(self, button: str = "left") -> None:
        raise NotImplementedError

    def scroll(self, amount: int) -> None:
        raise NotImplementedError

    def size(self) -> Tuple[int, int]:
        return screen_size()

    def close(self) -> None:
        pass


class NullMouse(_BaseMouse):
    """In-memory backend: records calls, moves nothing.

    Used by tests, ``--dry-run``, and as the last-resort fallback when no
    mouse library imports (e.g. a headless CI box), so the rest of the app can
    still run and be evaluated.
    """

    name = "null"

    def __init__(self, w: int = 1920, h: int = 1080) -> None:
        self._w, self._h = int(w), int(h)
        self._pos = (self._w // 2, self._h // 2)
        self.moves: List[Tuple[float, float]] = []
        self.clicks: List[str] = []
        self.scrolls: List[int] = []

    def move(self, x: float, y: float) -> None:
        self._pos = (int(round(x)), int(round(y)))
        self.moves.append((float(x), float(y)))

    def position(self) -> Tuple[int, int]:
        return self._pos

    def click(self, button: str = "left") -> None:
        self.clicks.append(button)

    def scroll(self, amount: int) -> None:
        if amount:
            self.scrolls.append(int(amount))

    def size(self) -> Tuple[int, int]:
        return self._w, self._h


class PyAutoGUIMouse(_BaseMouse):
    name = "pyautogui"

    def __init__(self) -> None:
        import pyautogui
        # Both of these are mandatory -- see the module docstring.
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        self._pg = pyautogui
        self._pos: Tuple[int, int] = tuple(int(v) for v in pyautogui.position())

    def move(self, x: float, y: float) -> None:
        ix, iy = int(round(x)), int(round(y))
        if (ix, iy) == self._pos:
            return
        self._pg.moveTo(ix, iy, duration=0)
        self._pos = (ix, iy)

    def position(self) -> Tuple[int, int]:
        try:
            self._pos = tuple(int(v) for v in self._pg.position())
        except Exception:
            pass
        return self._pos

    def click(self, button: str = "left") -> None:
        self._pg.click(button=button if button in ("left", "right", "middle") else "left")

    def scroll(self, amount: int) -> None:
        if amount:
            self._pg.scroll(int(amount))

    def size(self) -> Tuple[int, int]:
        w, h = self._pg.size()
        return int(w), int(h)


class PynputMouse(_BaseMouse):
    name = "pynput"

    def __init__(self) -> None:
        from pynput import mouse
        self._mouse = mouse
        self._ctrl = mouse.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right,
                         "middle": mouse.Button.middle}
        self._pos: Tuple[int, int] = tuple(int(v) for v in self._ctrl.position)

    def move(self, x: float, y: float) -> None:
        ix, iy = int(round(x)), int(round(y))
        if (ix, iy) == self._pos:
            return
        self._ctrl.position = (ix, iy)
        self._pos = (ix, iy)

    def position(self) -> Tuple[int, int]:
        try:
            self._pos = tuple(int(v) for v in self._ctrl.position)
        except Exception:
            pass
        return self._pos

    def click(self, button: str = "left") -> None:
        btn = self._buttons.get(button, self._buttons["left"])
        self._ctrl.click(btn, 1)

    def scroll(self, amount: int) -> None:
        if amount:
            self._ctrl.scroll(0, int(amount))

    def size(self) -> Tuple[int, int]:
        return screen_size()

    def close(self) -> None:
        pass


def create_mouse(backend: str = "auto", null_size: Tuple[int, int] = (1920, 1080)
                 ) -> Tuple[_BaseMouse, List[str]]:
    """Instantiate a backend. Never raises: degrades to :class:`NullMouse`."""
    notes: List[str] = []
    want = str(backend or "auto").lower()
    order: List[str] = []
    if want == "auto":
        order = ["pynput", "pyautogui"]
    elif want == "none":
        order = []
    else:
        order = [want]

    known = ("pynput", "pyautogui")
    for name in order:
        if name not in known:
            notes.append(f"unknown mouse backend {name!r} (expected one of {known})")
            continue
        try:
            if name == "pynput":
                return PynputMouse(), notes
            if name == "pyautogui":
                return PyAutoGUIMouse(), notes
        except Exception as exc:
            notes.append(f"mouse backend {name!r} unavailable: {type(exc).__name__}: {exc}")
    if want != "none":
        notes.append("falling back to NullMouse: the cursor will NOT move")
    return NullMouse(*null_size), notes


# ==========================================================================
# Controller
# ==========================================================================
@dataclass
class MouseStats:
    moves: int = 0
    clicks: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    scrolls: int = 0
    scroll_units: int = 0
    last_click: str = ""
    last_click_t: float = 0.0
    sniper_frames: int = 0


class MouseController:
    """Applies gaze output to the OS cursor, with sniper scaling.

    Sniper mode
    ~~~~~~~~~~~
    Rather than re-mapping gaze into a smaller screen region (which would
    teleport the cursor when the mode toggles), sniper mode scales the *delta*
    between the cursor's current position and the gaze target::

        new = current + (target - current) * sniper_factor

    The cursor therefore creeps toward where you are looking at 1/4 speed,
    which is exactly the fine-targeting behaviour wanted, and on exit the
    normal absolute mapping resumes with no discontinuity to unwind.
    """

    def __init__(self, cfg: Dict[str, Any], backend: Optional[_BaseMouse] = None,
                 screen: Optional[Tuple[int, int]] = None) -> None:
        self.cfg = cfg
        self.notes: List[str] = []
        self.backend: _BaseMouse
        if backend is not None:
            self.backend = backend
        else:
            if bool(get(cfg, "mouse.dpi_aware", True)):
                self.notes.append(f"dpi awareness: {ensure_dpi_aware()}")
            self.backend, notes = create_mouse(
                str(get(cfg, "mouse.backend", "auto")),
                null_size=screen or (1920, 1080),
            )
            self.notes.extend(notes)
        self.screen_w, self.screen_h = screen or self.backend.size()
        self.stats = MouseStats()
        self._target: Optional[Tuple[float, float]] = None
        self._pos: Tuple[float, float] = (self.screen_w / 2.0, self.screen_h / 2.0)
        # Last integer position actually handed to the OS, so repeated
        # identical targets can skip the (non-trivial) syscall.
        self._sent: Optional[Tuple[int, int]] = None
        try:
            self._pos = (float(self.backend.position()[0]), float(self.backend.position()[1]))
        except Exception:
            pass
        self._inset = float(get(cfg, "mouse.screen_inset_px", 6))

    # ------------------------------------------------------------------
    def apply_config(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._inset = float(get(cfg, "mouse.screen_inset_px", 6))

    def set_screen_size(self, w: int, h: int) -> None:
        self.screen_w, self.screen_h = int(w), int(h)

    @property
    def position(self) -> Tuple[float, float]:
        return self._pos

    def resync(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Adopt the real cursor position (called after calibration)."""
        try:
            px, py = self.backend.position()
            self._pos = (float(x if x is not None else px), float(y if y is not None else py))
        except Exception:
            if x is not None and y is not None:
                self._pos = (float(x), float(y))
        self._sent = None  # force the next move through to the OS

    # ------------------------------------------------------------------
    def move_to(self, x: float, y: float, sniper: bool = False, force: bool = True) -> bool:
        """Move the cursor. Returns True when an OS call was actually made.

        ``self._pos`` keeps the *unrounded* logical position. That matters in
        sniper mode: the step is a quarter of the residual, so rounding the
        stored position would swallow sub-pixel steps and freeze the cursor
        well short of the target. Only the value handed to the OS is rounded.
        """
        # Read live from config rather than a cached copy: the sliders change
        # these values at runtime and a stale inset silently mis-clamps.
        if bool(get(self.cfg, "mouse.clamp_to_screen", True)):
            inset = float(get(self.cfg, "mouse.screen_inset_px", 6))
            tx = clamp(float(x), inset, max(inset, self.screen_w - inset))
            ty = clamp(float(y), inset, max(inset, self.screen_h - inset))
        else:
            # Off entirely -- for multi-monitor setups where the user wants the
            # cursor to be able to leave the primary screen.
            tx, ty = float(x), float(y)
        self._target = (tx, ty)

        if sniper:
            factor = clamp(float(get(self.cfg, "precision.sniper_factor", 0.25)), 0.01, 1.0)
            tx = self._pos[0] + (tx - self._pos[0]) * factor
            ty = self._pos[1] + (ty - self._pos[1]) * factor
            self.stats.sniper_frames += 1

        self._pos = (float(tx), float(ty))
        ix, iy = int(round(tx)), int(round(ty))
        if self._sent == (ix, iy) and not force:
            return False
        try:
            self.backend.move(ix, iy)
        except Exception as exc:
            self.notes.append(f"move failed: {exc}")
            return False
        self._sent = (ix, iy)
        self.stats.moves += 1
        return True

    # ------------------------------------------------------------------
    def click(self, button: str = "left") -> bool:
        button = button if button in ("left", "right", "middle") else "left"
        try:
            self.backend.click(button)
        except Exception as exc:
            self.notes.append(f"click failed: {exc}")
            return False
        self.stats.clicks[button] = self.stats.clicks.get(button, 0) + 1
        self.stats.last_click = button
        self.stats.last_click_t = time.monotonic()
        return True

    def scroll(self, amount: int) -> bool:
        amount = int(amount)
        if amount == 0:
            return False
        try:
            self.backend.scroll(amount)
        except Exception as exc:
            self.notes.append(f"scroll failed: {exc}")
            return False
        self.stats.scrolls += 1
        self.stats.scroll_units += amount
        return True

    def close(self) -> None:
        try:
            self.backend.close()
        except Exception:
            pass
