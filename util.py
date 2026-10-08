"""Small shared helpers: FPS meter, clamping, monotonic clock.

Kept separate from the domain modules so ``gaze``, ``gestures`` and
``main`` can all use them without importing each other.
"""

from __future__ import annotations

import time
from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]`` (NaN-safe)."""
    if value != value:  # NaN
        return lo
    if hi < lo:
        lo, hi = hi, lo
    return lo if value < lo else (hi if value > hi else value)


def now_ms() -> int:
    """Monotonic wall clock in integer milliseconds."""
    return int(time.monotonic() * 1000.0)


class Clock:
    """Injectable clock so timing logic is unit-testable.

    ``Clock()`` follows the real monotonic clock; ``Clock(manual=True)`` only
    advances when :meth:`advance` is called, which lets tests assert on
    debounce/dwell/scroll timing deterministically.
    """

    def __init__(self, manual: bool = False, start_ms: int = 0) -> None:
        self._manual = manual
        self._t = float(start_ms) / 1000.0

    def ms(self) -> int:
        if self._manual:
            return int(round(self._t * 1000.0))
        return int(time.monotonic() * 1000.0)

    def advance(self, ms: float) -> None:
        if not self._manual:
            raise RuntimeError("Clock.advance() only works with manual=True")
        self._t += ms / 1000.0


class FPSMeter:
    """Exponentially-smoothed frames-per-second estimate."""

    def __init__(self, smoothing: float = 0.1, clock: Optional[Clock] = None) -> None:
        self._alpha = clamp(float(smoothing), 0.01, 1.0)
        self._clock = clock or Clock()
        self._last: Optional[int] = None
        self._fps = 0.0
        self._frames = 0
        self._t0 = self._clock.ms()

    def tick(self) -> float:
        t = self._clock.ms()
        if self._last is not None:
            dt = t - self._last
            if dt > 0:
                inst = 1000.0 / dt
                self._fps = inst if self._fps <= 0 else self._fps + self._alpha * (inst - self._fps)
        self._last = t
        self._frames += 1
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def elapsed_sec(self) -> float:
        return max(1e-6, (self._clock.ms() - self._t0) / 1000.0)

    @property
    def average_fps(self) -> float:
        return self._frames / self.elapsed_sec


class StageTimer:
    """Per-stage millisecond timer for the ``performance.show_timing`` HUD."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.stages: dict[str, float] = {}
        self._marks: dict[str, int] = {}

    def start(self, name: str) -> None:
        if self.enabled:
            self._marks[name] = now_ms()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        t0 = self._marks.pop(name, None)
        if t0 is None:
            return
        dt = now_ms() - t0
        prev = self.stages.get(name)
        self.stages[name] = dt if prev is None else prev + 0.15 * (dt - prev)

    def summary(self) -> str:
        if not self.enabled or not self.stages:
            return ""
        return "  ".join(f"{k}:{v:.0f}ms" for k, v in self.stages.items())
