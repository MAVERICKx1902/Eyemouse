"""On-screen visual feedback: preview HUD, tuning sliders, desktop gaze dot.

Three separate surfaces, each independently optional:

* **Preview HUD** (:class:`Overlay`) -- the camera view with the gaze point,
  iris markers, temple zones, dwell ring and a status strip (FPS, mode,
  click flashes). This is the primary feedback and satisfies the "show me
  where I'm looking and when I clicked" requirement.
* **Sliders** -- OpenCV trackbars that write straight into the config dict,
  so sensitivity/smoothing/deadzone/dwell/temple-radius can be tuned live and
  saved with a keypress.
* **Desktop dot** (:class:`DesktopDot`) -- an optional always-on-top,
  click-through marker drawn on the real desktop via tkinter + a Win32
  layered window. Off by default because it needs a display and a second
  thread, but it is the most useful feedback once the tracker is working.

Nothing here imports MediaPipe, and every drawing routine degrades to a
no-op if OpenCV is missing, so the app can still run headless for tests.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import landmarks as LMK
from config import get, set_path
from util import Clock, clamp

WINDOW_NAME = "Eyemouse"

# BGR colours
C_BG = (32, 30, 28)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 150)
C_GAZE = (60, 210, 255)
C_GAZE_TRAIL = (40, 140, 200)
C_CLICK_L = (90, 220, 120)
C_CLICK_R = (120, 140, 255)
C_SNIPER = (80, 90, 255)
C_SCROLL = (230, 200, 80)
C_WARN = (90, 90, 250)
C_OK = (120, 220, 140)

__all__ = ["Overlay", "DesktopDot", "SLIDERS", "WINDOW_NAME"]


# ==========================================================================
# Sliders
# ==========================================================================
@dataclass
class SliderSpec:
    label: str            # trackbar name (must be unique, <= 32 chars for cv2)
    path: str             # dotted config path written on change
    lo: int
    hi: int
    scale: float          # config_value = trackbar_value * scale
    fmt: str = "{:.2f}"

    def to_trackbar(self, value: float) -> int:
        raw = float(value) / self.scale if self.scale else float(value)
        return int(round(clamp(raw, self.lo, self.hi)))

    def from_trackbar(self, raw: int) -> float:
        return float(clamp(raw, self.lo, self.hi)) * self.scale


# Trackbar values are integers, so each slider carries a scale factor that
# maps it onto the real config value.
SLIDERS: List[SliderSpec] = [
    SliderSpec("Gain_x100", "gaze.gain", 20, 400, 0.01),
    SliderSpec("Smooth_win", "gaze.moving_average_window", 1, 20, 1.0, "{:.0f}"),
    SliderSpec("OneEuro_x1000", "gaze.one_euro.min_cutoff", 5, 500, 0.001, "{:.3f}"),
    SliderSpec("Deadzone_px", "gaze.deadzone_px", 0, 40, 1.0, "{:.0f}"),
    SliderSpec("Dwell_ms_x50", "clicks.dwell.ms", 4, 100, 50.0, "{:.0f}"),
    SliderSpec("Temple_r_x100", "clicks.temple.radius_ratio", 5, 120, 0.01),
    # trackbar range * scale must land inside the config's own valid range:
    # 0..100 * 50 -> 0..5000 ms, matching RANGES["clicks.debounce_ms"].
    SliderSpec("Debounce_ms", "clicks.debounce_ms", 0, 100, 50.0, "{:.0f}"),
]


# ==========================================================================
# Preview HUD
# ==========================================================================
class Overlay:
    """Renders the composite preview frame."""

    def __init__(self, cfg: Dict[str, Any], clock: Optional[Clock] = None) -> None:
        self.cfg = cfg
        self.clock = clock or Clock()
        self._trail: Deque[Tuple[int, int]] = deque(maxlen=24)
        self._flashes: Deque[Tuple[str, int, Tuple[int, int]]] = deque(maxlen=8)
        self._last_gaze: Optional[Tuple[int, int]] = None
        self.sliders_ready = False
        self._messages: List[str] = []
        self._message_until = 0

    # ------------------------------------------------------------------
    def flash(self, label: str, kind: str = "left") -> None:
        colour = C_CLICK_L if kind == "left" else C_CLICK_R if kind == "right" else C_SCROLL
        self._flashes.append((label, self.clock.ms(), colour))

    def say(self, text: str, ms: int = 2500) -> None:
        self._messages.append(text)
        self._messages = self._messages[-3:]
        self._message_until = self.clock.ms() + ms

    def _prune(self) -> None:
        flash_ms = int(get(self.cfg, "overlay.flash_ms", 320))
        now = self.clock.ms()
        while self._flashes and now - self._flashes[0][1] > flash_ms:
            self._flashes.popleft()
        if now > self._message_until:
            self._messages.clear()

    # ------------------------------------------------------------------
    def setup_sliders(self, window: str = WINDOW_NAME) -> bool:
        """Attach trackbars. Returns False (silently) if that is not possible."""
        if cv2 is None or self.sliders_ready:
            return self.sliders_ready
        if not bool(get(self.cfg, "overlay.sliders", True)):
            return False
        try:
            for spec in SLIDERS:
                current = get(self.cfg, spec.path)
                if current is None:
                    continue
                cv2.createTrackbar(spec.label, window, spec.to_trackbar(float(current)),
                                   spec.hi, lambda _v, s=spec: self._on_slider(s, _v))
            self.sliders_ready = True
        except Exception:
            self.sliders_ready = False
        return self.sliders_ready

    def _on_slider(self, spec: SliderSpec, value: int) -> None:
        new = spec.from_trackbar(int(value))
        if spec.path in ("gaze.moving_average_window", "clicks.dwell.ms", "clicks.debounce_ms"):
            new = int(round(new))
        set_path(self.cfg, spec.path, new)

    def read_sliders(self, window: str = WINDOW_NAME) -> None:
        """Poll trackbar positions into the config (the callback is not
        delivered on every OpenCV backend, so polling is the reliable path)."""
        if cv2 is None or not self.sliders_ready:
            return
        for spec in SLIDERS:
            try:
                raw = cv2.getTrackbarPos(spec.label, window)
            except Exception:
                continue
            new = spec.from_trackbar(int(raw))
            if spec.path in ("gaze.moving_average_window", "clicks.dwell.ms", "clicks.debounce_ms"):
                new = int(round(new))
            cur = get(self.cfg, spec.path)
            if cur is None or abs(float(cur) - float(new)) > 1e-9:
                set_path(self.cfg, spec.path, new)

    # ------------------------------------------------------------------
    def render(self, frame_bgr: Optional[np.ndarray], *, gaze=None, gesture=None,
               fps: float = 0.0, tracking: bool = True, calibrated: bool = True,
               mouse_stats=None, timing: str = "", face=None, hands=None,
               screen: Tuple[int, int] = (1920, 1080),
               hotkeys: Optional[Dict[str, str]] = None,
               notes: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Build the display frame. ``frame_bgr`` is the raw camera image."""
        if cv2 is None:
            return None
        self._prune()
        width = int(get(self.cfg, "overlay.width", 980))
        mirror = bool(get(self.cfg, "camera.mirror_preview", True))

        # --- camera panel ----------------------------------------------
        if frame_bgr is not None and frame_bgr.size:
            panel = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()
        else:
            panel = np.full((270, 480, 3), C_BG, dtype=np.uint8)
        ph = int(width * panel.shape[0] / max(1, panel.shape[1]))
        panel = cv2.resize(panel, (width, max(1, ph)), interpolation=cv2.INTER_AREA)
        canvas = panel

        sw, sh = (int(screen[0]) or 1920), (int(screen[1]) or 1080)
        H, W = canvas.shape[:2]

        # --- face / iris / hand debug drawing --------------------------
        if face is not None and bool(get(self.cfg, "overlay.draw_face", False)):
            self._draw_face(canvas, face, W, H, mirror)
        if face is not None and bool(get(self.cfg, "overlay.draw_iris", True)):
            self._draw_eyes(canvas, face, W, H, mirror)
        if bool(get(self.cfg, "overlay.draw_temple_zones", True)) and face is not None:
            self._draw_temples(canvas, face, W, H, mirror, gesture)
        if hands is not None and bool(get(self.cfg, "overlay.draw_hands", True)):
            self._draw_hands(canvas, hands, W, H, mirror)

        # --- gaze marker in the panel ----------------------------------
        if gaze is not None and getattr(gaze, "valid", False):
            gx = int(clamp(gaze.nx, 0, 1) * (W - 1))
            gy = int(clamp(gaze.ny, 0, 1) * (H - 1))
            if mirror:
                gx = W - 1 - gx
            self._trail.append((gx, gy))
            for i, (tx, ty) in enumerate(list(self._trail)[:-1]):
                a = 0.15 + 0.6 * (i / max(1, len(self._trail)))
                cv2.circle(canvas, (tx, ty), 2, tuple(int(c * a) for c in C_GAZE_TRAIL), -1)
            cv2.circle(canvas, (gx, gy), 9, C_GAZE, 2, cv2.LINE_AA)
            cv2.line(canvas, (gx - 16, gy), (gx + 16, gy), C_GAZE, 1, cv2.LINE_AA)
            cv2.line(canvas, (gx, gy - 16), (gx, gy + 16), C_GAZE, 1, cv2.LINE_AA)
            self._last_gaze = (gx, gy)
        else:
            self._trail.clear()

        # --- scroll band indicators ------------------------------------
        if gesture is not None:
            band = float(get(self.cfg, "scroll.edge.band_ratio", 0.06))
            bh = max(2, int(band * H))
            for y0, direction in ((0, -1), (H - bh, 1)):
                active = (getattr(gesture, "edge_scroll_dir", 0) == direction)
                prog = float(getattr(gesture, "edge_dwell_progress", 0.0))
                col = C_SCROLL if active else (70, 70, 70)
                cv2.rectangle(canvas, (0, y0), (W, y0 + bh), col, -1 if active else 1)
                if active and prog > 0:
                    cv2.rectangle(canvas, (0, y0), (int(W * prog), y0 + bh), (255, 255, 160), -1)

        # --- dwell ring ------------------------------------------------
        if gesture is not None and float(get(self.cfg, "clicks.dwell.show_progress", True)):
            prog = float(getattr(gesture, "dwell_progress", 0.0))
            if prog > 0.01 and self._last_gaze is not None and bool(get(self.cfg, "clicks.dwell.enabled", True)):
                gx, gy = self._last_gaze
                r = int(clamp(float(get(self.cfg, "clicks.dwell.radius_px", 24)) / sh * H, 10, 120))
                cv2.circle(canvas, (gx, gy), r, (90, 90, 90), 1, cv2.LINE_AA)
                if prog < 1.0:
                    cv2.ellipse(canvas, (gx, gy), (r, r), -90, 0, 360.0 * prog, C_CLICK_L, 3, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (gx, gy), r, C_CLICK_L, 2, cv2.LINE_AA)

        # --- click flashes ---------------------------------------------
        for i, (label, _t, colour) in enumerate(self._flashes):
            y = H - 26 - i * 22
            self._text(canvas, label, (12, y), 0.62, colour, 2)

        # --- HUD strip -------------------------------------------------
        if bool(get(self.cfg, "overlay.hud", True)):
            self._draw_hud(canvas, gaze=gaze, gesture=gesture, fps=fps,
                           tracking=tracking, calibrated=calibrated,
                           mouse_stats=mouse_stats, timing=timing,
                           screen=(sw, sh), hotkeys=hotkeys, notes=notes)
        return canvas

    # ------------------------------------------------------------------
    def _draw_hud(self, c: np.ndarray, **kw) -> None:
        H, W = c.shape[:2]
        bar_h = 74
        cv2.rectangle(c, (0, max(0, H - bar_h)), (W, H), (24, 22, 20), -1)

        fps = float(kw.get("fps", 0.0))
        fps_col = C_OK if fps >= 15 else C_SCROLL if fps >= 8 else C_WARN
        self._text(c, f"FPS {fps:4.1f}", (10, H - bar_h + 22), 0.62, fps_col, 2)

        gesture = kw.get("gesture")
        gaze = kw.get("gaze")
        screen = kw.get("screen", (1920, 1080))

        mode_bits: List[Tuple[str, Tuple[int, int, int]]] = []
        if not kw.get("tracking", True):
            mode_bits.append(("PAUSED", C_WARN))
        if not kw.get("calibrated", True):
            mode_bits.append(("NOT CALIBRATED", C_WARN))
        if gesture is not None:
            if getattr(gesture, "sniper", False):
                mode_bits.append(("SNIPER", C_SNIPER))
            if getattr(gesture, "scroll_mode", False):
                mode_bits.append(("SCROLL MODE", C_SCROLL))
            if getattr(gesture, "edge_scroll_active", False):
                d = "UP" if getattr(gesture, "edge_scroll_dir", 0) < 0 else "DOWN"
                mode_bits.append((f"SCROLL {d}", C_SCROLL))
            if getattr(gesture, "temple_left", False):
                mode_bits.append(("TEMPLE-L", C_CLICK_L))
            if getattr(gesture, "temple_right", False):
                mode_bits.append(("TEMPLE-R", C_CLICK_R))
            if getattr(gesture, "wink_left", False):
                mode_bits.append(("WINK-L", C_CLICK_L))
            if getattr(gesture, "wink_right", False):
                mode_bits.append(("WINK-R", C_CLICK_R))
            if getattr(gesture, "blink", False):
                mode_bits.append(("blink", C_DIM))
        x = 10
        y = H - bar_h + 46
        for label, col in mode_bits[:6]:
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(c, (x, y - 15), (x + tw + 12, y + 5), col, -1)
            cv2.putText(c, label, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            x += tw + 18

        if gaze is not None and getattr(gaze, "valid", False):
            self._text(c, f"({gaze.x:5.0f},{gaze.y:5.0f})", (W - 150, H - bar_h + 22), 0.55, C_DIM, 1)
        ms = kw.get("mouse_stats")
        if ms is not None:
            self._text(c, f"L{getattr(ms, 'clicks', {}).get('left', 0)} "
                          f"R{getattr(ms, 'clicks', {}).get('right', 0)} "
                          f"S{getattr(ms, 'scrolls', 0)}",
                       (W - 150, H - bar_h + 44), 0.55, C_DIM, 1)
        timing = kw.get("timing", "")
        if timing:
            self._text(c, timing, (10, H - bar_h + 68), 0.45, C_DIM, 1)
        hk = " ".join(f"{k}={v}" for k, v in (kw.get("hotkeys") or {}).items())
        if hk:
            self._text(c, hk, (W // 3, H - bar_h + 68), 0.45, C_DIM, 1)

        msgs = self._messages
        if msgs:
            for i, m in enumerate(msgs[-2:]):
                self._text(c, m, (10, 24 + i * 20), 0.55, C_GAZE, 2)

    # ------------------------------------------------------------------
    @staticmethod
    def _text(img, s, org, scale, colour, thickness=1):
        if cv2 is None:
            return
        cv2.putText(img, str(s), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    float(scale), tuple(int(v) for v in colour), int(thickness), cv2.LINE_AA)

    # --- landmark drawing ---------------------------------------------
    @staticmethod
    def _to_px(pt, W: int, H: int, mirror: bool) -> Tuple[int, int]:
        x = float(pt[0]) * W
        if mirror:
            x = W - x
        return int(clamp(x, 0, W - 1)), int(clamp(float(pt[1]) * H, 0, H - 1))

    def _draw_eyes(self, c, face, W, H, mirror) -> None:
        for side, spec in (("right", LMK.RIGHT_EYE_EAR), ("left", LMK.LEFT_EYE_EAR)):
            pts = [self._to_px(face[i, :2], W, H, mirror) for i in spec]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                cv2.line(c, a, b, (170, 170, 170), 1, cv2.LINE_AA)
        if face.shape[0] >= LMK.IRIS_LANDMARKS:
            for ci, ring in ((LMK.RIGHT_IRIS_CENTER, LMK.RIGHT_IRIS_RING),
                             (LMK.LEFT_IRIS_CENTER, LMK.LEFT_IRIS_RING)):
                ctr = self._to_px(face[ci, :2], W, H, mirror)
                rpts = [self._to_px(face[i, :2], W, H, mirror) for i in ring]
                rad = max(2, int(np.mean([math.dist(ctr, p) for p in rpts])))
                cv2.circle(c, ctr, rad, C_GAZE, 1, cv2.LINE_AA)
                cv2.circle(c, ctr, max(1, rad // 3), C_GAZE, -1, cv2.LINE_AA)

    def _draw_face(self, c, face, W, H, mirror) -> None:
        step = max(1, face.shape[0] // 220)
        for i in range(0, face.shape[0], step):
            cv2.circle(c, self._to_px(face[i, :2], W, H, mirror), 1, (90, 160, 90), -1)

    def _draw_temples(self, c, face, W, H, mirror, gesture) -> None:
        from gestures import temple_anchors
        anchors = temple_anchors(face)
        if not anchors:
            return
        touched_l = bool(getattr(gesture, "temple_left_raw", False)) if gesture else False
        touched_r = bool(getattr(gesture, "temple_right_raw", False)) if gesture else False
        for side, touched in (("left", touched_l), ("right", touched_r)):
            a = anchors[side]
            centre = self._to_px(a["point"], W, H, mirror)
            r = max(4, int(a["width"] * float(get(self.cfg, "clicks.temple.radius_ratio", 0.34)) * W))
            col = (C_CLICK_L if side == "left" else C_CLICK_R) if touched else (110, 110, 110)
            cv2.circle(c, centre, r, col, -1 if touched else 1, cv2.LINE_AA)
            if not touched:
                cv2.circle(c, centre, 2, col, -1)
            label = "L-click" if side == "left" else "R-click"
            self._text(c, label, (centre[0] - 22, centre[1] - r - 6), 0.45, col, 1)

    def _draw_hands(self, c, hands, W, H, mirror) -> None:
        lms = getattr(hands, "landmarks", None)
        if lms is None:
            return
        labels = list(getattr(hands, "handedness", []) or [])
        for hi in range(np.asarray(lms).shape[0]):
            hand = np.asarray(lms)[hi]
            pts = [self._to_px(hand[i, :2], W, H, mirror) for i in range(hand.shape[0])]
            for a, b in LMK_HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(c, pts[a], pts[b], (200, 170, 120), 1, cv2.LINE_AA)
            for tip in LMK.TOUCH_TIPS:
                cv2.circle(c, pts[tip], 4, (120, 220, 255), -1, cv2.LINE_AA)
            if hi < len(labels) and labels[hi]:
                cv2.putText(c, str(labels[hi]), pts[LMK.WRIST], cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)


# Hand skeleton, spelled out so the overlay does not depend on MediaPipe's
# drawing utils (which only exist in some versions).
LMK_HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ==========================================================================
# Desktop gaze dot
# ==========================================================================
class DesktopDot:
    """Click-through always-on-top gaze marker on the real desktop.

    Implemented with tkinter (bundled with CPython on Windows) plus a Win32
    layered window, so there is no extra dependency. The window is made
    transparent to mouse input with ``WS_EX_TRANSPARENT`` -- essential, since
    a dot that swallowed clicks would break the thing it is visualising.

    Everything runs on its own thread and every failure is swallowed: a
    machine without a usable display simply does not get the dot, and the
    tracker keeps working.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(get(cfg, "overlay.desktop_dot", False))
        self.notes: List[str] = []
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._target: Optional[Tuple[int, int, str]] = None   # x, y, state
        self._stop = threading.Event()
        self._alive = False

    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not self.enabled or self._thread is not None:
            return self._alive
        self._thread = threading.Thread(target=self._run, name="DesktopDot", daemon=True)
        self._thread.start()
        # Give the Tk thread a moment; if it died we report why.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not self._alive and not self._stop.is_set():
            time.sleep(0.02)
        return self._alive

    def update(self, x: int, y: int, state: str = "idle") -> None:
        with self._lock:
            self._target = (int(x), int(y), state)

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._alive = False

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.notes.append(f"desktop dot unavailable (tkinter): {type(exc).__name__}: {exc}")
            self._stop.set()
            return
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            size = int(get(self.cfg, "overlay.desktop_dot_size", 26))
            pad = size
            root.configure(bg="black")

            if sys.platform.startswith("win"):
                # Colour-key transparency + click-through.
                try:
                    root.wm_attributes("-transparentcolor", "black")
                except Exception:
                    self.notes.append("desktop dot: -transparentcolor unsupported")
                self._make_click_through(root)
            else:
                try:
                    root.wm_attributes("-alpha", 0.85)
                except Exception:
                    pass

            canvas = tk.Canvas(root, width=size * 2, height=size * 2, bg="black",
                               highlightthickness=0)
            canvas.pack()
            oval = canvas.create_oval(pad // 2, pad // 2, pad // 2 + size, pad // 2 + size,
                                      outline="#3CD2FF", width=3)
            cross = canvas.create_line(0, 0, 0, 0, fill="#3CD2FF", width=1)
            root.geometry(f"{size * 2}x{size * 2}+0+0")

            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            self._alive = True

            def tick() -> None:
                if self._stop.is_set():
                    try:
                        root.destroy()
                    except Exception:
                        pass
                    return
                with self._lock:
                    tgt = self._target
                if tgt is not None:
                    x, y, state = tgt
                    colour = {"sniper": "#FF5A50", "click": "#5ADC78",
                              "scroll": "#E6C850"}.get(state, "#3CD2FF")
                    gx = int(clamp(x, 0, max(0, sw - 1)))
                    gy = int(clamp(y, 0, max(0, sh - 1)))
                    root.geometry(f"+{max(0, gx - size)}+{max(0, gy - size)}")
                    canvas.itemconfigure(oval, outline=colour)
                    canvas.coords(cross, size, size - 6, size, size + 6)
                    canvas.itemconfigure(cross, fill=colour)
                root.after(16, tick)   # ~60 Hz, independent of the tracker loop

            root.after(16, tick)
            root.mainloop()
        except Exception as exc:
            self.notes.append(f"desktop dot failed: {type(exc).__name__}: {exc}")
            self._stop.set()
        finally:
            self._alive = False

    @staticmethod
    def _make_click_through(root) -> None:
        """Set WS_EX_LAYERED | WS_EX_TRANSPARENT on the Tk toplevel."""
        try:
            import ctypes
            from ctypes import wintypes
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(root.wm_frame(), 16) if hasattr(root, "wm_frame") else 0
            if not hwnd:
                return
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE, style)
        except Exception:
            pass
