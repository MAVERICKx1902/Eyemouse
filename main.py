"""Eyemouse -- webcam-only eye-tracker mouse for Windows.

Run it::

    python main.py                 # calibrate if needed, then track
    python main.py --calibrate     # force a fresh 9-point calibration
    python main.py --check         # diagnose camera / mediapipe / models
    python main.py --dry-run       # full pipeline, cursor never moves
    python main.py --write-config  # emit a fully-populated config.json

The class :class:`EyemouseApp` accepts an injected tracker and frame source so
``tools/simulate.py`` (and the test suite) can drive the *entire* pipeline --
gaze, calibration, gestures, mouse, overlay -- with synthetic landmarks and no
webcam, no MediaPipe and no display.

Loop shape
----------
``capture -> face landmarks -> (throttled) hand landmarks -> gaze features ->
calibration map -> smoothing -> gestures -> cursor/buttons -> overlay``

Hands are processed every ``performance.hands_every_n_frames`` frames and the
previous result is carried forward. Hand gestures are coarse and slow, so this
is free in practice and is the single biggest CPU saving available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import calibration as calib
import config as config_mod
import gaze as gaze_mod
import gestures as gest
import mouse_control as mctl
import mp_backend as mpb
from config import get, save_config, set_path, write_default_config
from hotkeys import HotkeyManager
from overlay import Overlay, DesktopDot, WINDOW_NAME
from util import Clock, FPSMeter, StageTimer, clamp

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ==========================================================================
# Camera
# ==========================================================================
class Camera:
    """OpenCV capture with a Windows-friendly backend choice."""

    def __init__(self, cfg: Dict[str, Any], index: Optional[int] = None) -> None:
        self.cfg = cfg
        self.index = int(index if index is not None else get(cfg, "camera.index", 0))
        self.notes: List[str] = []
        self.cap = None
        self.width = 0
        self.height = 0

    def open(self) -> bool:
        if cv2 is None:
            self.notes.append("OpenCV is not installed")
            return False
        idx = self.index
        backend_name = str(get(self.cfg, "camera.backend_windows", "dshow")).lower()
        backends: List[Tuple[str, int]] = []
        if sys.platform.startswith("win"):
            # DirectShow opens most USB webcams far faster and more reliably
            # than the default MSMF backend, which can hang for 10+ seconds.
            if backend_name == "dshow" and hasattr(cv2, "CAP_DSHOW"):
                backends.append(("dshow", cv2.CAP_DSHOW))
            if hasattr(cv2, "CAP_MSMF"):
                backends.append(("msmf", cv2.CAP_MSMF))
        backends.append(("default", cv2.CAP_ANY))

        for label, api in backends:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.notes.append(f"camera {idx} opened via {label}")
                    self._apply_props()
                    self.height, self.width = frame.shape[:2]
                    self.notes.append(f"capture size {self.width}x{self.height}")
                    return True
                self.notes.append(f"camera {idx} opened via {label} but returned no frame")
            cap.release()
            self.notes.append(f"camera {idx} unavailable via {label}")
        return False

    def _apply_props(self) -> None:
        w = int(get(self.cfg, "camera.width", 640))
        h = int(get(self.cfg, "camera.height", 480))
        fps = int(get(self.cfg, "camera.fps", 30))
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency
        except Exception as exc:
            self.notes.append(f"could not set camera properties: {exc}")

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# ==========================================================================
# Application
# ==========================================================================
class EyemouseApp:
    """Owns the whole runtime. Injectable pieces are marked in ``__init__``."""

    def __init__(self, cfg: Dict[str, Any], *,
                 tracker: Optional[Any] = None,
                 frame_source: Optional[Callable[[], Optional[np.ndarray]]] = None,
                 mouse: Optional[Any] = None,
                 screen: Optional[Tuple[int, int]] = None,
                 clock: Optional[Clock] = None,
                 show_overlay: Optional[bool] = None,
                 dry_run: bool = False) -> None:
        self.cfg = cfg
        self.clock = clock or Clock()
        self.dry_run = dry_run
        self.notes: List[str] = []
        self.running = False
        self.tracking = True
        self.quitting = False
        self._frame_source = frame_source
        self._tracker = tracker
        self._own_tracker = tracker is None

        # --- mouse -------------------------------------------------------
        self.mouse = mouse or mctl.MouseController(
            cfg, backend=mctl.NullMouse(*(screen or (1920, 1080))) if dry_run else None,
            screen=screen)
        self.notes.extend(getattr(self.mouse, "notes", []))
        self.screen_w, self.screen_h = self.mouse.screen_w, self.mouse.screen_h
        if screen:
            self.screen_w, self.screen_h = int(screen[0]), int(screen[1])
            self.mouse.set_screen_size(self.screen_w, self.screen_h)
        self.notes.append(f"screen {self.screen_w}x{self.screen_h}")

        # --- gaze --------------------------------------------------------
        self.extractor = gaze_mod.GazeExtractor(
            blink_ear=float(get(cfg, "clicks.wink.ear_threshold", 0.20)) * 0.95,
            min_face_width=float(get(cfg, "gaze.min_face_width", 0.10)),
        )
        self.pipeline = gaze_mod.GazePipeline(cfg, self.screen_w, self.screen_h)
        self.calibration_model: Optional[calib.CalibrationModel] = None

        # --- gestures ----------------------------------------------------
        self.gestures = gest.GestureEngine(cfg, self.clock)

        # --- ui ----------------------------------------------------------
        want_overlay = bool(get(cfg, "overlay.show", True)) if show_overlay is None else show_overlay
        self.show_overlay = want_overlay and cv2 is not None
        self.overlay = Overlay(cfg, self.clock)
        self.desktop_dot = DesktopDot(cfg)
        self.notes.extend(self.desktop_dot.notes)
        self.hotkeys = HotkeyManager(cfg)
        self.fps = FPSMeter(float(get(cfg, "performance.fps_smoothing", 0.1)), self.clock)
        self.timer = StageTimer(bool(get(cfg, "performance.show_timing", False)))

        # --- runtime state ----------------------------------------------
        self.frame_count = 0
        self._hands: Optional[Any] = None
        self._last_sample: Optional[gaze_mod.GazeSample] = None
        self._last_gaze: Optional[gaze_mod.GazeState] = None
        self._last_gesture: Optional[gest.GestureState] = None
        self._faces_lost = 0
        self._flash_state = "idle"
        self._flash_until = 0
        self._pending_recalibrate = False
        self._sniper_latched = False
        self.hooks: Dict[str, List[Callable[..., None]]] = {}

    # ------------------------------------------------------------------
    def on(self, event: str, fn: Callable[..., None]) -> None:
        """Test/simulation hook: subscribe to ``frame``, ``event``, ``quit``."""
        self.hooks.setdefault(event, []).append(fn)

    def _emit(self, event: str, *args, **kwargs) -> None:
        for fn in self.hooks.get(event, []):
            fn(*args, **kwargs)

    # ------------------------------------------------------------------
    def setup(self, force_calibration: bool = False, allow_calibration: bool = True) -> bool:
        """Open the camera and tracker, load or run calibration."""
        # --- tracker ---
        if self._tracker is None:
            try:
                self._tracker, notes = mpb.create_tracker(self.cfg)
                self.notes.extend(notes)
            except mpb.TrackerSetupError as exc:
                print(f"\n[Eyemouse] {exc}\n", file=sys.stderr)
                return False
        # --- camera ---
        if self._frame_source is None:
            self.camera = Camera(self.cfg)
            if not self.camera.open():
                print("\n[Eyemouse] Could not open a webcam:\n  "
                      + "\n  ".join(self.camera.notes)
                      + "\nTry: --camera 1, or close other apps using the camera.\n",
                      file=sys.stderr)
                return False
            self.notes.extend(self.camera.notes)
            self._frame_source = self.camera.read

        self.hotkeys.start()
        self.notes.extend(self.hotkeys.notes)
        if not self.hotkeys.global_hook:
            self.notes.append("hotkeys need the Eyemouse window focused (no global hook)")

        # --- calibration ---
        cal_path = self._calibration_path()
        loaded = None
        if bool(get(self.cfg, "calibration.auto_load", True)) and not force_calibration:
            loaded = calib.load_calibration(cal_path)
            if loaded is not None:
                age_days = (time.time() - (loaded.created or time.time())) / 86400.0
                self.notes.append(f"loaded calibration ({age_days:.1f} days old, "
                                  f"RMS {loaded.rms_px:.0f}px)")
                if (loaded.screen_w, loaded.screen_h) != (self.screen_w, self.screen_h):
                    self.notes.append(
                        f"calibration was for {loaded.screen_w}x{loaded.screen_h}; "
                        f"current screen is {self.screen_w}x{self.screen_h} -- "
                        f"coordinates are rescaled, but recalibrating is better")
        self._apply_calibration(loaded)

        if self.calibration_model is None and allow_calibration:
            ok = self.run_calibration()
            if not ok:
                return False
        elif self.calibration_model is None:
            self.notes.append("no calibration: cursor will not move until you calibrate (F8)")
        return True

    def _calibration_path(self) -> str:
        name = str(get(self.cfg, "calibration.file", "calibration.json"))
        if os.path.isabs(name):
            return name
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def _apply_calibration(self, model: Optional[calib.CalibrationModel]) -> None:
        self.calibration_model = model
        self.pipeline.set_calibration(model)
        self.pipeline.reset()
        self.gestures.reset()
        if model is not None:
            self.mouse.resync()

    # ------------------------------------------------------------------
    def run_calibration(self) -> bool:
        """Interactive 9-point calibration. Returns True on success."""
        n_points = int(get(self.cfg, "calibration.points", 9))
        inset = float(get(self.cfg, "calibration.inset_ratio", 0.12))
        grid = calib.calibration_grid(n_points, inset, self.screen_w, self.screen_h)
        session = calib.CalibrationSession(
            grid,
            samples_per_point=int(get(self.cfg, "calibration.samples_per_point", 30)),
            discard_first=int(get(self.cfg, "calibration.discard_first", 8)),
            clock=self.clock,
            screen_w=self.screen_w, screen_h=self.screen_h,
        )
        print(f"[Eyemouse] calibrating: {len(grid)} points, "
              f"{session.samples_per_point} samples each. Look AT each dot.")

        self._set_fullscreen(True)
        lost = 0
        t_start = time.monotonic()
        try:
            while not session.complete:
                if time.monotonic() - t_start > 240:
                    print("[Eyemouse] calibration timed out after 4 minutes")
                    return False
                frame = self._frame_source() if self._frame_source else None
                ts = self.clock.ms()
                face = self._tracker.process_face(frame, ts) if (self._tracker and frame is not None) else None
                sample = self.extractor.extract(
                    face.landmarks if face is not None else None, ts,
                    rgb=frame)
                if sample.valid:
                    lost = 0
                    session.add_sample(sample.gx, sample.gy, sample.hx, sample.hy)
                else:
                    lost += 1
                    if lost == 45:
                        print(f"[Eyemouse] face lost ({sample.reason}); "
                              f"re-centre yourself to continue")

                if self.show_overlay:
                    thumb = frame
                    if thumb is not None and bool(get(self.cfg, "camera.mirror_preview", True)):
                        thumb = cv2.flip(thumb, 1)
                    canvas = calib.draw_calibration_screen(
                        session, thumb=thumb,
                        radius=int(get(self.cfg, "calibration.point_radius_px", 14)),
                        canvas_w=self.screen_w, canvas_h=self.screen_h)
                    if canvas is not None:
                        if sample.valid:
                            px = int(clamp(sample.hx, 0, 1) * (canvas.shape[1] - 1))
                            py = int(clamp(sample.hy, 0, 1) * (canvas.shape[0] - 1))
                            cv2.circle(canvas, (px, py), 4, (255, 120, 120), -1)
                        cv2.imshow(WINDOW_NAME, canvas)
                else:
                    self._emit("calibration_progress", session.progress)
                    time.sleep(0.005)

                # One waitKey per iteration (inside _pump_keys). An extra one
                # here would double the calibration loop's latency.
                if self._pump_keys() in ("esc", "q"):
                    print("[Eyemouse] calibration cancelled")
                    return False
        finally:
            self._set_fullscreen(False)

        try:
            model = session.build(
                model=str(get(self.cfg, "calibration.model", "quadratic")),
                ridge=float(get(self.cfg, "calibration.ridge", 1e-4)),
                screen_w=self.screen_w, screen_h=self.screen_h)
        except ValueError as exc:
            print(f"[Eyemouse] calibration failed: {exc}")
            return False

        calib.save_calibration(model, self._calibration_path())
        self._apply_calibration(model)
        print(model.report())
        for line in session.history:
            print(f"  {line}")
        self.notes.append(f"calibrated: RMS {model.rms_px:.0f}px")
        self.overlay.say(f"Calibrated (RMS {model.rms_px:.0f}px)")
        return True

    def _set_fullscreen(self, on: bool) -> None:
        if cv2 is None or not self.show_overlay:
            return
        try:
            if on:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                w = int(get(self.cfg, "overlay.width", 980))
                cv2.resizeWindow(WINDOW_NAME, w, int(w * 0.62))
        except Exception as exc:
            self.notes.append(f"fullscreen toggle failed: {exc}")

    # ------------------------------------------------------------------
    def _pump_keys(self) -> Optional[str]:
        """Drain window + global keys. Returns 'esc'/'q' when quitting."""
        if cv2 is not None and self.show_overlay:
            code = cv2.waitKey(1) & 0xFF
            if code != 255:
                self.hotkeys.feed_keycode(code)
        for name in self.hotkeys.poll():
            if self._handle_hotkey(name):
                return name
        return None

    def _handle_hotkey(self, name: str) -> bool:
        """Returns True when the app should quit."""
        hk = get(self.cfg, "hotkeys", {}) or {}
        if name == str(get(hk, "quit", "esc")).lower() or name == str(get(hk, "quit_alt", "q")).lower():
            self.quitting = True
            self.running = False
            return True
        if name == str(get(hk, "recalibrate", "f8")).lower():
            self._pending_recalibrate = True
        elif name == str(get(hk, "toggle_tracking", "f7")).lower():
            self.tracking = not self.tracking
            self.overlay.say(f"Tracking {'ON' if self.tracking else 'PAUSED'}")
            if self.tracking:
                self.pipeline.reset()
                self.gestures.reset()
        elif name == str(get(hk, "toggle_sniper", "f6")).lower():
            self._sniper_latched = not getattr(self, "_sniper_latched", False)
            self.overlay.say(f"Sniper latch {'ON' if self._sniper_latched else 'OFF'}")
        elif name == str(get(hk, "save_config", "f9")).lower():
            try:
                path = save_config(self.cfg)
                self.overlay.say(f"Config saved: {os.path.basename(path)}")
                print(f"[Eyemouse] config saved to {path}")
            except Exception as exc:
                self.overlay.say(f"Config save FAILED: {exc}")
        return False

    # ------------------------------------------------------------------
    def _set_flash(self, state: str, ms: int) -> None:
        """Hold a desktop-dot colour for ``ms`` so a click is visible."""
        self._flash_state = state
        self._flash_until = self.clock.ms() + max(0, int(ms))

    def _handle_events(self, st: gest.GestureState) -> None:
        flash_ms = int(get(self.cfg, "overlay.flash_ms", 320))
        for ev in st.events:
            if ev.kind == gest.CLICK_LEFT:
                if self.tracking:
                    self.mouse.click("left")
                self.overlay.flash(f"LEFT CLICK ({ev.source})", "left")
                self._set_flash("click", flash_ms)
            elif ev.kind == gest.CLICK_RIGHT:
                if self.tracking:
                    self.mouse.click("right")
                self.overlay.flash(f"RIGHT CLICK ({ev.source})", "right")
                self._set_flash("click", flash_ms)
            elif ev.kind == gest.SCROLL:
                if self.tracking:
                    self.mouse.scroll(ev.amount)
                self._set_flash("scroll", flash_ms)
            elif ev.kind == gest.SCROLL_MODE_ON:
                self.overlay.say("Scroll mode ON - move eyes vertically")
            elif ev.kind == gest.SCROLL_MODE_OFF:
                self.overlay.say("Scroll mode OFF")
            elif ev.kind == gest.SNIPER_ON:
                self.overlay.say("Sniper mode (1/4 speed)")
                # The dot colour keys off gesture.sniper directly, so nothing
                # to latch here -- just lean the smoothing in.
                self.pipeline.set_smoothing_boost(float(get(self.cfg, "precision.smoothing_boost", 2.2)))
            elif ev.kind == gest.SNIPER_OFF:
                self.pipeline.set_smoothing_boost(1.0)
            self._emit("event", ev, st)

    # ------------------------------------------------------------------
    def step(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run exactly one pipeline iteration. Returns a dict of frame state.

        Split out from :meth:`run` so tests and the simulator can step the
        whole system deterministically.
        """
        # Timestamps come from the injectable clock: identical to monotonic
        # time in production, deterministic under simulation.
        ts = self.clock.ms()
        if frame is None and self._frame_source is not None:
            frame = self._frame_source()

        # Inference runs on a downscaled copy. MediaPipe returns normalized
        # coordinates, so the landmarks stay valid for the full-size frame the
        # overlay displays and the pupil fallback samples.
        proc = frame
        scale = float(get(self.cfg, "performance.process_scale", 1.0))
        if frame is not None and 0 < scale < 0.999 and cv2 is not None:
            fh, fw = frame.shape[:2]
            proc = cv2.resize(frame, (max(32, int(fw * scale)), max(24, int(fh * scale))),
                              interpolation=cv2.INTER_AREA)

        self.timer.start("face")
        face = self._tracker.process_face(proc, ts) if (self._tracker and proc is not None) else None
        self.timer.stop("face")

        face_lms = face.landmarks if (face is not None and face.present) else None

        # Hands only every N frames; carry the previous result forward.
        every_n = max(1, int(get(self.cfg, "performance.hands_every_n_frames", 3)))
        need_hands = (str(get(self.cfg, "scroll.mode", "edge")) == "palm"
                      or str(get(self.cfg, "clicks.gesture", "both")) in ("temple", "both"))
        if need_hands and proc is not None and self._tracker is not None and (self.frame_count % every_n == 0):
            self.timer.start("hands")
            self._hands = self._tracker.process_hands(proc, ts)
            self.timer.stop("hands")
        hands = self._hands if (self._hands is not None and getattr(self._hands, "present", False)) else None

        self.timer.start("gaze")
        sample = self.extractor.extract(face_lms, ts, rgb=frame)
        self._last_sample = sample
        gaze_state = self.pipeline.process(sample)
        self._last_gaze = gaze_state
        self.timer.stop("gaze")

        sniper_key = (self.hotkeys.is_held(str(get(self.cfg, "precision.sniper_key", "ctrl")))
                      or getattr(self, "_sniper_latched", False))
        self.timer.start("gestures")
        st = self.gestures.update(sample, gaze_state, face_lms, hands, sniper_key=sniper_key)
        self._last_gesture = st
        self.timer.stop("gestures")

        self._handle_events(st)

        # --- cursor ------------------------------------------------------
        self.timer.start("mouse")
        moved = False
        if self.tracking and gaze_state.valid and self.calibration_model is not None:
            moved = self.mouse.move_to(gaze_state.x, gaze_state.y,
                                       sniper=st.sniper, force=gaze_state.moved)
        self.timer.stop("mouse")

        if self.desktop_dot.enabled and gaze_state.valid:
            now = self.clock.ms()
            if st.sniper:
                dot_state = "sniper"
            elif now < self._flash_until:
                dot_state = self._flash_state
            else:
                dot_state = "idle"
            self.desktop_dot.update(int(gaze_state.x), int(gaze_state.y), dot_state)

        self.frame_count += 1
        fps = self.fps.tick()
        self._emit("frame", self.frame_count, gaze_state, st, fps)
        return {"frame": frame, "face": face_lms, "hands": hands, "sample": sample,
                "gaze": gaze_state, "gesture": st, "fps": fps, "moved": moved}

    # ------------------------------------------------------------------
    def run(self, max_frames: int = 0) -> int:
        """Main loop. Returns a process exit code."""
        if cv2 is not None and self.show_overlay:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            w = int(get(self.cfg, "overlay.width", 980))
            try:
                cv2.resizeWindow(WINDOW_NAME, w, int(w * 0.62))
            except Exception:
                pass
            self.overlay.setup_sliders(WINDOW_NAME)
        if self.desktop_dot.enabled:
            self.desktop_dot.start()

        self.running = True
        min_interval = 0.0
        max_fps = int(get(self.cfg, "performance.max_fps", 0))
        if max_fps > 0:
            min_interval = 1.0 / max_fps
        hk_map = self.hotkeys.hotkey_map()

        try:
            while self.running:
                loop_t0 = time.monotonic()
                if getattr(self, "_pending_recalibrate", False):
                    self._pending_recalibrate = False
                    self.tracking = True
                    self.run_calibration()

                state = self.step()
                if self.quitting:
                    break

                if self.show_overlay:
                    self.overlay.read_sliders(WINDOW_NAME)
                    self._push_live_config()
                    img = self.overlay.render(
                        state["frame"], gaze=state["gaze"], gesture=state["gesture"],
                        fps=state["fps"], tracking=self.tracking,
                        calibrated=self.calibration_model is not None,
                        mouse_stats=self.mouse.stats, timing=self.timer.summary(),
                        face=state["face"], hands=state["hands"],
                        screen=(self.screen_w, self.screen_h), hotkeys=hk_map)
                    if img is not None:
                        cv2.imshow(WINDOW_NAME, img)
                    if self._pump_keys() in ("esc", "q"):
                        break
                else:
                    if self._pump_keys() in ("esc", "q"):
                        break

                if min_interval > 0:
                    slack = min_interval - (time.monotonic() - loop_t0)
                    if slack > 0:
                        time.sleep(slack)
                if max_frames and self.frame_count >= max_frames:
                    break
        except KeyboardInterrupt:
            print("\n[Eyemouse] interrupted")
        finally:
            self.teardown()
        return 0

    def _push_live_config(self) -> None:
        """Propagate slider edits into the live objects."""
        self.pipeline.apply_config(self.cfg)
        self.gestures.apply_config(self.cfg)
        self.mouse.apply_config(self.cfg)

    # ------------------------------------------------------------------
    def teardown(self) -> None:
        self.running = False
        try:
            self.desktop_dot.stop()
        except Exception:
            pass
        try:
            self.hotkeys.stop()
        except Exception:
            pass
        if self._own_tracker and self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:
                pass
        cam = getattr(self, "camera", None)
        if cam is not None:
            cam.release()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
                # Windows needs one more event pump to actually close them.
                for _ in range(3):
                    cv2.waitKey(1)
            except Exception:
                pass
        self.mouse.close()
        self._emit("quit")

    # ------------------------------------------------------------------
    def print_diagnostics(self) -> None:
        print("[Eyemouse] startup notes:")
        for n in self.notes:
            print(f"  - {n}")


# ==========================================================================
# --check diagnostics
# ==========================================================================
def run_checks(cfg: Dict[str, Any], camera_index: Optional[int] = None) -> int:
    """Print an environment report and exit non-zero on a blocking problem."""
    print("=" * 68)
    print("Eyemouse environment check")
    print("=" * 68)
    problems = 0

    print(f"python     : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform   : {sys.platform}")
    if not sys.platform.startswith("win"):
        print("             NOTE: Eyemouse targets Windows; cursor control will")
        print("             be limited or unavailable elsewhere.")

    for mod in ("cv2", "numpy", "mediapipe", "pyautogui", "pynput"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "ok")
            print(f"{mod:<11}: {ver}")
        except Exception as exc:
            optional = mod in ("pynput",)
            print(f"{mod:<11}: MISSING ({type(exc).__name__}: {exc})"
                  + ("  [optional]" if optional else ""))
            if not optional:
                problems += 1

    info = mpb.probe()
    print(f"mediapipe api: {info.get('api')}")
    for e in info.get("errors", []):
        print(f"             ! {e}")
    if info.get("api") is None:
        problems += 1

    models = mpb.find_models()
    for name, path in models.items():
        if path:
            print(f"model {name:<22}: {path} ({os.path.getsize(path) // 1024} KB)")
        else:
            need = info.get("api") == "tasks"
            print(f"model {name:<22}: NOT FOUND" + (" (required for the Tasks API)" if need else ""))
            if need:
                problems += 1
    if info.get("api") == "tasks" and any(p is None for p in models.values()):
        print("             -> fix with: python tools/download_models.py")

    print(f"dpi aware  : {mctl.ensure_dpi_aware()}")
    print(f"screen     : {mctl.screen_size()}")

    mouse, notes = mctl.create_mouse(str(get(cfg, "mouse.backend", "auto")))
    for n in notes:
        print(f"mouse      ! {n}")
    print(f"mouse      : {mouse.name}")
    if isinstance(mouse, mctl.NullMouse):
        problems += 1

    if cv2 is not None:
        cam = Camera(cfg, camera_index)
        if cam.open():
            print(f"camera     : OK ({cam.width}x{cam.height})")
            for n in cam.notes:
                print(f"             - {n}")
            cam.release()
        else:
            print("camera     : FAILED")
            for n in cam.notes:
                print(f"             - {n}")
            problems += 1
    print("=" * 68)
    print("RESULT: ready" if problems == 0 else f"RESULT: {problems} problem(s) to fix")
    return 0 if problems == 0 else 1


# ==========================================================================
# CLI
# ==========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eyemouse",
        description="Webcam-only eye-tracker mouse for Windows (OpenCV + MediaPipe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gestures (defaults, all configurable in config.json):
  look around            move the cursor
  touch your LEFT temple left click        (or wink with your left eye)
  touch your RIGHT temple right click      (or wink with your right eye)
  touch BOTH temples     sniper mode (cursor at 1/4 speed)
  hold gaze at top edge  scroll up         (800 ms dwell)
  hold gaze at bot edge  scroll down
  hold still             dwell click
  stay still 1.4 s       left click (fallback)

Hotkeys:
  F8 recalibrate   F7 pause/resume   F6 sniper latch   F9 save config
  ESC or q quit
""")
    p.add_argument("--config", default=None, help="path to config.json")
    p.add_argument("--calibrate", action="store_true", help="force a fresh calibration")
    p.add_argument("--no-calibrate", action="store_true", help="skip calibration even if none is saved")
    p.add_argument("--camera", type=int, default=None, help="camera index (default: config)")
    p.add_argument("--dry-run", action="store_true", help="never move or click the real cursor")
    p.add_argument("--no-overlay", action="store_true", help="disable the preview window")
    p.add_argument("--check", action="store_true", help="diagnose the environment and exit")
    p.add_argument("--write-config", action="store_true", help="write a default config.json and exit")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    p.add_argument("--models-dir", default=None, help="where to look for the .task model bundles")
    p.add_argument("--quiet", action="store_true", help="suppress the startup notes")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.write_config:
        path = args.config or config_mod.default_config_path()
        write_default_config(path)
        print(f"[Eyemouse] wrote default config to {path}")
        return 0

    try:
        cfg = config_mod.load_config(args.config)
    except config_mod.ConfigError as exc:
        print(f"[Eyemouse] {exc}", file=sys.stderr)
        return 2
    if args.camera is not None:
        set_path(cfg, "camera.index", int(args.camera))
    if args.no_overlay:
        set_path(cfg, "overlay.show", False)

    if not args.quiet:
        for note in cfg["_meta"]["notes"]:
            if "clamped" in note or "not in" in note or "unknown" in note:
                print(f"[config] {note}")

    if args.check:
        return run_checks(cfg, args.camera)

    if cv2 is None:
        print("[Eyemouse] OpenCV is required: pip install opencv-python", file=sys.stderr)
        return 2

    app = EyemouseApp(cfg, dry_run=args.dry_run)
    if not args.quiet:
        app.print_diagnostics()
    ok = app.setup(force_calibration=args.calibrate,
                   allow_calibration=not args.no_calibrate)
    if not ok:
        app.teardown()
        return 1
    if not args.quiet:
        app.print_diagnostics()
        _print_help(cfg, app)
    return app.run(max_frames=args.frames)


def _print_help(cfg: Dict[str, Any], app: EyemouseApp) -> None:
    hk = get(cfg, "hotkeys", {}) or {}
    print("-" * 68)
    print("Eyemouse running. Look at the calibration to move the cursor.")
    print(f"  click gesture : {get(cfg, 'clicks.gesture')}   "
          f"scroll: {get(cfg, 'scroll.mode')}   "
          f"sniper factor: {get(cfg, 'precision.sniper_factor')}")
    print(f"  hotkeys       : {hk.get('recalibrate')}=recalibrate  "
          f"{hk.get('toggle_tracking')}=pause  {hk.get('toggle_sniper')}=sniper latch  "
          f"{hk.get('save_config')}=save config  ESC/q=quit")
    if not app.hotkeys.global_hook:
        print("  NOTE: no global keyboard hook (install pynput) - hotkeys only")
        print("        work while the Eyemouse preview window has focus.")
    print("-" * 68)


if __name__ == "__main__":
    sys.exit(main())
