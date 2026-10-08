"""Global hotkeys and hold-to-activate keys.

An eye tracker is useless if its hotkeys only work while its own window has
focus -- the whole point is to be interacting with *another* application. So
the primary mechanism is a ``pynput`` keyboard hook (a real global hook on
Windows), with two fallbacks:

1. no ``pynput`` -> try the ``keyboard`` package
2. neither -> OpenCV window keys only, via :meth:`HotkeyManager.feed_keycode`

Keys are **edge triggered**: the callback fires once on key-down and will not
re-fire on OS auto-repeat. Separately, :meth:`is_held` exposes the live
pressed state, which is what sniper mode needs (press and *hold*).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from config import get

__all__ = ["HotkeyManager", "normalize_key"]

# pynput reports modifier keys with a side suffix; fold those together so a
# config entry of "ctrl" matches both Ctrl keys.
_MODIFIER_ALIASES = {
    "ctrl_l": "ctrl", "ctrl_r": "ctrl", "ctrl": "ctrl",
    "shift_l": "shift", "shift_r": "shift", "shift": "shift",
    "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt", "alt": "alt",
    "cmd": "super", "cmd_l": "super", "cmd_r": "super",
    "super_l": "super", "super_r": "super",
    "esc": "esc", "escape": "esc",
    "return": "enter", "enter": "enter",
    "space": "space", "backspace": "backspace", "tab": "tab",
}

# OpenCV waitKey codes for the keys we bind by default. cv2 returns the ASCII
# code for printable keys and a large platform-specific constant for the rest,
# so only the printable ones are handled here -- function keys need the hook.
_CV2_KEYCODES: Dict[int, str] = {
    27: "esc",
    ord("q"): "q",
    13: "enter",
    32: "space",
}


def normalize_key(key: Any) -> str:
    """Map a pynput key object (or a raw string) onto a canonical name."""
    if key is None:
        return ""
    if isinstance(key, str):
        name = key.strip().lower()
    else:
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        name = str(getattr(key, "name", "") or "").strip().lower()
    if not name:
        return ""
    return _MODIFIER_ALIASES.get(name, name)


class HotkeyManager:
    """Collects key-down edges and live hold state from the best available hook."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self._pressed: Deque[str] = deque(maxlen=256)
        # Held state is ref-counted per canonical name, keyed by the *raw* key
        # identity. Without the raw key, releasing one Ctrl while the other is
        # still down would drop sniper mode mid-use, because both normalise to
        # the same name.
        self._held_raw: Set[Any] = set()
        self._held: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._listener: Any = None
        self._backend = "none"
        self.notes: List[str] = []
        self.enabled = True

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def global_hook(self) -> bool:
        """True when hotkeys work even if another app has focus."""
        return self._backend in ("pynput", "keyboard")

    def start(self) -> None:
        if self._listener is not None:
            return
        if self._try_pynput():
            return
        if self._try_keyboard_pkg():
            return
        self._backend = "cv2-window-keys"
        self.notes.append(
            "no global keyboard hook available (install pynput); hotkeys only "
            "work while the Eyemouse preview window has focus"
        )

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is None:
            return
        try:
            stop = getattr(listener, "stop", None)
            if stop is not None:
                stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _try_pynput(self) -> bool:
        try:
            from pynput import keyboard
        except Exception as exc:
            self.notes.append(f"pynput keyboard hook unavailable: {type(exc).__name__}: {exc}")
            return False
        try:
            self._listener = keyboard.Listener(on_press=self._on_press,
                                               on_release=self._on_release)
            self._listener.daemon = True
            self._listener.start()
            self._backend = "pynput"
            self.notes.append("global hotkeys via pynput")
            return True
        except Exception as exc:
            self._listener = None
            self.notes.append(f"pynput listener failed to start: {type(exc).__name__}: {exc}")
            return False

    def _try_keyboard_pkg(self) -> bool:
        try:
            import keyboard as kb
        except Exception as exc:
            self.notes.append(f"'keyboard' package unavailable: {type(exc).__name__}: {exc}")
            return False
        try:
            kb.hook(self._on_keyboard_pkg_event)
            self._backend = "keyboard"
            self.notes.append("global hotkeys via the 'keyboard' package")
            return True
        except Exception as exc:
            self.notes.append(f"'keyboard' hook failed: {type(exc).__name__}: {exc}")
            return False

    # ------------------------------------------------------------------
    def _on_press(self, key: Any) -> None:
        name = normalize_key(key)
        if not name:
            return
        raw = self._raw_id(key)
        with self._lock:
            if raw not in self._held_raw:   # suppresses OS auto-repeat
                self._pressed.append(name)
                self._held_raw.add(raw)
                self._held[name] = self._held.get(name, 0) + 1

    def _on_release(self, key: Any) -> None:
        name = normalize_key(key)
        if not name:
            return
        raw = self._raw_id(key)
        with self._lock:
            if raw in self._held_raw:
                self._held_raw.discard(raw)
                left = self._held.get(name, 0) - 1
                if left > 0:
                    self._held[name] = left
                else:
                    self._held.pop(name, None)

    @staticmethod
    def _raw_id(key: Any) -> Any:
        """A *stable* identity for a physical key press.

        Built from the key's attributes rather than the object itself: a
        backend that constructs a fresh key object per event would otherwise
        defeat auto-repeat suppression (every repeat would look like a new
        key). Ctrl-left and Ctrl-right still get distinct identities, which is
        what keeps a two-Ctrl hold from dropping sniper mode.
        """
        if isinstance(key, str):
            return ("str", key)
        char = getattr(key, "char", None)
        name = getattr(key, "name", None)
        vk = getattr(key, "vk", None)
        if char is not None or name is not None or vk is not None:
            return ("key", char, name, vk)
        try:
            hash(key)
            return ("obj", key)
        except TypeError:
            return ("repr", repr(key))

    def _on_keyboard_pkg_event(self, event: Any) -> None:
        name = normalize_key(getattr(event, "name", None))
        if not name:
            return
        if getattr(event, "event_type", "") == "down":
            self._on_press(name)
        else:
            self._on_release(name)

    def feed_keycode(self, code: int) -> Optional[str]:
        """Feed an OpenCV ``waitKey`` result. Returns the canonical name.

        ``waitKey`` returns an ASCII code for printable keys, but function keys
        come back as large platform-specific constants (on Windows the scan
        code sits in the high word), so the low byte is retried as a fallback.
        """
        if code is None or code < 0:
            return None
        name = self._decode(code)
        if name is None and code > 0xFF:
            # Windows reports special keys with the scan code in the high word
            # and the virtual key in the low word; try both.
            name = self._decode(code & 0xFF) or self._decode((code >> 16) & 0xFF)
        if name:
            with self._lock:
                self._pressed.append(name)
        return name

    @staticmethod
    def _decode(code: int) -> Optional[str]:
        if code in _CV2_KEYCODES:
            return _CV2_KEYCODES[code]
        # Printable ASCII only: chr(255) is 'ÿ', which .isalnum() accepts, so
        # an unrestricted range would treat "no key pending" as a keypress.
        if 32 <= code < 127:
            ch = chr(code).lower()
            if ch.isalnum():
                return ch
        return None

    # ------------------------------------------------------------------
    def is_held(self, name: str) -> bool:
        """Live hold state. ``"ctrl"`` matches either Ctrl key."""
        target = normalize_key(name)
        if not target:
            return False
        with self._lock:
            return self._held.get(target, 0) > 0

    def any_held(self, names: Any) -> bool:
        return any(self.is_held(n) for n in (names or []))

    def poll(self) -> List[str]:
        """Drain and return key-down edges since the last call."""
        with self._lock:
            out = list(self._pressed)
            self._pressed.clear()
        return out

    def pressed(self, name: str) -> bool:
        """True if ``name`` had a key-down edge since the last :meth:`poll`."""
        target = normalize_key(name)
        with self._lock:
            if target in self._pressed:
                try:
                    self._pressed.remove(target)
                except ValueError:
                    pass
                return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._held.clear()
            self._held_raw.clear()

    # ------------------------------------------------------------------
    def hotkey_map(self) -> Dict[str, str]:
        """Configured action -> key name, for the on-screen help text."""
        hk = get(self.cfg, "hotkeys", {}) or {}
        return {k: normalize_key(v) for k, v in hk.items() if v}
