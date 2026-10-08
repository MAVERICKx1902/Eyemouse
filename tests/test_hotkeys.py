"""Hotkey normalisation, edge triggering and hold state."""

from __future__ import annotations

import pytest

from config import load_config
from hotkeys import HotkeyManager, normalize_key

NOFILE = "/definitely/not/here/config.json"


class _Key:
    """Stand-in for a pynput key object."""

    def __init__(self, char=None, name=None):
        self.char = char
        self.name = name


class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("F8", "f8"), ("f8", "f8"), (" ESC ", "esc"), ("escape", "esc"),
        ("q", "q"), ("Q", "q"), ("space", "space"), ("enter", "enter"),
        ("return", "enter"),
    ])
    def test_strings(self, raw, expected):
        assert normalize_key(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("ctrl_l", "ctrl"), ("ctrl_r", "ctrl"), ("CTRL_L", "ctrl"),
        ("shift_l", "shift"), ("shift_r", "shift"),
        ("alt_l", "alt"), ("alt_gr", "alt"),
        ("cmd", "super"), ("cmd_l", "super"),
    ])
    def test_modifiers_collapse_to_one_name(self, raw, expected):
        assert normalize_key(raw) == expected

    def test_character_key_object(self):
        assert normalize_key(_Key(char="q")) == "q"
        assert normalize_key(_Key(char="Q")) == "q"

    def test_named_key_object(self):
        assert normalize_key(_Key(name="f8")) == "f8"
        assert normalize_key(_Key(name="esc")) == "esc"
        assert normalize_key(_Key(name="ctrl_l")) == "ctrl"

    def test_char_wins_over_name(self):
        assert normalize_key(_Key(char="a", name="something")) == "a"

    def test_junk_is_empty(self):
        assert normalize_key(None) == ""
        assert normalize_key("") == ""
        assert normalize_key(_Key()) == ""


class TestEdgeTriggering:
    def test_press_then_release_produces_one_event(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        hk._on_release(_Key(name="f8"))
        assert hk.poll() == ["f8"]

    def test_auto_repeat_is_suppressed(self):
        """Holding a key must not queue hundreds of recalibrations."""
        hk = HotkeyManager()
        for _ in range(50):
            hk._on_press(_Key(name="f8"))
        assert hk.poll() == ["f8"]

    def test_release_then_press_again_re_fires(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        hk.poll()
        hk._on_release(_Key(name="f8"))
        hk._on_press(_Key(name="f8"))
        assert hk.poll() == ["f8"]

    def test_multiple_keys_keep_their_order(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        hk._on_press(_Key(char="q"))
        assert hk.poll() == ["f8", "q"]

    def test_poll_drains(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        assert hk.poll() == ["f8"]
        assert hk.poll() == []

    def test_pressed_consumes_a_single_key(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        hk._on_press(_Key(name="f7"))
        assert hk.pressed("f8") is True
        assert hk.pressed("f8") is False
        assert hk.poll() == ["f7"]


class TestHoldState:
    def test_is_held_tracks_press_and_release(self):
        hk = HotkeyManager()
        assert hk.is_held("ctrl") is False
        hk._on_press(_Key(name="ctrl_l"))
        assert hk.is_held("ctrl") is True
        hk._on_release(_Key(name="ctrl_l"))
        assert hk.is_held("ctrl") is False

    def test_either_ctrl_key_satisfies_ctrl(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="ctrl_r"))
        assert hk.is_held("ctrl") is True

    def test_both_ctrl_keys_need_releasing(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="ctrl_l"))
        hk._on_press(_Key(name="ctrl_r"))
        hk._on_release(_Key(name="ctrl_l"))
        assert hk.is_held("ctrl") is True
        hk._on_release(_Key(name="ctrl_r"))
        assert hk.is_held("ctrl") is False

    def test_any_held(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="shift_l"))
        assert hk.any_held(["ctrl", "shift"]) is True
        assert hk.any_held(["ctrl"]) is False
        assert hk.any_held([]) is False
        assert hk.any_held(None) is False

    def test_is_held_normalises_its_argument(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="ctrl_l"))
        assert hk.is_held("CTRL_L") is True

    def test_clear_drops_everything(self):
        hk = HotkeyManager()
        hk._on_press(_Key(name="f8"))
        hk._on_press(_Key(char="q"))
        hk.clear()
        assert hk.poll() == []
        assert hk.is_held("q") is False


class TestCv2Keycodes:
    def test_escape(self):
        hk = HotkeyManager()
        assert hk.feed_keycode(27) == "esc"
        assert hk.poll() == ["esc"]

    def test_printable_characters(self):
        hk = HotkeyManager()
        assert hk.feed_keycode(ord("q")) == "q"
        assert hk.feed_keycode(ord("Q")) == "q"
        assert hk.feed_keycode(ord("a")) == "a"

    def test_space_and_enter(self):
        hk = HotkeyManager()
        assert hk.feed_keycode(32) == "space"
        assert hk.feed_keycode(13) == "enter"

    def test_negative_and_none_are_ignored(self):
        hk = HotkeyManager()
        assert hk.feed_keycode(-1) is None
        assert hk.feed_keycode(None) is None
        assert hk.poll() == []

    def test_no_waitkey_pending_is_ignored(self):
        """cv2.waitKey returns 255 when nothing was pressed."""
        hk = HotkeyManager()
        assert hk.feed_keycode(255) is None

    def test_windows_function_key_high_word_fallback(self):
        """Windows packs the scan code into the high word for special keys."""
        hk = HotkeyManager()
        assert hk.feed_keycode(0x001B0000) == "esc"

    def test_non_printable_low_codes_are_ignored(self):
        hk = HotkeyManager()
        assert hk.feed_keycode(8) == "backspace" or hk.feed_keycode(8) is None


class TestKeyboardPackageBackend:
    def test_down_and_up_events(self):
        hk = HotkeyManager()

        class Ev:
            def __init__(self, name, event_type):
                self.name, self.event_type = name, event_type

        hk._on_keyboard_pkg_event(Ev("f8", "down"))
        assert hk.is_held("f8") is True
        assert hk.poll() == ["f8"]
        hk._on_keyboard_pkg_event(Ev("f8", "up"))
        assert hk.is_held("f8") is False

    def test_unknown_event_name_is_ignored(self):
        hk = HotkeyManager()
        hk._on_keyboard_pkg_event(type("E", (), {"name": None, "event_type": "down"})())
        assert hk.poll() == []


class TestConfigIntegration:
    def test_hotkey_map_comes_from_config(self):
        cfg = load_config(NOFILE)
        hk = HotkeyManager(cfg)
        m = hk.hotkey_map()
        assert m["recalibrate"] == "f8"
        assert m["quit"] == "esc"
        assert m["toggle_tracking"] == "f7"
        assert m["save_config"] == "f9"

    def test_backend_before_start_is_none(self):
        assert HotkeyManager().backend == "none"
        assert HotkeyManager().global_hook is False

    def test_stop_without_start_is_safe(self):
        hk = HotkeyManager()
        hk.stop()
        hk.stop()

    def test_enabled_flag_defaults_true(self):
        assert HotkeyManager().enabled is True
