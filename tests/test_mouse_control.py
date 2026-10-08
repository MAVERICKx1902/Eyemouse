"""Cursor backend selection, clamping, sniper scaling, DPI plumbing."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

import mouse_control as mc
from config import load_config, set_path

NOFILE = "/definitely/not/here/config.json"
SCREEN = (1600, 900)


@pytest.fixture
def cfg():
    c = load_config(NOFILE)
    c["_meta"]["notes"] = []
    return c


@pytest.fixture
def null(cfg):
    return mc.MouseController(cfg, backend=mc.NullMouse(*SCREEN), screen=SCREEN)


# --------------------------------------------------------------------------
# NullMouse
# --------------------------------------------------------------------------
class TestNullMouse:
    def test_records_moves_clicks_and_scrolls(self):
        m = mc.NullMouse(1000, 800)
        m.move(10, 20)
        m.click("left")
        m.click("right")
        m.scroll(3)
        m.scroll(0)
        assert m.moves == [(10.0, 20.0)]
        assert m.clicks == ["left", "right"]
        assert m.scrolls == [3]          # zero-amount scrolls are dropped
        assert m.position() == (10, 20)
        assert m.size() == (1000, 800)

    def test_starting_position_is_centred(self):
        m = mc.NullMouse(1000, 800)
        assert m.position() == (500, 400)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
class TestCreateMouse:
    def test_none_selects_null(self):
        m, notes = mc.create_mouse("none")
        assert isinstance(m, mc.NullMouse)

    def test_unknown_backend_degrades_to_null_with_a_note(self):
        m, notes = mc.create_mouse("telepathy")
        assert isinstance(m, mc.NullMouse)
        assert any("unknown mouse backend" in n for n in notes)
        assert any("NullMouse" in n for n in notes)

    def test_auto_never_raises(self):
        m, notes = mc.create_mouse("auto", null_size=SCREEN)
        assert m is not None
        w, h = m.size()
        assert w > 0 and h > 0

    def test_requested_backend_is_honoured_when_installed(self, fake_pyautogui):
        m, notes = mc.create_mouse("pyautogui", null_size=SCREEN)
        assert m.name == "pyautogui"

    def test_pyautogui_failsafe_is_disabled(self, fake_pyautogui):
        """A gaze at the screen corner must not raise FailSafeException."""
        mc.create_mouse("pyautogui", null_size=SCREEN)
        mod = sys.modules["pyautogui"]
        assert mod.FAILSAFE is False
        assert mod.PAUSE == 0.0

    def test_broken_backend_import_degrades_gracefully(self, monkeypatch):
        """A backend whose *import* blows up must not take the app down.

        This is not hypothetical: on a machine without a display, importing
        pyautogui raises KeyError('DISPLAY') from mouseinfo -- not ImportError.
        """
        import types
        broken = types.ModuleType("pyautogui")

        def explode(*a, **k):
            raise KeyError("DISPLAY")

        broken.FAILSAFE = True
        broken.PAUSE = 0.1
        for name in ("moveTo", "click", "scroll", "size", "position"):
            setattr(broken, name, explode)
        monkeypatch.setitem(sys.modules, "pyautogui", broken)
        monkeypatch.setitem(sys.modules, "pynput", None)   # forces an ImportError
        m, notes = mc.create_mouse("auto", null_size=SCREEN)
        assert isinstance(m, mc.NullMouse)
        assert any("DISPLAY" in n for n in notes), notes


# --------------------------------------------------------------------------
# Clamping and movement
# --------------------------------------------------------------------------
class TestMovement:
    def test_target_is_clamped_inside_the_inset(self, null, cfg):
        set_path(cfg, "mouse.screen_inset_px", 20)
        null.move_to(-5000, -5000)
        assert null.position == (20.0, 20.0)
        null.move_to(99999, 99999)
        assert null.position == (1580.0, 880.0)

    def test_inset_zero_reaches_the_true_corner(self, null, cfg):
        set_path(cfg, "mouse.screen_inset_px", 0)
        null.move_to(0, 0)
        assert null.position == (0.0, 0.0)
        null.move_to(1600, 900)
        assert null.position == (1600.0, 900.0)

    def test_clamping_can_be_disabled(self, null, cfg):
        set_path(cfg, "mouse.clamp_to_screen", False)
        set_path(cfg, "mouse.screen_inset_px", 50)
        null.move_to(-100, -100)
        assert null.position == (-100.0, -100.0)

    def test_force_false_skips_a_repeat_position(self, null):
        null.move_to(400, 300)
        n = null.backend.moves.__len__()
        assert null.move_to(400, 300, force=False) is False
        assert len(null.backend.moves) == n

    def test_force_true_always_moves(self, null):
        null.move_to(400, 300)
        assert null.move_to(400, 300, force=True) is True

    def test_moves_are_counted(self, null):
        for i in range(5):
            null.move_to(100 + i * 10, 100)
        assert null.stats.moves == 5

    def test_resync_adopts_the_backend_position(self, null):
        null.backend.move(123, 456)
        null.resync()
        assert null.position == (123.0, 456.0)

    def test_resync_forces_the_next_move(self, null):
        null.move_to(400, 300)
        null.resync(900, 800)
        assert null.move_to(400, 300, force=False) is True

    def test_set_screen_size_rescales_clamping(self, null):
        null.set_screen_size(3200, 1800)
        null.move_to(99999, 99999)
        assert null.position[0] > 1600

    def test_apply_config_picks_up_a_new_inset(self, null, cfg):
        null.move_to(0, 0)
        set_path(cfg, "mouse.screen_inset_px", 100)
        null.apply_config(cfg)
        null.move_to(0, 0)
        assert null.position == (100.0, 100.0)


# --------------------------------------------------------------------------
# Sniper mode
# --------------------------------------------------------------------------
class TestSniper:
    def test_single_step_is_scaled_by_the_factor(self, null, cfg):
        set_path(cfg, "precision.sniper_factor", 0.25)
        null.resync(800, 450)
        null.move_to(1200, 450, sniper=True)
        assert null.position[0] == pytest.approx(800 + 400 * 0.25)

    def test_normal_move_is_absolute(self, null, cfg):
        null.resync(800, 450)
        null.move_to(1200, 450, sniper=False)
        assert null.position[0] == pytest.approx(1200)

    def test_converges_on_the_target(self, null, cfg):
        """Repeated sniper steps must approach, never overshoot or stall."""
        set_path(cfg, "precision.sniper_factor", 0.25)
        null.resync(0, 0)
        target = 1000.0
        positions = []
        for _ in range(60):
            null.move_to(target, 0, sniper=True)
            positions.append(null.position[0])
        assert positions == sorted(positions), "must approach monotonically"
        assert all(p <= target + 1e-9 for p in positions), "must not overshoot"
        assert abs(positions[-1] - target) < 5.0, "must converge"

    def test_sub_pixel_steps_are_not_rounded_away(self, null, cfg):
        """The tracked position keeps its fractional part.

        If it were rounded each frame, a 0.25 factor on a 2px residual would
        produce a 0.5px step that rounds to zero and the cursor would freeze
        short of the target forever.
        """
        set_path(cfg, "precision.sniper_factor", 0.25)
        set_path(cfg, "mouse.screen_inset_px", 0)   # else 3.0 is clamped to 6
        null.resync(0, 0)
        for _ in range(200):
            null.move_to(3.0, 0.0, sniper=True)
        assert null.position[0] == pytest.approx(3.0, abs=0.01)

    def test_factor_is_clamped(self, null, cfg):
        set_path(cfg, "precision.sniper_factor", 5.0)   # out of range
        null.resync(0, 0)
        null.move_to(100, 0, sniper=True)
        assert 0 < null.position[0] <= 100

    def test_sniper_frames_are_counted(self, null, cfg):
        null.move_to(500, 500, sniper=True)
        null.move_to(600, 500, sniper=True)
        assert null.stats.sniper_frames == 2

    def test_sniper_still_clamps_to_the_screen(self, null, cfg):
        set_path(cfg, "precision.sniper_factor", 1.0)
        null.move_to(-9999, -9999, sniper=True)
        assert null.position[0] >= 0 and null.position[1] >= 0


# --------------------------------------------------------------------------
# Buttons
# --------------------------------------------------------------------------
class TestButtons:
    def test_click_counts_by_button(self, null):
        null.click("left")
        null.click("left")
        null.click("right")
        assert null.stats.clicks["left"] == 2
        assert null.stats.clicks["right"] == 1
        assert null.stats.last_click == "right"

    def test_unknown_button_falls_back_to_left(self, null):
        null.click("middle_finger")
        assert null.backend.clicks == ["left"]

    def test_zero_scroll_is_ignored(self, null):
        assert null.scroll(0) is False
        assert null.stats.scrolls == 0

    def test_scroll_accumulates_units(self, null):
        null.scroll(3)
        null.scroll(-5)
        assert null.stats.scrolls == 2
        assert null.stats.scroll_units == -2

    def test_backend_failure_is_reported_not_raised(self, cfg):
        class Broken(mc.NullMouse):
            def click(self, button="left"):
                raise RuntimeError("os refused")

        m = mc.MouseController(cfg, backend=Broken(*SCREEN), screen=SCREEN)
        assert m.click("left") is False
        assert any("click failed" in n for n in m.notes)


# --------------------------------------------------------------------------
# Platform helpers
# --------------------------------------------------------------------------
class TestPlatform:
    def test_ensure_dpi_aware_returns_a_string(self):
        result = mc.ensure_dpi_aware()
        assert isinstance(result, str) and result

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows-only")
    def test_dpi_aware_on_windows(self):
        assert mc.ensure_dpi_aware() != "n/a (not Windows)"

    def test_screen_size_is_positive(self):
        w, h = mc.screen_size()
        assert w > 0 and h > 0

    def test_screen_size_falls_back_without_a_display(self, monkeypatch):
        monkeypatch.setattr(mc, "_IS_WINDOWS", False)
        def boom():
            raise RuntimeError("no display")
        monkeypatch.setattr(mc, "_size_via_pyautogui", boom)
        assert mc.screen_size() == (1920, 1080)

    def test_dpi_awareness_can_be_disabled(self, cfg):
        set_path(cfg, "mouse.dpi_aware", False)
        m = mc.MouseController(cfg, backend=mc.NullMouse(*SCREEN), screen=SCREEN)
        assert not any("dpi awareness" in n for n in m.notes)

    def test_close_is_safe(self, null):
        null.close()
        null.close()


# --------------------------------------------------------------------------
# The real Windows backends, exercised through injected fake modules.
# pynput cannot be installed in this sandbox and pyautogui needs a display,
# so without these the code that actually moves the user's cursor on Windows
# would never execute at all.
# --------------------------------------------------------------------------
class _FakeController:
    """Mimics ``pynput.mouse.Controller``: a position property and click/scroll."""

    def __init__(self):
        self._pos = (0, 0)
        self.clicks = []
        self.scrolls = []
        self.screen_size = (1920, 1080)

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, value):
        self._pos = tuple(value)

    def click(self, button, count=1):
        self.clicks.append((button, count))

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))


@pytest.fixture
def fake_pynput(monkeypatch):
    import types

    created = []

    class Controller(_FakeController):
        def __init__(self):
            super().__init__()
            created.append(self)

    class Button:
        left = "Button.left"
        right = "Button.right"
        middle = "Button.middle"

    mod = types.ModuleType("pynput")
    mouse = types.ModuleType("pynput.mouse")
    mouse.Controller = Controller
    mouse.Button = Button
    mod.mouse = mouse
    monkeypatch.setitem(sys.modules, "pynput", mod)
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)
    return SimpleNamespace(created=created, Button=Button)


@pytest.fixture
def fake_pyautogui(monkeypatch):
    import types

    calls = SimpleNamespace(moves=[], clicks=[], scrolls=[], sizes=[(1920, 1080)])
    mod = types.ModuleType("pyautogui")
    mod.FAILSAFE = True
    mod.PAUSE = 0.2
    mod.moveTo = lambda x, y, **kw: calls.moves.append((x, y, kw))
    mod.position = lambda: (calls.moves[-1][0], calls.moves[-1][1]) if calls.moves else (0, 0)
    mod.click = lambda button=None, **kw: calls.clicks.append((button, kw))
    mod.scroll = lambda amount, **kw: calls.scrolls.append((amount, kw))
    mod.size = lambda: calls.sizes[-1]
    monkeypatch.setitem(sys.modules, "pyautogui", mod)
    return calls


class TestPynputBackend:
    def test_construct_reads_the_button_enum(self, fake_pynput):
        m = mc.PynputMouse()
        assert m._buttons == {"left": "Button.left", "right": "Button.right",
                              "middle": "Button.middle"}
        assert m.position() == (0, 0)

    def test_move_sends_ints(self, fake_pynput):
        m = mc.PynputMouse()
        m.move(640.4, 360.6)
        # int(round(...)): pynput's position setter wants ints on Windows.
        assert fake_pynput.created[0].position == (640, 361)
        assert m.position() == (640, 361)

    def test_move_skips_a_noop(self, fake_pynput):
        m = mc.PynputMouse()
        m.move(100, 100)
        m.move(100, 100)
        assert len(fake_pynput.created) == 1

    def test_controller_keeps_floats_while_the_backend_rounds(self, cfg, fake_pynput):
        m = mc.MouseController(cfg, backend=mc.PynputMouse(), screen=(1920, 1080))
        m.resync(0, 0)
        m.move_to(640.4, 360.6)
        assert fake_pynput.created[0].position == (640, 361)
        assert m.position == pytest.approx((640.4, 360.6))

    def test_click_uses_the_button_enum(self, cfg, fake_pynput):
        m = mc.MouseController(cfg, backend=mc.PynputMouse(), screen=(1920, 1080))
        m.click("left")
        m.click("right")
        assert [c[0] for c in fake_pynput.created[0].clicks] == ["Button.left", "Button.right"]

    def test_scroll_passes_a_vertical_delta(self, fake_pynput):
        m = mc.PynputMouse()
        m.scroll(5)
        assert fake_pynput.created[0].scrolls == [(0, 5)]

    def test_zero_scroll_is_suppressed(self, fake_pynput):
        m = mc.PynputMouse()
        m.scroll(0)
        assert fake_pynput.created[0].scrolls == []

    def test_position_reads_the_real_cursor(self, fake_pynput):
        m = mc.PynputMouse()
        m._ctrl.position = (800, 450)
        assert m.position() == (800, 450)

    def test_resync_adopts_the_real_cursor_position(self, cfg, fake_pynput):
        m = mc.MouseController(cfg, backend=mc.PynputMouse(), screen=(1920, 1080))
        m.move_to(100, 100)
        m.backend._ctrl.position = (1500, 800)    # the user moved the real mouse
        m.resync()
        assert m.position == pytest.approx((1500.0, 800.0))

    def test_backend_crash_is_swallowed(self, cfg, fake_pynput):
        m = mc.MouseController(cfg, backend=mc.PynputMouse(), screen=(1920, 1080))

        def boom(*a):
            raise OSError("device removed")

        m.backend.move = boom
        assert m.move_to(500, 500) is False


class TestPyAutoGUIBackend:
    def test_construct_disables_failsafe_and_pause(self, fake_pyautogui):
        mc.PyAutoGUIMouse()
        mod = sys.modules["pyautogui"]
        assert mod.FAILSAFE is False, "corner-slamming must not kill the app"
        assert mod.PAUSE == 0.0, "pyautogui sleeps 0.1s per call unless PAUSE is 0"

    def test_move_passes_through(self, cfg, fake_pyautogui):
        m = mc.MouseController(cfg, backend=mc.PyAutoGUIMouse(), screen=(1920, 1080))
        m.resync(0, 0)
        m.move_to(123.6, 456.2)
        # The backend rounds to ints; MouseController keeps the float itself.
        assert fake_pyautogui.moves[-1][:2] == (124, 456)
        assert m.position == pytest.approx((123.6, 456.2))

    def test_click_passes_button_name(self, cfg, fake_pyautogui):
        m = mc.MouseController(cfg, backend=mc.PyAutoGUIMouse(), screen=(1920, 1080))
        m.click("right")
        assert fake_pyautogui.clicks[-1][0] == "right"

    def test_scroll_passes_signed_amount(self, cfg, fake_pyautogui):
        m = mc.MouseController(cfg, backend=mc.PyAutoGUIMouse(), screen=(1920, 1080))
        m.scroll(4)
        m.scroll(-4)
        assert [a for a, _ in fake_pyautogui.scrolls] == [4, -4]

    def test_screen_size_falls_back_to_pyautogui(self, fake_pyautogui, monkeypatch):
        monkeypatch.setattr(mc, "_IS_WINDOWS", False)
        fake_pyautogui.sizes.append((2560, 1440))
        assert mc.screen_size() == (2560, 1440)

    def test_screen_size_uses_a_sane_default_when_nothing_works(self, monkeypatch):
        monkeypatch.setattr(mc, "_IS_WINDOWS", False)
        monkeypatch.setitem(sys.modules, "pyautogui", None)   # import fails
        w, h = mc.screen_size()
        assert (w, h) == (1920, 1080)

    def test_pynput_is_preferred_over_pyautogui(self, cfg, fake_pynput, fake_pyautogui):
        m, notes = mc.create_mouse("auto")
        assert m.name == "pynput"
