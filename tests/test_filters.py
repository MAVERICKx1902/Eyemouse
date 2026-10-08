"""Smoothing and deadzone behaviour.

These are the components that decide how the cursor *feels*, so the tests
assert on measurable properties (noise reduction, lag, convergence) rather
than on exact coefficient values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaze import (Deadzone, KalmanCV2D, MovingAverage, OneEuroFilter2D,
                  build_filter_chain)
from config import load_config, set_path
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Moving average
# --------------------------------------------------------------------------
class TestMovingAverage:
    def test_converges_to_constant_input(self):
        ma = MovingAverage(5)
        for _ in range(20):
            out = ma(100.0, 200.0)
        assert out == pytest.approx((100.0, 200.0))

    def test_window_length_respected(self):
        ma = MovingAverage(3)
        ma(0.0, 0.0); ma(3.0, 0.0); ma(6.0, 0.0)
        assert ma(9.0, 0.0)[0] == pytest.approx(6.0)   # mean of 3, 6, 9

    def test_first_sample_does_not_ramp_from_origin(self):
        """A cursor jumping from (0,0) on the first frame is a real bug."""
        ma = MovingAverage(8)
        out = ma(1500.0, 800.0)
        assert out == pytest.approx((1500.0, 800.0))

    def test_set_window_preserves_history(self):
        ma = MovingAverage(3)
        for v in (10.0, 20.0, 30.0):
            ma(v, 0.0)
        ma.set_window(5)
        assert ma.window == 5
        # History survived, so the mean is still near 30 rather than restarting.
        assert ma(30.0, 0.0)[0] == pytest.approx(22.5, abs=1.0)

    def test_set_window_noop_when_unchanged(self):
        ma = MovingAverage(4)
        for v in (1.0, 2.0, 3.0, 4.0):
            ma(v, 0.0)
        buf_before = ma._buf.copy()
        n_before = ma._n
        ma.set_window(4)          # same window -> must not touch the buffer
        assert ma._n == n_before
        assert np.array_equal(ma._buf, buf_before)

    def test_reduces_noise(self):
        rng = np.random.default_rng(0)
        ma = MovingAverage(7)
        raw, out = [], []
        for _ in range(200):
            v = 100.0 + rng.normal(0, 5)
            raw.append(v)
            out.append(ma(v, 0.0)[0])
        assert np.std(out[20:]) < np.std(raw[20:]) * 0.6


# --------------------------------------------------------------------------
# One Euro
# --------------------------------------------------------------------------
class TestOneEuro:
    def test_seeds_on_first_sample(self):
        f = OneEuroFilter2D()
        assert f(900.0, 400.0, 1 / 60) == pytest.approx((900.0, 400.0))

    def test_reduces_jitter(self):
        rng = np.random.default_rng(1)
        f = OneEuroFilter2D(min_cutoff=1.0, beta=0.01)
        raw, out = [], []
        for _ in range(300):
            v = 500.0 + rng.normal(0, 4)
            raw.append(v)
            out.append(f(v, 0.0, 1 / 60)[0])
        assert np.std(out[50:]) < np.std(raw[50:]) * 0.5

    def test_beta_trades_lag_for_jitter(self):
        """A higher beta must track a fast ramp with less lag."""
        def lag(beta):
            f = OneEuroFilter2D(min_cutoff=1.0, beta=beta)
            errs = []
            for i in range(120):
                target = 100.0 + i * 8.0
                errs.append(abs(f(target, 0.0, 1 / 60)[0] - target))
            return float(np.mean(errs[40:]))

        assert lag(0.6) < lag(0.001)

    def test_handles_zero_and_negative_dt(self):
        """A stalled clock must not produce NaN or an exception."""
        f = OneEuroFilter2D()
        f(10.0, 10.0, 0.0)
        out = f(20.0, 20.0, -1.0)
        assert all(math.isfinite(v) for v in out)

    def test_set_params_takes_effect(self):
        f = OneEuroFilter2D(min_cutoff=5.0, beta=0.0)
        f.set_params(min_cutoff=0.05)
        assert f.min_cutoff == pytest.approx(0.05)

    def test_reset_clears_state(self):
        f = OneEuroFilter2D()
        for _ in range(30):
            f(100.0, 100.0, 1 / 60)
        f.reset()
        assert f(5.0, 5.0, 1 / 60) == pytest.approx((5.0, 5.0))


# --------------------------------------------------------------------------
# Kalman
# --------------------------------------------------------------------------
class TestKalman:
    def test_tracks_a_ramp(self):
        k = KalmanCV2D(process_noise=1e-2, measurement_noise=1e-2)
        err = []
        for i in range(150):
            target = 50.0 + i * 3.0
            out = k(target, 0.0, 1 / 30)
            err.append(abs(out[0] - target))
        assert np.mean(err[60:]) < 6.0

    def test_reduces_noise(self):
        rng = np.random.default_rng(2)
        k = KalmanCV2D(process_noise=1e-3, measurement_noise=5e-2)
        raw, out = [], []
        for _ in range(300):
            v = 300.0 + rng.normal(0, 6)
            raw.append(v)
            out.append(k(v, 0.0, 1 / 60)[0])
        assert np.std(out[60:]) < np.std(raw[60:]) * 0.6

    def test_first_sample_is_passed_through(self):
        k = KalmanCV2D()
        assert k(777.0, 333.0, 1 / 60) == pytest.approx((777.0, 333.0))

    def test_reset_reinitialises(self):
        k = KalmanCV2D()
        for i in range(40):
            k(float(i), 0.0, 1 / 60)
        k.reset()
        assert k(900.0, 900.0, 1 / 60) == pytest.approx((900.0, 900.0))


# --------------------------------------------------------------------------
# Deadzone
# --------------------------------------------------------------------------
class TestDeadzone:
    def test_holds_below_radius(self):
        dz = Deadzone(radius_px=5.0)
        dz(100.0, 100.0)
        out, moved = dz(102.0, 101.0)
        assert out == (100.0, 100.0)
        assert moved is False

    def test_releases_above_radius(self):
        dz = Deadzone(radius_px=5.0)
        dz(100.0, 100.0)
        out, moved = dz(110.0, 100.0)
        assert out == (110.0, 100.0)
        assert moved is True

    def test_slow_drift_is_never_lost(self):
        """The key property: 1px steps with a 5px deadzone must still arrive.

        Comparing against the last *emitted absolute* position (rather than the
        previous input) is what makes this work; a delta-based deadzone would
        swallow every step and the cursor would never move.
        """
        dz = Deadzone(radius_px=5.0)
        dz(0.0, 0.0)
        out = (0.0, 0.0)
        for i in range(1, 61):
            out, _ = dz(float(i), 0.0)
        assert out[0] == pytest.approx(60.0)

    def test_radius_zero_passes_everything(self):
        dz = Deadzone(radius_px=0.0)
        dz(10.0, 10.0)
        out, moved = dz(10.2, 10.0)
        assert out == pytest.approx((10.2, 10.0))
        assert moved is True

    def test_soft_mode_attenuates_instead_of_freezing(self):
        dz = Deadzone(radius_px=6.0, soft=0.25)
        dz(100.0, 100.0)
        out, moved = dz(104.0, 100.0)
        assert moved is False
        assert 100.0 < out[0] < 104.0
        assert out[0] == pytest.approx(101.0)

    def test_reset_seeds_position(self):
        dz = Deadzone(radius_px=5.0)
        dz.reset(500.0, 500.0)
        out, moved = dz(502.0, 500.0)
        assert out == (500.0, 500.0) and moved is False


# --------------------------------------------------------------------------
# Chain construction
# --------------------------------------------------------------------------
class TestChain:
    @pytest.mark.parametrize("name,expected", [
        ("none", 0),
        ("moving_average", 1),
        ("one_euro", 1),
        ("kalman", 1),
        ("ma_one_euro", 2),
    ])
    def test_build_filter_chain(self, cfg, name, expected):
        set_path(cfg, "gaze.filter", name)
        assert len(build_filter_chain(cfg)) == expected

    def test_default_chain_is_ma_then_one_euro(self, cfg):
        chain = build_filter_chain(cfg)
        assert isinstance(chain[0], MovingAverage)
        assert isinstance(chain[1], OneEuroFilter2D)

    def test_chain_params_come_from_config(self, cfg):
        set_path(cfg, "gaze.moving_average_window", 9)
        set_path(cfg, "gaze.one_euro.beta", 0.5)
        chain = build_filter_chain(cfg)
        assert chain[0].window == 9
        assert chain[1].beta == pytest.approx(0.5)
