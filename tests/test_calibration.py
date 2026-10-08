"""Calibration: grid layout, polynomial fit, persistence, session logic.

The fit is checked against mappings with a known closed form, so a regression
in the design matrix or the standardization shows up as an accuracy loss
rather than a crash.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import calibration as calib
from config import get
from util import Clock


def rows_for(fn, grid, screen=(1600, 900), head=(0.5, 0.42)):
    """Build [gx,gy,hx,hy,sx,sy] rows from a known feature->screen function."""
    out = []
    for sx, sy in grid:
        gx, gy = fn(sx, sy, screen)
        out.append([gx, gy, head[0], head[1], float(sx), float(sy)])
    return out


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
class TestGrid:
    def test_nine_points_unique_and_inset(self):
        g = calib.calibration_grid(9, 0.12, 1600, 900)
        assert len(g) == 9
        assert len(set(g)) == 9
        for x, y in g:
            assert 0.12 * 1600 - 1 <= x <= 0.88 * 1600 + 1
            assert 0.12 * 900 - 1 <= y <= 0.88 * 900 + 1

    def test_nine_points_form_a_3x3(self):
        g = calib.calibration_grid(9, 0.1, 1000, 1000)
        xs = sorted({x for x, _ in g})
        ys = sorted({y for _, y in g})
        assert len(xs) == 3 and len(ys) == 3

    def test_five_points_are_corners_plus_centre(self):
        g = calib.calibration_grid(5, 0.1, 1000, 1000)
        assert len(g) == 5
        assert (500, 500) in g

    def test_thirteen_points_unique(self):
        g = calib.calibration_grid(13, 0.1, 1600, 900)
        assert len(g) == 13 and len(set(g)) == 13

    def test_zero_inset_still_inside_screen(self):
        g = calib.calibration_grid(9, 0.0, 1920, 1080)
        for x, y in g:
            assert 0 <= x <= 1920 and 0 <= y <= 1080

    def test_unsupported_count_falls_back_to_a_grid(self):
        g = calib.calibration_grid(7, 0.1, 1600, 900)
        assert len(g) == 7 and len(set(g)) == 7

    def test_inset_keeps_points_clear_of_the_scroll_band(self, cfg):
        band = float(get(cfg, "scroll.edge.band_ratio"))
        inset = float(get(cfg, "calibration.inset_ratio"))
        assert inset > band, "calibration points must not sit inside the scroll band"
        g = calib.calibration_grid(9, inset, 1600, 900)
        assert min(y for _, y in g) > band * 900


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
class TestFit:
    def test_recovers_a_linear_mapping(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)

        def fn(sx, sy, screen):
            w, h = screen
            return (sx - w / 2) / (0.4 * w), (sy - h / 2) / (0.4 * h)

        m = calib.fit_calibration(rows_for(fn, grid), model="quadratic",
                                  screen_w=1600, screen_h=900, points=grid)
        assert m.rms_px < 1.0, m.report()
        for sx, sy in grid:
            px, py = m.apply_px(*fn(sx, sy, (1600, 900)))
            assert math.dist((px, py), (sx, sy)) < 3.0

    def test_recovers_a_nonlinear_mapping(self):
        """A quadratic model should absorb a genuinely curved relationship."""
        grid = calib.calibration_grid(9, 0.12, 1600, 900)

        def fn(sx, sy, screen):
            w, h = screen
            gx = (sx - w / 2) / (0.4 * w)
            gy = (sy - h / 2) / (0.4 * h)
            return gx + 0.12 * gx * gx, gy - 0.08 * gy * gy

        m = calib.fit_calibration(rows_for(fn, grid), model="quadratic",
                                  screen_w=1600, screen_h=900, points=grid)
        assert m.rms_px < 25.0, m.report()

    def test_linear_model_is_worse_than_quadratic_on_curved_data(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)

        def fn(sx, sy, screen):
            w, h = screen
            gx = (sx - w / 2) / (0.4 * w)
            gy = (sy - h / 2) / (0.4 * h)
            return gx + 0.35 * gx * gx, gy - 0.3 * gy * gy

        rows = rows_for(fn, grid)
        lin = calib.fit_calibration(rows, model="linear", screen_w=1600, screen_h=900)
        quad = calib.fit_calibration(rows, model="quadratic", screen_w=1600, screen_h=900)
        assert quad.rms_px < lin.rms_px

    def test_auto_downgrades_when_underdetermined(self):
        """3 rows cannot fit 8 coefficients; the model must step down."""
        rows = [[0.0, 0.0, 0.5, 0.5, 800, 450],
                [0.2, 0.0, 0.5, 0.5, 1100, 450],
                [0.0, 0.2, 0.5, 0.5, 800, 650]]
        m = calib.fit_calibration(rows, model="quadratic_head", screen_w=1600, screen_h=900)
        assert m.model == "linear"
        assert len(m.coef_x) == 3

    def test_quadratic_needs_six_distinct_points(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)[:6]

        def fn(sx, sy, screen):
            return sx / 1600.0, sy / 900.0

        m = calib.fit_calibration(rows_for(fn, grid), model="quadratic",
                                  screen_w=1600, screen_h=900)
        assert m.model == "quadratic"

    def test_rejects_too_few_rows(self):
        with pytest.raises(ValueError):
            calib.fit_calibration([[0, 0, 0.5, 0.5, 100, 100]], screen_w=1600, screen_h=900)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError):
            calib.fit_calibration([[0, 0, 0.5]], screen_w=1600, screen_h=900)

    def test_zero_variance_feature_does_not_produce_nan(self):
        """A user who never moves their head gives hx/hy zero variance."""
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        m = calib.fit_calibration(rows, model="quadratic_head", screen_w=1600, screen_h=900)
        assert np.all(np.isfinite(m.coef_x)) and np.all(np.isfinite(m.std))
        nx, ny = m.apply(0.0, 0.0, 0.5, 0.5)
        assert math.isfinite(nx) and math.isfinite(ny)

    def test_output_is_always_clamped_to_the_screen(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        m = calib.fit_calibration(rows, model="quadratic", screen_w=1600, screen_h=900)
        for gx in (-50.0, -5.0, 5.0, 50.0):
            for gy in (-50.0, 0.0, 50.0):
                nx, ny = m.apply(gx, gy, 0.5, 0.5)
                assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0

    def test_nonfinite_input_does_not_crash(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        m = calib.fit_calibration(rows, model="quadratic", screen_w=1600, screen_h=900)
        nx, ny = m.apply(float("nan"), float("inf"), 0.5, 0.5)
        assert math.isfinite(nx) and math.isfinite(ny)

    def test_per_point_error_is_recorded(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        m = calib.fit_calibration(rows, model="quadratic", screen_w=1600, screen_h=900,
                                  points=grid)
        assert len(m.per_point_px) == 9
        assert m.max_px == pytest.approx(max(m.per_point_px), abs=1e-6)

    def test_unknown_model_name_defaults_to_quadratic(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        m = calib.fit_calibration(rows, model="cubic_magic", screen_w=1600, screen_h=900)
        assert m.model in calib.TERM_SETS

    def test_report_grades_accuracy(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        good = calib.fit_calibration(rows, model="quadratic", screen_w=1600, screen_h=900,
                                     points=grid)
        assert "excellent" in good.report() or "good" in good.report()
        # Inject noise so the fit is genuinely poor and the advice appears.
        rng = np.random.default_rng(0)
        noisy = [[r[0] + rng.normal(0, 0.9), r[1] + rng.normal(0, 0.9),
                  r[2], r[3], r[4], r[5]] for r in rows]
        bad = calib.fit_calibration(noisy, model="quadratic", screen_w=1600, screen_h=900,
                                    points=grid)
        assert "lighting" in bad.report().lower()


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
class TestPersistence:
    def _model(self):
        grid = calib.calibration_grid(9, 0.12, 1600, 900)
        rows = [[(sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5, float(sx), float(sy)]
                for sx, sy in grid]
        return calib.fit_calibration(rows, model="quadratic", screen_w=1600,
                                     screen_h=900, points=grid)

    def test_round_trip_is_lossless(self, tmp_path):
        m = self._model()
        p = str(tmp_path / "cal.json")
        calib.save_calibration(m, p)
        back = calib.load_calibration(p)
        assert back is not None
        assert back.model == m.model
        assert np.allclose(back.coef_x, m.coef_x)
        assert np.allclose(back.coef_y, m.coef_y)
        assert np.allclose(back.mean, m.mean)
        assert np.allclose(back.std, m.std)
        assert back.points == m.points
        assert back.rms_px == pytest.approx(m.rms_px)
        for gx in (-0.5, 0.0, 0.5):
            assert m.apply(gx, 0.1) == pytest.approx(back.apply(gx, 0.1))

    def test_saved_file_is_valid_json(self, tmp_path):
        p = str(tmp_path / "cal.json")
        calib.save_calibration(self._model(), p)
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["version"] == calib.CALIBRATION_VERSION
        assert len(data["coef_x"]) == 6

    def test_creates_missing_parent_directory(self, tmp_path):
        p = str(tmp_path / "nested" / "dir" / "cal.json")
        calib.save_calibration(self._model(), p)
        assert os.path.exists(p)

    def test_load_missing_file_returns_none(self):
        assert calib.load_calibration("/no/such/calibration.json") is None

    def test_load_corrupt_file_returns_none(self, tmp_path, capsys):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert calib.load_calibration(str(p)) is None
        assert "ignoring unreadable" in capsys.readouterr().out

    def test_truncated_coefficients_are_padded(self):
        m = calib.CalibrationModel.from_dict({
            "version": calib.CALIBRATION_VERSION, "model": "quadratic",
            "mean": [0, 0, 0.5, 0.5], "std": [1, 1, 1, 1],
            "coef_x": [0.5, 0.2], "coef_y": [0.5],
            "screen_w": 1600, "screen_h": 900,
        })
        assert len(m.coef_x) == 6 and len(m.coef_y) == 6
        nx, ny = m.apply(0.0, 0.0)
        assert 0.0 <= nx <= 1.0

    def test_newer_version_is_rejected(self):
        with pytest.raises(ValueError, match="newer than this build"):
            calib.CalibrationModel.from_dict({"version": calib.CALIBRATION_VERSION + 99})

    def test_zero_std_is_repaired(self):
        m = calib.CalibrationModel.from_dict({
            "version": calib.CALIBRATION_VERSION, "model": "quadratic",
            "mean": [0, 0, 0.5, 0.5], "std": [0.0, 0.0, 0.0, 0.0],
            "coef_x": [0.5, 0, 0, 0, 0, 0], "coef_y": [0.5, 0, 0, 0, 0, 0],
            "screen_w": 1600, "screen_h": 900,
        })
        assert np.all(m.std > 0)
        assert math.isfinite(m.apply(0.1, 0.1)[0])


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
class TestSession:
    def _grid(self):
        return calib.calibration_grid(9, 0.12, 1600, 900)

    def test_requires_at_least_three_points(self):
        with pytest.raises(ValueError):
            calib.CalibrationSession([(1, 1), (2, 2)])

    def test_discards_the_first_n_samples(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=10,
                                     discard_first=4, clock=clk)
        clk.advance(1000)                       # clear the settle window
        # 4 discarded + 6 kept; the kept ones are all the same value, so the
        # discarded outliers must not influence the result.
        for _ in range(4):
            s.add_sample(99.0, 99.0, 99.0, 99.0)
        for _ in range(6):
            s.add_sample(0.1, 0.2, 0.5, 0.5)  # 10 total -> point finalised
        assert s.index == 1
        obs = s.observations()[0][0]
        assert obs[0] == pytest.approx(0.1, abs=1e-9)

    def test_settle_window_ignores_early_samples(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=5,
                                     discard_first=0, clock=clk, settle_ms=300)
        assert s.settling is True
        assert s.add_sample(1, 1, 1, 1) is False
        clk.advance(400)
        assert s.settling is False
        s.add_sample(0.1, 0.1, 0.5, 0.5)
        assert len(s.observations()) == 0 or s.index == 0

    def test_outlier_samples_are_rejected(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=20,
                                     discard_first=2, clock=clk)
        clk.advance(1000)
        for _ in range(18):
            s.add_sample(0.30, 0.30, 0.5, 0.5)
        # One wild frame -- a blink or a tracking glitch.
        s.add_sample(5.0, -5.0, 0.5, 0.5)
        s.add_sample(0.30, 0.30, 0.5, 0.5)   # 20th sample finalises the point
        obs = s.observations()[0][0]
        assert obs[0] == pytest.approx(0.30, abs=0.02)
        assert any("outlier" in h for h in s.history)

    def test_progress_and_completion(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=6,
                                     discard_first=1, clock=clk)
        clk.advance(1000)
        assert s.complete is False
        assert s.progress == pytest.approx(0.0)
        for pt in range(9):
            clk.advance(400)          # each point has its own settle window
            for i in range(6):
                s.add_sample(0.1 * pt, 0.0, 0.5, 0.5)
            assert s.progress == pytest.approx((pt + 1) / 9, abs=1e-6)
        assert s.complete is True
        assert len(s.observations()) == 9

    def test_no_samples_after_completion(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=5,
                                     discard_first=0, clock=clk)
        clk.advance(1000)
        for i in range(60):
            if i % 5 == 0:
                clk.advance(300)      # clear each point's settle window
            s.add_sample(0.1, 0.1, 0.5, 0.5)
        assert s.complete
        assert s.add_sample(9.0, 9.0, 9.0, 9.0) is False

    def test_skip_point_abandons_the_target(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=5,
                                     discard_first=0, clock=clk)
        clk.advance(1000)
        s.add_sample(0.1, 0.1, 0.5, 0.5)
        assert s.skip_point() is False      # moved to point 2, not complete
        assert s.index == 1
        assert len(s.observations()) == 0   # the abandoned point is not kept

    def test_nonfinite_samples_ignored(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=5,
                                     discard_first=0, clock=clk)
        clk.advance(1000)
        assert s.add_sample(float("nan"), 0.1, 0.5, 0.5) is False
        assert s.add_sample(0.1, float("inf"), 0.5, 0.5) is False

    def test_build_raises_when_too_few_points(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=5,
                                     discard_first=0, clock=clk)
        clk.advance(1000)
        with pytest.raises(ValueError, match="at least 3"):
            s.build()

    def test_build_produces_a_usable_model(self):
        clk = Clock(manual=True)
        grid = self._grid()
        s = calib.CalibrationSession(grid, samples_per_point=8, discard_first=2, clock=clk)
        clk.advance(1000)
        for (sx, sy) in grid:
            clk.advance(400)          # clear each point's settle window
            for _ in range(8):
                s.add_sample((sx - 800) / 400.0, (sy - 450) / 400.0, 0.5, 0.5)
        m = s.build(model="quadratic", screen_w=1600, screen_h=900)
        assert m.rms_px < 2.0, m.report()
        assert len(m.samples_per_point) == 9
        assert m.samples_per_point[0] == 6      # 8 collected - 2 discarded

    def test_point_progress_ramps(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(self._grid(), samples_per_point=10,
                                     discard_first=2, clock=clk)
        clk.advance(1000)
        assert s.point_progress() == 0.0
        for _ in range(4):
            s.add_sample(0.1, 0.1, 0.5, 0.5)
        # 4 collected, first 2 discarded, 8 needed -> 2/8.
        assert s.point_progress() == pytest.approx(0.25)


class TestRendering:
    def test_draw_calibration_screen_places_the_dot_exactly(self):
        cv2 = pytest.importorskip("cv2")
        clk = Clock(manual=True)
        grid = calib.calibration_grid(9, 0.12, 800, 600)
        s = calib.CalibrationSession(grid, clock=clk, screen_w=800, screen_h=600)
        canvas = calib.draw_calibration_screen(s, radius=10, canvas_w=800, canvas_h=600)
        assert canvas is not None and canvas.shape[:2] == (600, 800)

        # Locate the active dot by its orange fill (BGR 0,200,255) and check
        # its centroid lands on the exact target pixel. Sampling a fixed patch
        # around the centre is not enough: the dot pulses, so the dark pupil
        # drawn on top of it grows and shrinks with wall-clock time and can
        # cover the centre entirely.
        orange = ((canvas[:, :, 0] < 90) & (canvas[:, :, 1] > 150)
                  & (canvas[:, :, 2] > 200))
        assert orange.any(), "the active target dot was not drawn"
        ys, xs = np.nonzero(orange)
        centroid = (float(xs.mean()), float(ys.mean()))
        assert centroid[0] == pytest.approx(grid[0][0], abs=1.0)
        assert centroid[1] == pytest.approx(grid[0][1], abs=1.0)

    def test_dot_moves_to_each_target_in_order(self):
        cv2 = pytest.importorskip("cv2")
        clk = Clock(manual=True)
        grid = calib.calibration_grid(9, 0.12, 800, 600)
        s = calib.CalibrationSession(grid, samples_per_point=5, discard_first=0,
                                     clock=clk, screen_w=800, screen_h=600)
        seen = []
        for _ in range(3):
            clk.advance(400)
            canvas = calib.draw_calibration_screen(s, radius=10, canvas_w=800, canvas_h=600)
            orange = ((canvas[:, :, 0] < 90) & (canvas[:, :, 1] > 150)
                      & (canvas[:, :, 2] > 200))
            ys, xs = np.nonzero(orange)
            seen.append((int(round(xs.mean())), int(round(ys.mean()))))
            for _ in range(5):
                s.add_sample(0.1, 0.1, 0.5, 0.5)
        assert seen[0] != seen[1] != seen[2], "the dot must advance between targets"
        for drawn, target in zip(seen, grid[:3]):
            assert abs(drawn[0] - target[0]) <= 1 and abs(drawn[1] - target[1]) <= 1

    def test_thumbnail_is_inset(self):
        cv2 = pytest.importorskip("cv2")
        clk = Clock(manual=True)
        s = calib.CalibrationSession(calib.calibration_grid(9, 0.12, 800, 600),
                                     clock=clk, screen_w=800, screen_h=600)
        thumb = np.zeros((120, 160, 3), np.uint8)
        canvas = calib.draw_calibration_screen(s, thumb=thumb, canvas_w=800, canvas_h=600)
        assert canvas is not None and canvas.shape[:2] == (600, 800)

    def test_is_point_done(self):
        clk = Clock(manual=True)
        s = calib.CalibrationSession(calib.calibration_grid(9, 0.12, 800, 600),
                                     samples_per_point=5, discard_first=0, clock=clk)
        clk.advance(1000)
        assert s.is_point_done(0) is False
        for _ in range(5):
            s.add_sample(0.1, 0.1, 0.5, 0.5)
        assert s.is_point_done(0) is True
        assert s.is_point_done(99) is False
