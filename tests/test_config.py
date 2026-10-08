"""Config loading, validation, clamping and persistence."""

from __future__ import annotations

import copy
import json
import os

import pytest

import config as C


NOFILE = "/definitely/not/here/config.json"


class TestDefaults:
    def test_missing_file_yields_full_defaults(self):
        cfg = C.load_config(NOFILE)
        for key, value in C.DEFAULTS.items():
            assert key in cfg
            if isinstance(value, dict):
                for sub in value:
                    assert sub in cfg[key]

    def test_defaults_are_not_shared_mutable_state(self):
        """Mutating one loaded config must not corrupt DEFAULTS."""
        cfg = C.load_config(NOFILE)
        cfg["gaze"]["gain"] = 99.0
        cfg["clicks"]["temple"]["radius_ratio"] = 0.001
        fresh = C.load_config(NOFILE)
        assert fresh["gaze"]["gain"] == C.DEFAULTS["gaze"]["gain"]
        assert C.DEFAULTS["gaze"]["gain"] != 99.0

    def test_defaults_are_self_consistent(self):
        """Shipped defaults must pass validation without a single change."""
        cfg = copy.deepcopy(C.DEFAULTS)
        notes = C.validate(cfg)
        assert notes == [], f"shipped defaults are invalid: {notes}"

    def test_default_choices_are_all_legal(self):
        for path, allowed in C.CHOICES.items():
            value = C.get(C.DEFAULTS, path)
            assert str(value).lower() in {str(a).lower() for a in allowed}, path

    def test_default_values_are_inside_their_ranges(self):
        for path, (lo, hi) in C.RANGES.items():
            value = C.get(C.DEFAULTS, path)
            assert value is not None, f"{path} missing from DEFAULTS"
            assert lo <= float(value) <= hi, f"{path}={value} outside {lo}..{hi}"


class TestAccessors:
    def test_get_dotted_path(self):
        cfg = C.load_config(NOFILE)
        assert C.get(cfg, "gaze.one_euro.beta") == pytest.approx(0.012)

    def test_get_missing_returns_default(self):
        assert C.get({}, "a.b.c", "fallback") == "fallback"

    def test_get_through_a_non_dict(self):
        assert C.get({"a": 5}, "a.b", "fallback") == "fallback"

    def test_set_path_creates_intermediates(self):
        cfg = {}
        C.set_path(cfg, "x.y.z", 7)
        assert cfg == {"x": {"y": {"z": 7}}}

    def test_set_path_overwrites_a_scalar(self):
        cfg = {"x": 5}
        C.set_path(cfg, "x.y", 7)
        assert cfg == {"x": {"y": 7}}

    def test_round_trip(self):
        cfg = {"a": {"b": 1}}
        C.set_path(cfg, "a.b", 2)
        assert C.get(cfg, "a.b") == 2


class TestMerge:
    def test_partial_user_file_keeps_other_defaults(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"gaze": {"gain": 2.5}}), encoding="utf-8")
        cfg = C.load_config(str(p))
        assert cfg["gaze"]["gain"] == 2.5
        # Siblings of the overridden key survive.
        assert cfg["gaze"]["deadzone_px"] == C.DEFAULTS["gaze"]["deadzone_px"]
        assert cfg["clicks"]["debounce_ms"] == C.DEFAULTS["clicks"]["debounce_ms"]

    def test_deep_merge_does_not_mutate_base(self):
        base = {"a": {"b": 1, "c": 2}}
        out = C.deep_merge(base, {"a": {"b": 9}})
        assert base["a"]["b"] == 1
        assert out == {"a": {"b": 9, "c": 2}}

    def test_unknown_keys_are_reported_not_fatal(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"gaze": {"nonexistent": 1}, "bogus": {}}), encoding="utf-8")
        cfg = C.load_config(str(p))
        notes = cfg["_meta"]["notes"]
        assert any("gaze.nonexistent" in n for n in notes)
        assert any("bogus" in n for n in notes)


class TestValidation:
    def test_out_of_range_values_are_clamped(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["gaze"]["gain"] = 999.0
        cfg["clicks"]["debounce_ms"] = -50
        cfg["scroll"]["edge"]["band_ratio"] = 5.0
        C.validate(cfg)
        assert cfg["gaze"]["gain"] == C.RANGES["gaze.gain"][1]
        assert cfg["clicks"]["debounce_ms"] == 0
        assert cfg["scroll"]["edge"]["band_ratio"] == C.RANGES["scroll.edge.band_ratio"][1]

    def test_clamping_is_reported(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["gaze"]["gain"] = 999.0
        notes = C.validate(cfg)
        assert any("clamped" in n for n in notes)

    def test_invalid_choice_falls_back_to_default(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["gaze"]["filter"] = "banana"
        notes = C.validate(cfg)
        assert cfg["gaze"]["filter"] == C.DEFAULTS["gaze"]["filter"]
        assert any("banana" in n for n in notes)

    def test_choice_case_is_normalised(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["gaze"]["filter"] = "One_Euro"
        C.validate(cfg)
        assert cfg["gaze"]["filter"] == "one_euro"

    def test_point_count_accepts_the_supported_values(self):
        for n in (5, 9, 13):
            cfg = copy.deepcopy(C.DEFAULTS)
            cfg["calibration"]["points"] = n
            C.validate(cfg)
            assert cfg["calibration"]["points"] == n

    def test_string_numbers_are_coerced(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["camera"]["index"] = "1"
        cfg["gaze"]["gain"] = "2.0"
        notes = C.validate(cfg)
        assert cfg["camera"]["index"] == 1
        assert isinstance(cfg["camera"]["index"], int)
        assert cfg["gaze"]["gain"] == 2.0

    def test_string_booleans_are_coerced(self):
        for truthy in ("true", "TRUE", "1", "yes", "on"):
            cfg = copy.deepcopy(C.DEFAULTS)
            cfg["overlay"]["show"] = truthy
            C.validate(cfg)
            assert cfg["overlay"]["show"] is True, truthy
        for falsy in ("false", "0", "no", "off"):
            cfg = copy.deepcopy(C.DEFAULTS)
            cfg["overlay"]["show"] = falsy
            C.validate(cfg)
            assert cfg["overlay"]["show"] is False, falsy

    def test_non_numeric_value_restores_default(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["gaze"]["gain"] = "lots"
        C.validate(cfg)
        assert cfg["gaze"]["gain"] == C.DEFAULTS["gaze"]["gain"]

    def test_booleans_are_never_clamped_as_numbers(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["overlay"]["show"] = True
        C.validate(cfg)
        assert cfg["overlay"]["show"] is True


class TestCrossField:
    def test_scroll_min_cannot_exceed_max(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["scroll"]["edge"]["min_amount"] = 20
        cfg["scroll"]["edge"]["max_amount"] = 5
        notes = C.validate(cfg)
        assert cfg["scroll"]["edge"]["max_amount"] >= cfg["scroll"]["edge"]["min_amount"]
        assert any("max_amount" in n for n in notes)

    def test_right_dwell_must_exceed_left_dwell(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["clicks"]["dwell"]["ms"] = 1500
        cfg["clicks"]["dwell"]["right_click_ms"] = 900
        C.validate(cfg)
        assert cfg["clicks"]["dwell"]["right_click_ms"] > 1500

    def test_calibration_inset_clears_the_scroll_band(self):
        """Otherwise the outer calibration dots would trigger scrolling."""
        cfg = copy.deepcopy(C.DEFAULTS)
        cfg["scroll"]["edge"]["band_ratio"] = 0.20
        cfg["calibration"]["inset_ratio"] = 0.05
        C.validate(cfg)
        assert cfg["calibration"]["inset_ratio"] > cfg["scroll"]["edge"]["band_ratio"]

    def test_default_inset_already_clears_the_default_band(self):
        cfg = copy.deepcopy(C.DEFAULTS)
        assert (cfg["calibration"]["inset_ratio"]
                > cfg["scroll"]["edge"]["band_ratio"])


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        p = str(tmp_path / "c.json")
        cfg = C.load_config(NOFILE)
        cfg["gaze"]["gain"] = 3.0
        C.save_config(cfg, p)
        back = C.load_config(p)
        assert back["gaze"]["gain"] == 3.0

    def test_saved_file_excludes_internal_meta(self, tmp_path):
        p = str(tmp_path / "c.json")
        cfg = C.load_config(NOFILE)
        C.save_config(cfg, p)
        data = json.loads(open(p, encoding="utf-8").read())
        assert "_meta" not in data
        assert not any(k.startswith("_") for k in data)

    def test_saved_file_is_indented_and_sorted_stably(self, tmp_path):
        p = str(tmp_path / "c.json")
        C.save_config(C.load_config(NOFILE), p)
        text = open(p, encoding="utf-8").read()
        assert "\n  " in text and text.endswith("\n")

    def test_save_is_atomic(self, tmp_path):
        """A crash mid-write must not leave a truncated config behind."""
        p = str(tmp_path / "c.json")
        C.save_config(C.load_config(NOFILE), p)
        assert not os.path.exists(p + ".tmp")

    def test_write_default_config_then_load_is_clean(self, tmp_path):
        p = str(tmp_path / "c.json")
        C.write_default_config(p)
        cfg = C.load_config(p)
        notes = [n for n in cfg["_meta"]["notes"]
                 if "clamped" in n or "not in" in n or "unknown" in n]
        assert notes == []

    def test_invalid_json_raises_config_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{oops", encoding="utf-8")
        with pytest.raises(C.ConfigError, match="not valid JSON"):
            C.load_config(str(p))

    def test_non_object_json_raises(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(C.ConfigError, match="JSON object"):
            C.load_config(str(p))

    def test_config_path_is_recorded(self):
        cfg = C.load_config(NOFILE)
        assert cfg["_meta"]["path"] == NOFILE
        assert cfg["_meta"]["existed"] is False
