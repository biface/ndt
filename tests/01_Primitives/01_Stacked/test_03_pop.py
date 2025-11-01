import re

import pytest

from ndict_tools import StackedKeyError
from ndict_tools.tools import _StackedDict


class TestScenario01StrictSD:

    @pytest.mark.parametrize(
        "pop_item_result",
        [
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                60,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                1,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "nginx",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                30,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                3,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                ["global_settings", "networking", "load_balancer"],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                ["global_settings", "networking"],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
        ],
    )
    def test_pop_item(self, strict_c_sd, pop_item_result):
        assert strict_c_sd.popitem() == pop_item_result

    @pytest.mark.parametrize(
        "keys, pop_result",
        [
            (
                ["global_settings", "security"],
                _StackedDict(
                    {"encryption": "mandatory", "level": 100},
                    default_setup={"indent": 2, "default_factory": None},
                ),
            ),
            (
                "global_settings",
                _StackedDict(
                    {
                        ("security", "encryption"): {
                            "algorithm": "AES-256-GCM",
                            "key_rotation": {
                                ("env", "production"): 90,
                                ("env", "dev"): 365,
                            },
                        }
                    },
                    default_setup={"indent": 2, "default_factory": None},
                ),
            ),
        ],
    )
    def test_pop(self, strict_c_sd, keys, pop_result):
        assert strict_c_sd.pop(keys) == pop_result

    @pytest.mark.parametrize(
        "keys, error_smg",
        [
            (
                ["global_settings", "security"],
                "Key path ['global_settings', 'security'] does not exist. (key: ['global_settings', 'security'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'])",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'])",
            ),
        ],
    )
    def test_pop_failed(self, strict_c_sd, keys, error_smg):
        with pytest.raises(StackedKeyError, match=re.escape(error_smg)):
            strict_c_sd.pop(keys)


class TestScenario01SmoothSD:

    @pytest.mark.parametrize(
        "pop_item_result",
        [
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                60,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                1,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "nginx",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                30,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                3,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                ["global_settings", "networking", "load_balancer"],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                ["global_settings", "networking"],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
        ],
    )
    def test_pop_item(self, smooth_c_sd, pop_item_result):
        assert smooth_c_sd.popitem() == pop_item_result

    @pytest.mark.parametrize(
        "keys, pop_result",
        [
            (
                ["global_settings", "security"],
                _StackedDict(
                    {"encryption": "mandatory", "level": 100},
                    default_setup={"indent": 2, "default_factory": _StackedDict},
                ),
            ),
            (
                "global_settings",
                _StackedDict(
                    {
                        ("security", "encryption"): {
                            "algorithm": "AES-256-GCM",
                            "key_rotation": {
                                ("env", "production"): 90,
                                ("env", "dev"): 365,
                            },
                        }
                    },
                    default_setup={"indent": 2, "default_factory": _StackedDict},
                ),
            ),
        ],
    )
    def test_pop(self, smooth_c_sd, keys, pop_result):
        assert smooth_c_sd.pop(keys) == pop_result

    @pytest.mark.parametrize(
        "keys, error_smg",
        [
            (
                ["global_settings", "security"],
                "Key path ['global_settings', 'security'] does not exist. (key: ['global_settings', 'security'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'])",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'])",
            ),
        ],
    )
    def test_pop_failed(self, smooth_c_sd, keys, error_smg):
        with pytest.raises(StackedKeyError, match=re.escape(error_smg)):
            smooth_c_sd.pop(keys)


class TestPopFunctionExtrasStrict:

    def test_pop_flat_key_success(self, strict_f_sd):
        val = strict_f_sd.pop("monitoring")
        assert isinstance(val, _StackedDict)
        # Check a known leaf still present in the returned subtree
        assert ("metrics", "cpu") in val
        # Ensure key removed from top-level
        assert "monitoring" not in strict_f_sd

    def test_pop_flat_key_missing_with_default(self, strict_f_sd):
        sentinel = object()
        assert strict_f_sd.pop("__does_not_exist__", sentinel) is sentinel

    def test_pop_flat_key_missing_raises_key_error(self, strict_f_sd):
        with pytest.raises(
            StackedKeyError,
            match=re.escape(
                "Key path ['global_settings', 'does_not'] does not exist. (key: does_not)"
            ),
        ):
            strict_f_sd.pop(["global_settings", "does_not"])

    def test_pop_hier_intermediate_missing_with_default(self, strict_f_sd):
        # Intermediate key is missing -> should return default, no exception
        assert strict_f_sd.pop(["global_settings", "__nok__", "x"], default=42) == 42

    def test_pop_hier_final_missing_with_default(self, strict_f_sd):
        # Path exists up to parent, final key missing -> should return default
        path = [
            "global_settings",
            "networking",
            "load_balancer",
            ("env", "dev"),
            "__missing_leaf__",
        ]
        assert strict_f_sd.pop(path, default="x") == "x"

    def test_cleanup_empty_parents(self, standard_strict_f_setup):
        sd = _StackedDict({"a": {"b": {"c": 1}}}, default_setup=standard_strict_f_setup)
        assert sd.pop(["a", "b", "c"]) == 1
        # After removing the only chain, empty parents should be cleaned up
        assert "a" not in sd

    def test_cleanup_preserve_non_empty(self, standard_strict_f_setup):
        sd = _StackedDict(
            {"a": {"b": {"c": 1}, "d": 2}}, default_setup=standard_strict_f_setup
        )
        assert sd.pop(["a", "b", "c"]) == 1
        # "b" was emptied and must be removed, but "a" still has "d"
        assert "a" in sd
        assert "b" not in sd["a"]
        assert sd["a"]["d"] == 2


class TestPopFunctionExtrasSmooth:

    def test_pop_flat_key_success(self, smooth_f_sd):
        val = smooth_f_sd.pop("monitoring")
        assert isinstance(val, _StackedDict)
        assert ("metrics", "cpu") in val
        assert "monitoring" not in smooth_f_sd

    def test_pop_flat_key_missing_with_default(self, smooth_f_sd):
        sentinel = object()
        assert smooth_f_sd.pop("__does_not_exist__", sentinel) is sentinel

    def test_pop_flat_key_missing_raises_keyerror(self, smooth_f_sd):
        with pytest.raises(
            StackedKeyError,
            match=re.escape(
                "Key path ['global_settings', 'does_not'] does not exist. (key: does_not)"
            ),
        ):
            smooth_f_sd.pop(["global_settings", "does_not"])

    def test_pop_hier_intermediate_missing_with_default(self, smooth_f_sd):
        assert smooth_f_sd.pop(["global_settings", "__nok__", "x"], default=42) == 42

    def test_pop_hier_final_missing_with_default(self, smooth_f_sd):
        path = [
            "global_settings",
            "networking",
            "load_balancer",
            ("env", "dev"),
            "__missing_leaf__",
        ]
        assert smooth_f_sd.pop(path, default="x") == "x"

    def test_cleanup_empty_parents(self, standard_smooth_f_setup):
        sd = _StackedDict({"a": {"b": {"c": 1}}}, default_setup=standard_smooth_f_setup)
        assert sd.pop(["a", "b", "c"]) == 1
        assert "a" not in sd

    def test_cleanup_preserve_non_empty(self, standard_smooth_f_setup):
        sd = _StackedDict(
            {"a": {"b": {"c": 1}, "d": 2}}, default_setup=standard_smooth_f_setup
        )
        assert sd.pop(["a", "b", "c"]) == 1
        assert "a" in sd
        assert "b" not in sd["a"]
        assert sd["a"]["d"] == 2
