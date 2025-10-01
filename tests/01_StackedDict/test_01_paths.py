import re

import pytest

from ndict_tools.exception import StackedTypeError
from ndict_tools.tools import _StackedDict


@pytest.mark.parametrize(
    "path, leaf",
    [
        ([("env", "production"), "database", "port"], 5432),
        ([frozenset(["cache", "redis"]), "config", "memory"], "2GB"),
        (["monitoring", ("metrics", "cpu")], [80, 90, 95]),
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
    ],
)
def test_stacked_dict_path(strict_f_sd, path, leaf):
    strict_f_sd[path] == leaf


class TestStrictStackedDictPath:

    @pytest.mark.parametrize(
        "path, old_value, new_value",
        [
            ([("env", "production"), "database", "port"], 5432, 5342),
            ([frozenset(["cache", "redis"]), "config", "memory"], "2GB", "4GB"),
            (["monitoring", ("metrics", "cpu")], [80, 90, 95], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
                "AZURE US",
            ),
        ],
    )
    def test_change_value(self, strict_c_sd, path, old_value, new_value):
        assert strict_c_sd[path] == old_value
        strict_c_sd[path] = new_value
        assert strict_c_sd[path] != old_value

    @pytest.mark.parametrize(
        "path, leaf",
        [
            ([("env", "production"), "database", "port"], 5342),
            ([frozenset(["cache", "redis"]), "config", "memory"], "4GB"),
            (["monitoring", ("metrics", "cpu")], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AZURE US",
            ),
        ],
    )
    def test_change_control(self, strict_c_sd, path, leaf):
        assert strict_c_sd[path] == leaf

    @pytest.mark.parametrize(
        "a_path, b_path, v_type",
        [
            (
                ["global_settings", ("security", "encryption")],
                ["global_settings", "security", "encryption"],
                _StackedDict,
            ),
            (
                ["global_settings", "security", "encryption"],
                ["global_settings", ("security", "encryption")],
                str,
            ),
        ],
    )
    def test_hierarchical(self, strict_c_sd, a_path, b_path, v_type):
        assert strict_c_sd[a_path] != strict_c_sd[b_path]
        assert isinstance(strict_c_sd[a_path], v_type)

    @pytest.mark.parametrize(
        "keys, false_end_key, error, error_msg",
        [
            ([("env", "production"), "database"], "ports", KeyError, "'ports'"),
            ([frozenset(["cache", "redis"]), "config"], "me_ory", KeyError, "'me_ory'"),
            (["monitoring"], ("metrics", "cpus"), KeyError, "('metrics', 'cpus')"),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "types",
                KeyError,
                "'types'",
            ),
        ],
    )
    def test_change_paths_failed(
        self, strict_c_sd, keys, false_end_key, error, error_msg
    ):
        d = strict_c_sd
        for key in keys:
            d = d[key]
        assert isinstance(d, _StackedDict)
        assert d.default_factory == None
        with pytest.raises(error, match=re.escape(error_msg)):
            test = d[false_end_key]

    @pytest.mark.parametrize(
        "false_keys_type, error, error_msg",
        [
            ({1, 2}, TypeError, "unhashable type: 'set'"),
            (
                [1, [1, 2]],
                StackedTypeError,
                "Nested lists are not allowed as keys in _StackedDict. (expected: str, got: list)",
            ),
        ],
    )
    def test_paths_type_failed(self, strict_c_sd, false_keys_type, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            strict_c_sd[false_keys_type] = None


class TestSmoothStackedDictPath:

    @pytest.mark.parametrize(
        "path, old_value, new_value",
        [
            ([("env", "production"), "database", "port"], 5432, 5342),
            ([frozenset(["cache", "redis"]), "config", "memory"], "2GB", "4GB"),
            (["monitoring", ("metrics", "cpu")], [80, 90, 95], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
                "AZURE US",
            ),
        ],
    )
    def test_change_value(self, smooth_c_sd, path, old_value, new_value):
        assert smooth_c_sd[path] == old_value
        smooth_c_sd[path] = new_value
        assert smooth_c_sd[path] != old_value

    @pytest.mark.parametrize(
        "path, leaf",
        [
            ([("env", "production"), "database", "port"], 5342),
            ([frozenset(["cache", "redis"]), "config", "memory"], "4GB"),
            (["monitoring", ("metrics", "cpu")], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AZURE US",
            ),
        ],
    )
    def test_change_control(self, smooth_c_sd, path, leaf):
        assert smooth_c_sd[path] == leaf

    @pytest.mark.parametrize(
        "a_path, b_path, v_type",
        [
            (
                ["global_settings", ("security", "encryption")],
                ["global_settings", "security", "encryption"],
                _StackedDict,
            ),
            (
                ["global_settings", "security", "encryption"],
                ["global_settings", ("security", "encryption")],
                str,
            ),
        ],
    )
    def test_hierarchical(self, smooth_c_sd, a_path, b_path, v_type):
        assert smooth_c_sd[a_path] != smooth_c_sd[b_path]
        assert isinstance(smooth_c_sd[a_path], v_type)

    @pytest.mark.parametrize(
        "false_keys_type, error, error_msg",
        [
            ({1, 2}, TypeError, "unhashable type: 'set'"),
            (
                [1, [1, 2]],
                StackedTypeError,
                "Nested lists are not allowed as keys in _StackedDict. (expected: str, got: list)",
            ),
        ],
    )
    def test_path_type_failed(self, smooth_c_sd, false_keys_type, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            smooth_c_sd[false_keys_type] = None
