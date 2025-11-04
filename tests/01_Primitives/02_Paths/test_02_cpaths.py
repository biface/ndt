"""
This test_02_cpaths test file is used in ndt to test functions, classes and methods with pytest library.

Created 26/10/2025
"""

import re

import pytest

import ndict_tools
from ndict_tools.tools import _CPaths, _HKey, _StackedDict


def test_init_empty():
    c_paths = _CPaths()
    assert isinstance(c_paths, _CPaths)
    assert c_paths._stacked_dict is None
    assert c_paths._hkey is None
    assert c_paths._structure is None
    assert c_paths._ensure_hkey() is None
    assert c_paths._ensure_hkey() is None
    assert c_paths.structure is None


class TestCPathsInit:

    @pytest.mark.parametrize("dictionary_name", ["smooth_c_sd", "strict_c_sd"])
    def test_simple_init(self, dictionary_name, request):
        c_paths = _CPaths(request.getfixturevalue(dictionary_name))
        assert isinstance(c_paths, _CPaths)
        assert c_paths._stacked_dict is not None and isinstance(
            c_paths._stacked_dict, ndict_tools.tools._StackedDict
        )
        assert c_paths._structure is None

    @pytest.mark.parametrize(
        "dictionary_name, compact_path",
        [
            (
                "smooth_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
            (
                "strict_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
        ],
    )
    def test_structure_property(self, dictionary_name, compact_path, request):
        c_paths = _CPaths(request.getfixturevalue(dictionary_name))
        assert c_paths.structure == compact_path
        assert c_paths._structure is not None
        assert c_paths._structure == compact_path

    @pytest.mark.parametrize(
        "dictionary_name, compact_path",
        [
            (
                "smooth_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
            (
                "strict_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
        ],
    )
    def test_structure_setter_with_dictionary_name(
        self, dictionary_name, compact_path, request
    ):
        c_paths = _CPaths()
        dictionary = request.getfixturevalue(dictionary_name)
        c_paths.structure = dictionary
        assert c_paths.structure == compact_path
        assert c_paths._stacked_dict is not None
        assert isinstance(c_paths._stacked_dict, _StackedDict)
        assert c_paths._stacked_dict == dictionary
        assert c_paths._structure is not None
        assert c_paths._structure == compact_path

    @pytest.mark.parametrize(
        "dictionary_name, compact_path",
        [
            (
                "smooth_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
            (
                "strict_c_sd",
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
            ),
        ],
    )
    def test_structure_setter_with_hkey(self, dictionary_name, compact_path, request):
        c_paths = _CPaths()
        hkey = ndict_tools.tools._HKey.build_forest(
            request.getfixturevalue(dictionary_name)
        )
        c_paths.structure = hkey
        assert c_paths.structure == compact_path
        assert c_paths._stacked_dict is None
        assert c_paths._hkey is not None
        assert isinstance(c_paths._hkey, _HKey)
        assert c_paths._hkey == hkey
        assert c_paths._structure is not None
        assert c_paths._structure == compact_path

    @pytest.mark.parametrize(
        "compact_paths",
        [
            [
                ("env", "production"),
                [
                    "database",
                    "host",
                    "port",
                    "pools",
                    [
                        "replicas",
                        [1, "region", "status", "id"],
                        [2, "region", "status", "id"],
                    ],
                    [
                        "instances",
                        [
                            42,
                            "name",
                            "max_connections",
                            "type",
                            "maintenance_window",
                        ],
                        [54, "name", "max_connections", "type", "sync_lag"],
                    ],
                ],
                ["api", "rate_limit", "timeout"],
            ],
            [
                ("env", "dev"),
                [
                    "database",
                    "host",
                    "port",
                    "pools",
                    [
                        "replicas",
                        [1, "region", "status", "id"],
                        [2, "region", "status", "id"],
                    ],
                    "backup_frequency",
                    [
                        "instances",
                        [
                            12,
                            "name",
                            "max_connections",
                            "type",
                            "auto_cleanup",
                            "reset_schedule",
                        ],
                        [
                            34,
                            "name",
                            "max_connections",
                            "type",
                            "isolation_level",
                            "ephemeral",
                        ],
                    ],
                ],
                ["api", "rate_limit", "timeout", "debug_mode"],
                [
                    "features",
                    "experimental",
                    ["flags", "enable_logging", "mock_external_apis"],
                ],
            ],
            [
                frozenset({"redis", "cache"}),
                "nodes",
                ["config", "ttl", "memory"],
                [
                    "environments",
                    [
                        ("env", "production"),
                        "cluster_size",
                        "persistence",
                        "max_memory_policy",
                    ],
                    [
                        ("env", "dev"),
                        "cluster_size",
                        "persistence",
                        "max_memory_policy",
                    ],
                ],
            ],
            [
                "monitoring",
                ("metrics", "cpu"),
                [("logs", "level"), "error", "debug"],
                [
                    "dashboards",
                    [
                        ("env", "production"),
                        "grafana_url",
                        "alerts",
                        "retention",
                    ],
                    [("env", "dev"), "grafana_url", "alerts", "retention"],
                ],
            ],
            [
                "global_settings",
                [
                    ("security", "encryption"),
                    "algorithm",
                    ["key_rotation", ("env", "production"), ("env", "dev")],
                ],
                ["security", "encryption", "level"],
                [
                    "networking",
                    [
                        "load_balancer",
                        [
                            ("env", "production"),
                            "type",
                            "instances",
                            "health_check_interval",
                        ],
                        [
                            ("env", "dev"),
                            "type",
                            "instances",
                            "health_check_interval",
                        ],
                    ],
                ],
            ],
        ],
    )
    def test_structure_setter_with_list(self, compact_paths):
        c_paths = _CPaths()
        c_paths.structure = compact_paths
        assert c_paths.structure == compact_paths
        assert c_paths._stacked_dict is None
        assert c_paths._hkey is None
        assert c_paths._structure is not None
        assert c_paths._ensure_structure() is not None
        assert c_paths._structure == compact_paths

    @pytest.mark.parametrize(
        "compact_paths, validate_msg_error, error, setter_msg_error",
        [
            (
                "[this [is not [a list]]]",
                "Structure must be a list, got str",
                TypeError,
                "Unsupported type for structure: str. Expected _StackedDict, _HKey or list.",
            ),
            (
                [[1, [2, [3, []]]]],
                "Empty list not allowed in structure",
                ValueError,
                "Empty list not allowed in structure",
            ),
        ],
    )
    def test_structure_setter_with_failed_lists(
        self, compact_paths, validate_msg_error, error, setter_msg_error
    ):
        c_paths = _CPaths()
        with pytest.raises(ValueError, match=re.escape(validate_msg_error)):
            ndict_tools.tools._CPaths._validate_structure(compact_paths)

        with pytest.raises(error, match=re.escape(setter_msg_error)):
            c_paths.structure = compact_paths

    # Maximum depth is defined in ndict_tools.tools.MAX_DEPTH
    def test_structure_with_too_deeply_nested(self):
        ndict_tools.tools.MAX_DEPTH = 5
        with pytest.raises(
            ValueError, match=re.escape("Structure too deeply nested (max depth: 5)")
        ):
            c_paths = _CPaths()
            c_paths.structure = [[1, [2, [3, [4, [5, [6, 7, 8]]]]]]]


class TestCPathsContent:

    def test_c_paths_compact(self, strict_c_sd):
        c_paths = _CPaths()
        c_paths.structure = strict_c_sd
        assert strict_c_sd.compact_paths() == c_paths

    def test_paths_to_cpaths(self, strict_c_sd):
        paths = strict_c_sd.paths()
        c_paths = paths.to_compact()
        assert c_paths == strict_c_sd.paths()

    def test_c_paths_expand(self, strict_c_sd):
        c_paths = _CPaths()
        c_paths.structure = strict_c_sd
        d_paths = c_paths.expand()
        assert d_paths == strict_c_sd.paths()


class TestCPathsCovering:

    def test_full_coverage_verified(self, strict_c_sd):
        c_paths = _CPaths()
        c_paths.structure = strict_c_sd
        assert c_paths.is_covering(strict_c_sd) is True

    def test_full_coverage_value(self, strict_c_sd):
        c_paths = _CPaths()
        c_paths.structure = strict_c_sd
        assert c_paths.coverage(strict_c_sd) == 1.0

    @pytest.mark.parametrize(
        "test_structure, covering, coverage, uncovered, missing",
        [
            (
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                    ["local settings"],
                ],
                False,
                1.0,
                [],
                [["local settings"]],
            ),
            (
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                    [
                        "global_settings",
                        [
                            ("security", "encryption"),
                            "algorithm",
                            ["key_rotation", ("env", "production"), ("env", "dev")],
                        ],
                        ["security", "encryption", "level"],
                        [
                            "networking",
                            [
                                "load_balancer",
                                [
                                    ("env", "production"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                                [
                                    ("env", "dev"),
                                    "type",
                                    "instances",
                                    "health_check_interval",
                                ],
                            ],
                        ],
                    ],
                ],
                True,
                1.0,
                [],
                [],
            ),
            (
                [
                    [
                        ("env", "production"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            [
                                "instances",
                                [
                                    42,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "maintenance_window",
                                ],
                                [54, "name", "max_connections", "type", "sync_lag"],
                            ],
                        ],
                        ["api", "rate_limit", "timeout"],
                    ],
                    [
                        ("env", "dev"),
                        [
                            "database",
                            "host",
                            "port",
                            "pools",
                            [
                                "replicas",
                                [1, "region", "status", "id"],
                                [2, "region", "status", "id"],
                            ],
                            "backup_frequency",
                            [
                                "instances",
                                [
                                    12,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "auto_cleanup",
                                    "reset_schedule",
                                ],
                                [
                                    34,
                                    "name",
                                    "max_connections",
                                    "type",
                                    "isolation_level",
                                    "ephemeral",
                                ],
                            ],
                        ],
                        ["api", "rate_limit", "timeout", "debug_mode"],
                        [
                            "features",
                            "experimental",
                            ["flags", "enable_logging", "mock_external_apis"],
                        ],
                    ],
                    [
                        frozenset({"redis", "cache"}),
                        "nodes",
                        ["config", "ttl", "memory"],
                        [
                            "environments",
                            [
                                ("env", "production"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                            [
                                ("env", "dev"),
                                "cluster_size",
                                "persistence",
                                "max_memory_policy",
                            ],
                        ],
                    ],
                    [
                        "monitoring",
                        ("metrics", "cpu"),
                        [("logs", "level"), "error", "debug"],
                        [
                            "dashboards",
                            [
                                ("env", "production"),
                                "grafana_url",
                                "alerts",
                                "retention",
                            ],
                            [("env", "dev"), "grafana_url", "alerts", "retention"],
                        ],
                    ],
                ],
                False,
                0.8303571428571429,
                [
                    ["global_settings"],
                    ["global_settings", ("security", "encryption")],
                    ["global_settings", ("security", "encryption"), "algorithm"],
                    ["global_settings", ("security", "encryption"), "key_rotation"],
                    [
                        "global_settings",
                        ("security", "encryption"),
                        "key_rotation",
                        ("env", "production"),
                    ],
                    [
                        "global_settings",
                        ("security", "encryption"),
                        "key_rotation",
                        ("env", "dev"),
                    ],
                    ["global_settings", "security"],
                    ["global_settings", "security", "encryption"],
                    ["global_settings", "security", "level"],
                    ["global_settings", "networking"],
                    ["global_settings", "networking", "load_balancer"],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                    ],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "type",
                    ],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "health_check_interval",
                    ],
                    ["global_settings", "networking", "load_balancer", ("env", "dev")],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ],
                    [
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ],
                ],
                [],
            ),
        ],
    )
    def test_partial_structure(
        self, strict_c_sd, test_structure, covering, coverage, uncovered, missing
    ):
        c_paths = _CPaths()
        c_paths.structure = test_structure
        assert c_paths.is_covering(strict_c_sd) is covering
        assert c_paths.coverage(strict_c_sd) == coverage
        assert c_paths.uncovered_paths(strict_c_sd) == uncovered
        assert c_paths.missing_paths(strict_c_sd) == missing
