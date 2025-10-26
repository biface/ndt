"""
This test_02_cpaths test file is used in ndt to test functions, classes and methods with pytest library.

Created 26/10/2025
"""

import pytest

import ndict_tools
from ndict_tools.tools import _CPaths


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
        "dictionary_name, default_setup_name, compact_path",
        [
            (
                "smooth_c_sd",
                "standard_smooth_c_setup",
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
                "standard_strict_c_setup",
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
        self, dictionary_name, default_setup_name, compact_path, request
    ):
        default_setup_parameters = request.getfixturevalue(default_setup_name)
        c_paths = _CPaths(
            ndict_tools.tools._StackedDict({}, default_setup=default_setup_parameters)
        )
        dictionary = request.getfixturevalue(dictionary_name)
        c_paths.structure = dictionary
        assert c_paths.structure == compact_path
        assert c_paths._stacked_dict is not None
        assert c_paths._stacked_dict == dictionary
        assert c_paths._structure is not None
        assert c_paths._structure == compact_path

    @pytest.mark.parametrize(
        "dictionary_name, default_setup_name, compact_path",
        [
            (
                "smooth_c_sd",
                "standard_smooth_c_setup",
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
                "standard_strict_c_setup",
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
    def test_structure_setter_with_hkey(
        self, dictionary_name, default_setup_name, compact_path, request
    ):
        default_setup_parameters = request.getfixturevalue(default_setup_name)
        c_paths = _CPaths(
            ndict_tools.tools._StackedDict({}, default_setup=default_setup_parameters)
        )
        hkey = ndict_tools.tools._HKey.build_forest(
            request.getfixturevalue(dictionary_name)
        )
        c_paths.structure = hkey
        assert c_paths.structure == compact_path
        assert c_paths._stacked_dict is not None
        assert c_paths._hkey is not None
        assert c_paths._hkey == hkey
        assert c_paths._structure is not None
        assert c_paths._structure == compact_path
