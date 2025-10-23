import pytest


class TestHKeyIter:

    @pytest.mark.parametrize(
        "level, length, keys",
        [
            (
                0,
                5,
                [
                    ("env", "production"),
                    ("env", "dev"),
                    frozenset({"cache", "redis"}),
                    "monitoring",
                    "global_settings",
                ],
            ),
            (
                1,
                14,
                [
                    "database",
                    "api",
                    "database",
                    "api",
                    "features",
                    "nodes",
                    "config",
                    "environments",
                    ("metrics", "cpu"),
                    ("logs", "level"),
                    "dashboards",
                    ("security", "encryption"),
                    "security",
                    "networking",
                ],
            ),
            (
                2,
                31,
                [
                    "host",
                    "port",
                    "pools",
                    "replicas",
                    "instances",
                    "rate_limit",
                    "timeout",
                    "host",
                    "port",
                    "pools",
                    "replicas",
                    "backup_frequency",
                    "instances",
                    "rate_limit",
                    "timeout",
                    "debug_mode",
                    "experimental",
                    "flags",
                    "ttl",
                    "memory",
                    ("env", "production"),
                    ("env", "dev"),
                    "error",
                    "debug",
                    ("env", "production"),
                    ("env", "dev"),
                    "algorithm",
                    "key_rotation",
                    "encryption",
                    "level",
                    "load_balancer",
                ],
            ),
            (
                3,
                26,
                [
                    1,
                    2,
                    42,
                    54,
                    1,
                    2,
                    12,
                    34,
                    "enable_logging",
                    "mock_external_apis",
                    "cluster_size",
                    "persistence",
                    "max_memory_policy",
                    "cluster_size",
                    "persistence",
                    "max_memory_policy",
                    "grafana_url",
                    "alerts",
                    "retention",
                    "grafana_url",
                    "alerts",
                    "retention",
                    ("env", "production"),
                    ("env", "dev"),
                    ("env", "production"),
                    ("env", "dev"),
                ],
            ),
            (
                4,
                36,
                [
                    "region",
                    "status",
                    "id",
                    "region",
                    "status",
                    "id",
                    "name",
                    "max_connections",
                    "type",
                    "maintenance_window",
                    "name",
                    "max_connections",
                    "type",
                    "sync_lag",
                    "region",
                    "status",
                    "id",
                    "region",
                    "status",
                    "id",
                    "name",
                    "max_connections",
                    "type",
                    "auto_cleanup",
                    "reset_schedule",
                    "name",
                    "max_connections",
                    "type",
                    "isolation_level",
                    "ephemeral",
                    "type",
                    "instances",
                    "health_check_interval",
                    "type",
                    "instances",
                    "health_check_interval",
                ],
            ),
        ],
    )
    def test_iter_by_level(self, key_tree, level, length, keys):
        for depth, nodes in key_tree.iter_by_level():
            if depth == level:
                assert length == len(nodes)
                n_keys = []
                for node in nodes:
                    n_keys.append(node.key)
                assert n_keys == keys
                break

    @pytest.mark.parametrize(
        "key, valid",
        [
            ("host", True),
            ("monitoring", False),
            ("id", True),
            ("status", True),
            ("database", False),
            (("env", "dev"), True),
            (frozenset({"cache", "redis"}), False),
        ],
    )
    def test_iter_leaves(self, key_tree, key, valid):
        assert (key in [child.key for child in key_tree.iter_leaves()]) is valid

    @pytest.mark.parametrize(
        "child_key, depth, children",
        [
            (("env", "production"), 0, 2),
            (("env", "dev"), 0, 3),
            (frozenset({"cache", "redis"}), 0, 3),
        ],
    )
    def test_iter_child(self, key_tree, child_key, depth, children):
        assert (child_key, depth, children) in [
            (child.key, child.get_depth(), len(child.children))
            for child in key_tree.iter_children()
        ]
