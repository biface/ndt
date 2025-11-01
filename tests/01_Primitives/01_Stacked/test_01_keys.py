import re

import pytest

from ndict_tools.exception import StackedKeyError, StackedTypeError
from ndict_tools.tools import _StackedDict


@pytest.mark.parametrize(
    "keys, leaf",
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
def test_stacked_dict_keys(strict_f_sd, keys, leaf):
    value = strict_f_sd[keys[0]]
    for key in keys[1:]:
        value = value[key]
    assert value == leaf


class TestKeysStrictSD:

    @pytest.mark.parametrize(
        "keys, end_key, old_value, new_value",
        [
            ([("env", "production"), "database"], "port", 5432, 5342),
            ([frozenset(["cache", "redis"]), "config"], "memory", "2GB", "4GB"),
            (["monitoring"], ("metrics", "cpu"), [80, 90, 95], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "type",
                "AWS ALB",
                "AZURE US",
            ),
        ],
    )
    def test_change_value(self, strict_c_sd, keys, end_key, old_value, new_value):
        d = strict_c_sd
        for key in keys:
            d = d[key]
        assert d[end_key] == old_value
        d[end_key] = new_value
        assert d[end_key] == new_value

    @pytest.mark.parametrize(
        "keys, leaf",
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
    def test_change_control(self, strict_c_sd, keys, leaf):
        value = strict_c_sd[keys[0]]
        for key in keys[1:]:
            value = value[key]
        assert value == leaf

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
    def test_change_keys_failed(
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
    def test_key_type_failed(self, strict_c_sd, false_keys_type, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            strict_c_sd[false_keys_type] = None

    @pytest.mark.parametrize(
        "key_a, key_b, type",
        [
            (("security", "encryption"), ["security", "encryption"], _StackedDict),
            (["security", "encryption"], ("security", "encryption"), str),
        ],
    )
    def test_hybrid_keys(self, strict_c_sd, key_a, key_b, type):
        assert (
            strict_c_sd["global_settings"][key_a]
            != strict_c_sd["global_settings"][key_b]
        )
        assert isinstance(strict_c_sd["global_settings"][key_a], type)


class TestUnpackStrictSD:

    @pytest.mark.parametrize(
        "unpacked_keys",
        [
            (("env", "production"), "database", "host"),
            (("env", "production"), "database", "port"),
            (("env", "production"), "database", "pools"),
            (("env", "production"), "database", "replicas", 1, "status"),
            (("env", "production"), "database", "replicas", 1, "id"),
            (("env", "production"), "database", "replicas", 2, "region"),
            (("env", "production"), "database", "replicas", 2, "id"),
            (("env", "production"), "database", "instances", 42, "name"),
            (("env", "production"), "database", "instances", 42, "max_connections"),
            (("env", "production"), "database", "instances", 42, "maintenance_window"),
            (("env", "production"), "database", "instances", 54, "type"),
            (("env", "production"), "database", "instances", 54, "sync_lag"),
            (("env", "production"), "api", "rate_limit"),
            (("env", "dev"), "database", "host"),
            (("env", "dev"), "database", "port"),
            (("env", "dev"), "database", "pools"),
            (("env", "dev"), "database", "replicas", 1, "id"),
            (("env", "dev"), "database", "replicas", 2, "region"),
            (("env", "dev"), "database", "backup_frequency"),
            (("env", "dev"), "database", "instances", 12, "name"),
            (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
            (("env", "dev"), "database", "instances", 12, "reset_schedule"),
            (("env", "dev"), "database", "instances", 34, "isolation_level"),
            (("env", "dev"), "database", "instances", 34, "ephemeral"),
            (("env", "dev"), "api", "debug_mode"),
            (("env", "dev"), "features", "experimental"),
            (("env", "dev"), "features", "flags", "mock_external_apis"),
            (frozenset({"redis", "cache"}), "nodes"),
            (frozenset({"redis", "cache"}), "config", "ttl"),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "production"),
                "cluster_size",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "dev"),
                "cluster_size",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "dev"),
                "persistence",
            ),
            ("monitoring", ("metrics", "cpu")),
            ("monitoring", ("logs", "level"), "error"),
            ("monitoring", "dashboards", ("env", "production"), "grafana_url"),
            ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
            ("monitoring", "dashboards", ("env", "dev"), "alerts"),
            (
                "global_settings",
                ("security", "encryption"),
                "key_rotation",
                ("env", "production"),
            ),
            (
                "global_settings",
                ("security", "encryption"),
                "key_rotation",
                ("env", "dev"),
            ),
            (
                "global_settings",
                "networking",
                "load_balancer",
                ("env", "production"),
                "health_check_interval",
            ),
            (
                "global_settings",
                "networking",
                "load_balancer",
                ("env", "dev"),
                "health_check_interval",
            ),
        ],
    )
    def test_upack_keys(self, strict_c_sd, unpacked_keys):
        assert unpacked_keys in strict_c_sd.unpacked_keys()

    @pytest.mark.parametrize(
        "unpacked_items",
        [
            ((("env", "production"), "database", "pools"), [5, 10, 15]),
            ((("env", "production"), "database", "replicas", 1, "region"), "us-east"),
            ((("env", "production"), "database", "replicas", 1, "status"), "active"),
            ((("env", "production"), "database", "replicas", 1, "id"), 42),
            ((("env", "production"), "database", "replicas", 2, "region"), "eu-west"),
            ((("env", "production"), "database", "replicas", 2, "status"), "standby"),
            ((("env", "production"), "database", "replicas", 2, "id"), 54),
            (
                (
                    ("env", "production"),
                    "database",
                    "instances",
                    42,
                    "maintenance_window",
                ),
                "02:00-04:00 UTC",
            ),
            ((("env", "production"), "database", "instances", 54, "sync_lag"), "< 1s"),
            ((("env", "production"), "api", "rate_limit"), 10000),
            ((("env", "production"), "api", "timeout"), 30),
            ((("env", "dev"), "database", "host"), "dev-db.internal.com"),
            ((("env", "dev"), "database", "port"), 5433),
            ((("env", "dev"), "database", "pools"), [2, 5, 8]),
            ((("env", "dev"), "database", "instances", 12, "name"), "dev-main"),
            ((("env", "dev"), "database", "instances", 12, "max_connections"), 200),
            ((("env", "dev"), "database", "instances", 12, "type"), "development"),
            ((("env", "dev"), "database", "instances", 12, "auto_cleanup"), True),
            ((("env", "dev"), "database", "instances", 34, "type"), "testing"),
            (
                (("env", "dev"), "database", "instances", 34, "isolation_level"),
                "READ_UNCOMMITTED",
            ),
            ((("env", "dev"), "api", "timeout"), 60),
            ((("env", "dev"), "api", "debug_mode"), True),
            ((("env", "dev"), "features", "experimental"), ["new_auth", "beta_ui"]),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "production"),
                    "persistence",
                ),
                "rdb",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "production"),
                    "max_memory_policy",
                ),
                "allkeys-lru",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "cluster_size",
                ),
                2,
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "persistence",
                ),
                "none",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "max_memory_policy",
                ),
                "volatile-lru",
            ),
            (
                ("monitoring", "dashboards", ("env", "production"), "retention"),
                "1 year",
            ),
            (
                ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
                "http://dev-monitoring.internal.com",
            ),
            (("monitoring", "dashboards", ("env", "dev"), "alerts"), ["email"]),
            (("monitoring", "dashboards", ("env", "dev"), "retention"), "30 days"),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ),
                3,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ),
                30,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ),
                "nginx",
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ),
                1,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ),
                60,
            ),
        ],
    )
    def test_unpack_items(self, strict_c_sd, unpacked_items):
        assert unpacked_items in strict_c_sd.unpacked_items()

    @pytest.mark.parametrize(
        "unpacked_values",
        [
            "prod-db.company.com",
            5432,
            [5, 10, 15],
            True,
            "http://dev-monitoring.internal.com",
        ],
    )
    def test_unpack_values(self, strict_c_sd, unpacked_values):
        assert unpacked_values in strict_c_sd.unpacked_values()

    @pytest.mark.parametrize(
        "key, associated_list",
        [
            (
                "networking",
                [
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "health_check_interval",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            (
                ("env", "dev"),
                [
                    (("env", "dev"), "database", "host"),
                    (("env", "dev"), "database", "port"),
                    (("env", "dev"), "database", "pools"),
                    (("env", "dev"), "database", "replicas", 1, "region"),
                    (("env", "dev"), "database", "replicas", 1, "status"),
                    (("env", "dev"), "database", "replicas", 1, "id"),
                    (("env", "dev"), "database", "replicas", 2, "region"),
                    (("env", "dev"), "database", "replicas", 2, "status"),
                    (("env", "dev"), "database", "replicas", 2, "id"),
                    (("env", "dev"), "database", "backup_frequency"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (("env", "dev"), "api", "rate_limit"),
                    (("env", "dev"), "api", "timeout"),
                    (("env", "dev"), "api", "debug_mode"),
                    (("env", "dev"), "features", "experimental"),
                    (("env", "dev"), "features", "flags", "enable_logging"),
                    (("env", "dev"), "features", "flags", "mock_external_apis"),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "cluster_size",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "persistence",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "max_memory_policy",
                    ),
                    ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
                    ("monitoring", "dashboards", ("env", "dev"), "alerts"),
                    ("monitoring", "dashboards", ("env", "dev"), "retention"),
                    (
                        "global_settings",
                        ("security", "encryption"),
                        "key_rotation",
                        ("env", "dev"),
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            ("encryption", [("global_settings", "security", "encryption")]),
            ("level", [("global_settings", "security", "level")]),
            (
                "instances",
                [
                    (("env", "production"), "database", "instances", 42, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 42, "type"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "maintenance_window",
                    ),
                    (("env", "production"), "database", "instances", 54, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        54,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 54, "type"),
                    (("env", "production"), "database", "instances", 54, "sync_lag"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                ],
            ),
            (
                "sync_lag",
                [(("env", "production"), "database", "instances", 54, "sync_lag")],
            ),
        ],
    )
    def test_key_list(self, strict_c_sd, key, associated_list):
        assert strict_c_sd.key_list(key) == associated_list

    @pytest.mark.parametrize(
        "false_key, error, error_msg",
        [
            (
                "network",
                StackedKeyError,
                "Cannot find the key: network in the stacked dictionary (key: network)",
            ),
            (
                ["network"],
                StackedKeyError,
                "This function manages only atomic keys (key: ['network'])",
            ),
            (
                ("network", "test"),
                StackedKeyError,
                "Cannot find the key: ('network', 'test') in the stacked dictionary (key: ('network', 'test'))",
            ),
        ],
    )
    def test_key_list_false(self, strict_c_sd, false_key, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            strict_c_sd.key_list(false_key)

    @pytest.mark.parametrize(
        "key, items_list",
        [
            (
                "monitoring",
                [
                    [80, 90, 95],
                    "/var/log/error.log",
                    None,
                    "https://monitoring.company.com",
                    ["slack", "pagerduty"],
                    "1 year",
                    "http://dev-monitoring.internal.com",
                    ["email"],
                    "30 days",
                ],
            ),
            (
                "dashboards",
                [
                    "https://monitoring.company.com",
                    ["slack", "pagerduty"],
                    "1 year",
                    "http://dev-monitoring.internal.com",
                    ["email"],
                    "30 days",
                ],
            ),
            ("error", ["/var/log/error.log"]),
        ],
    )
    def test_items_list(self, strict_c_sd, key, items_list):
        assert strict_c_sd.items_list(key) == items_list

    @pytest.mark.parametrize(
        "false_key, error, error_msg",
        [
            (
                "dashboard",
                StackedKeyError,
                "Cannot find the key: dashboard in the stacked dictionary (key: dashboard)",
            ),
            (
                ["monitoring", "dashboard"],
                StackedKeyError,
                "This function manages only atomic keys (key: ['monitoring', 'dashboard'])",
            ),
            (
                None,
                TypeError,
                "items_list() missing 1 required positional argument: 'key'",
            ),
        ],
    )
    def test_items_list_false(self, strict_c_sd, false_key, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            if false_key:
                strict_c_sd.items_list(false_key)
            else:
                strict_c_sd.items_list()

    @pytest.mark.parametrize(
        "value, confirm, first_occurrence",
        [
            ("prod-db.company.com", True, 1),
            (5432, True, 2),
            ("prod-web.compagnie.fr", False, 0),
            ([5, 10, 15], True, 3),
            (5344, False, 0),
            ("us-east", True, 4),
            ("active", True, 5),
            ("slave", False, 0),
            (True, True, 33),
        ],
    )
    def test_leaves(self, strict_c_sd, value, confirm, first_occurrence):
        if confirm:
            assert value in strict_c_sd.leaves()
            assert strict_c_sd.leaves().index(value) == first_occurrence - 1
        else:
            assert value not in strict_c_sd.leaves()


class TestKeysSmoothSD:

    @pytest.mark.parametrize(
        "keys, end_key, old_value, new_value",
        [
            ([("env", "production"), "database"], "port", 5432, 5342),
            ([frozenset(["cache", "redis"]), "config"], "memory", "2GB", "4GB"),
            (["monitoring"], ("metrics", "cpu"), [80, 90, 95], [75, 85, 90]),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "type",
                "AWS ALB",
                "AZURE US",
            ),
        ],
    )
    def test_change_keys(self, smooth_c_sd, keys, end_key, old_value, new_value):
        d = smooth_c_sd
        for key in keys:
            d = d[key]
        assert d[end_key] == old_value
        d[end_key] = new_value
        assert d[end_key] == new_value

    @pytest.mark.parametrize(
        "keys, leaf",
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
    def test_change_control(self, smooth_c_sd, keys, leaf):
        value = smooth_c_sd[keys[0]]
        for key in keys[1:]:
            value = value[key]
        assert value == leaf

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
    def test_key_type_failed(self, smooth_c_sd, false_keys_type, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            smooth_c_sd[false_keys_type] = None

    @pytest.mark.parametrize(
        "key_a, key_b, type",
        [
            (("security", "encryption"), ["security", "encryption"], _StackedDict),
            (["security", "encryption"], ("security", "encryption"), str),
        ],
    )
    def test_hybrid_keys(self, strict_c_sd, key_a, key_b, type):
        assert (
            strict_c_sd["global_settings"][key_a]
            != strict_c_sd["global_settings"][key_b]
        )
        assert isinstance(strict_c_sd["global_settings"][key_a], type)

    @pytest.mark.parametrize(
        "key, associated_list",
        [
            (
                "networking",
                [
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "health_check_interval",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            (
                ("env", "dev"),
                [
                    (("env", "dev"), "database", "host"),
                    (("env", "dev"), "database", "port"),
                    (("env", "dev"), "database", "pools"),
                    (("env", "dev"), "database", "replicas", 1, "region"),
                    (("env", "dev"), "database", "replicas", 1, "status"),
                    (("env", "dev"), "database", "replicas", 1, "id"),
                    (("env", "dev"), "database", "replicas", 2, "region"),
                    (("env", "dev"), "database", "replicas", 2, "status"),
                    (("env", "dev"), "database", "replicas", 2, "id"),
                    (("env", "dev"), "database", "backup_frequency"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (("env", "dev"), "api", "rate_limit"),
                    (("env", "dev"), "api", "timeout"),
                    (("env", "dev"), "api", "debug_mode"),
                    (("env", "dev"), "features", "experimental"),
                    (("env", "dev"), "features", "flags", "enable_logging"),
                    (("env", "dev"), "features", "flags", "mock_external_apis"),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "cluster_size",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "persistence",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "max_memory_policy",
                    ),
                    ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
                    ("monitoring", "dashboards", ("env", "dev"), "alerts"),
                    ("monitoring", "dashboards", ("env", "dev"), "retention"),
                    (
                        "global_settings",
                        ("security", "encryption"),
                        "key_rotation",
                        ("env", "dev"),
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            ("encryption", [("global_settings", "security", "encryption")]),
            ("level", [("global_settings", "security", "level")]),
            (
                "instances",
                [
                    (("env", "production"), "database", "instances", 42, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 42, "type"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "maintenance_window",
                    ),
                    (("env", "production"), "database", "instances", 54, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        54,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 54, "type"),
                    (("env", "production"), "database", "instances", 54, "sync_lag"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                ],
            ),
            (
                "sync_lag",
                [(("env", "production"), "database", "instances", 54, "sync_lag")],
            ),
        ],
    )
    def test_key_list(self, strict_c_sd, key, associated_list):
        assert strict_c_sd.key_list(key) == associated_list


class TestUnpackSmoothSD:

    @pytest.mark.parametrize(
        "unpacked_keys",
        [
            (("env", "production"), "api", "timeout"),
            (("env", "dev"), "database", "host"),
            (("env", "dev"), "database", "port"),
            (("env", "dev"), "database", "pools"),
            (("env", "dev"), "database", "replicas", 1, "region"),
            (("env", "dev"), "database", "replicas", 1, "status"),
            (("env", "dev"), "database", "replicas", 1, "id"),
            (("env", "dev"), "database", "replicas", 2, "region"),
            (("env", "dev"), "database", "replicas", 2, "status"),
            (("env", "dev"), "api", "debug_mode"),
            (("env", "dev"), "features", "experimental"),
            (("env", "dev"), "features", "flags", "enable_logging"),
            (("env", "dev"), "features", "flags", "mock_external_apis"),
            (frozenset({"redis", "cache"}), "nodes"),
            (frozenset({"redis", "cache"}), "config", "ttl"),
            (frozenset({"redis", "cache"}), "config", "memory"),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "production"),
                "cluster_size",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "production"),
                "persistence",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "production"),
                "max_memory_policy",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "dev"),
                "cluster_size",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "dev"),
                "persistence",
            ),
            (
                frozenset({"redis", "cache"}),
                "environments",
                ("env", "dev"),
                "max_memory_policy",
            ),
            ("monitoring", ("metrics", "cpu")),
            ("monitoring", ("logs", "level"), "error"),
            ("monitoring", ("logs", "level"), "debug"),
            ("monitoring", "dashboards", ("env", "production"), "grafana_url"),
            ("monitoring", "dashboards", ("env", "production"), "alerts"),
            ("monitoring", "dashboards", ("env", "production"), "retention"),
            ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
            ("monitoring", "dashboards", ("env", "dev"), "alerts"),
            (
                "global_settings",
                "networking",
                "load_balancer",
                ("env", "production"),
                "health_check_interval",
            ),
            ("global_settings", "networking", "load_balancer", ("env", "dev"), "type"),
            (
                "global_settings",
                "networking",
                "load_balancer",
                ("env", "dev"),
                "instances",
            ),
            (
                "global_settings",
                "networking",
                "load_balancer",
                ("env", "dev"),
                "health_check_interval",
            ),
        ],
    )
    def test_upack_keys(self, smooth_c_sd, unpacked_keys):
        assert unpacked_keys in smooth_c_sd.unpacked_keys()

    @pytest.mark.parametrize(
        "unpacked_items",
        [
            ((("env", "production"), "database", "host"), "prod-db.company.com"),
            ((("env", "production"), "database", "port"), 5432),
            ((("env", "production"), "database", "pools"), [5, 10, 15]),
            ((("env", "production"), "database", "replicas", 1, "region"), "us-east"),
            (
                (("env", "production"), "database", "instances", 54, "type"),
                "read_replica",
            ),
            ((("env", "production"), "database", "instances", 54, "sync_lag"), "< 1s"),
            ((("env", "production"), "api", "rate_limit"), 10000),
            ((("env", "production"), "api", "timeout"), 30),
            ((("env", "dev"), "database", "host"), "dev-db.internal.com"),
            ((("env", "dev"), "database", "port"), 5433),
            ((("env", "dev"), "database", "pools"), [2, 5, 8]),
            ((("env", "dev"), "database", "replicas", 1, "region"), "us-east"),
            ((("env", "dev"), "database", "replicas", 1, "status"), "active"),
            ((("env", "dev"), "database", "replicas", 1, "id"), 12),
            ((("env", "dev"), "database", "replicas", 2, "region"), "eu-west"),
            ((("env", "dev"), "database", "replicas", 2, "status"), "standby"),
            ((("env", "dev"), "database", "replicas", 2, "id"), 34),
            ((("env", "dev"), "database", "backup_frequency"), "daily"),
            ((("env", "dev"), "database", "instances", 12, "name"), "dev-main"),
            ((("env", "dev"), "database", "instances", 12, "max_connections"), 200),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "production"),
                    "cluster_size",
                ),
                6,
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "production"),
                    "persistence",
                ),
                "rdb",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "production"),
                    "max_memory_policy",
                ),
                "allkeys-lru",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "cluster_size",
                ),
                2,
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "persistence",
                ),
                "none",
            ),
            (
                (
                    frozenset({"redis", "cache"}),
                    "environments",
                    ("env", "dev"),
                    "max_memory_policy",
                ),
                "volatile-lru",
            ),
            (("monitoring", ("metrics", "cpu")), [80, 90, 95]),
            (("monitoring", ("logs", "level"), "error"), "/var/log/error.log"),
            (("monitoring", ("logs", "level"), "debug"), None),
            (
                ("monitoring", "dashboards", ("env", "production"), "grafana_url"),
                "https://monitoring.company.com",
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ),
                3,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ),
                30,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ),
                "nginx",
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ),
                1,
            ),
            (
                (
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ),
                60,
            ),
        ],
    )
    def test_unpack_items(self, smooth_c_sd, unpacked_items):
        assert unpacked_items in smooth_c_sd.unpacked_items()

    @pytest.mark.parametrize(
        "unpacked_values",
        [
            "prod-db.company.com",
            5432,
            [5, 10, 15],
            True,
            "http://dev-monitoring.internal.com",
            "standby",
            "eu-west",
        ],
    )
    def test_unpack_values(self, strict_c_sd, unpacked_values):
        assert unpacked_values in strict_c_sd.unpacked_values()

    @pytest.mark.parametrize(
        "key, associated_list",
        [
            (
                "networking",
                [
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "health_check_interval",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            (
                ("env", "dev"),
                [
                    (("env", "dev"), "database", "host"),
                    (("env", "dev"), "database", "port"),
                    (("env", "dev"), "database", "pools"),
                    (("env", "dev"), "database", "replicas", 1, "region"),
                    (("env", "dev"), "database", "replicas", 1, "status"),
                    (("env", "dev"), "database", "replicas", 1, "id"),
                    (("env", "dev"), "database", "replicas", 2, "region"),
                    (("env", "dev"), "database", "replicas", 2, "status"),
                    (("env", "dev"), "database", "replicas", 2, "id"),
                    (("env", "dev"), "database", "backup_frequency"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (("env", "dev"), "api", "rate_limit"),
                    (("env", "dev"), "api", "timeout"),
                    (("env", "dev"), "api", "debug_mode"),
                    (("env", "dev"), "features", "experimental"),
                    (("env", "dev"), "features", "flags", "enable_logging"),
                    (("env", "dev"), "features", "flags", "mock_external_apis"),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "cluster_size",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "persistence",
                    ),
                    (
                        frozenset({"cache", "redis"}),
                        "environments",
                        ("env", "dev"),
                        "max_memory_policy",
                    ),
                    ("monitoring", "dashboards", ("env", "dev"), "grafana_url"),
                    ("monitoring", "dashboards", ("env", "dev"), "alerts"),
                    ("monitoring", "dashboards", ("env", "dev"), "retention"),
                    (
                        "global_settings",
                        ("security", "encryption"),
                        "key_rotation",
                        ("env", "dev"),
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "type",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "health_check_interval",
                    ),
                ],
            ),
            ("encryption", [("global_settings", "security", "encryption")]),
            ("level", [("global_settings", "security", "level")]),
            (
                "instances",
                [
                    (("env", "production"), "database", "instances", 42, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 42, "type"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        42,
                        "maintenance_window",
                    ),
                    (("env", "production"), "database", "instances", 54, "name"),
                    (
                        ("env", "production"),
                        "database",
                        "instances",
                        54,
                        "max_connections",
                    ),
                    (("env", "production"), "database", "instances", 54, "type"),
                    (("env", "production"), "database", "instances", 54, "sync_lag"),
                    (("env", "dev"), "database", "instances", 12, "name"),
                    (("env", "dev"), "database", "instances", 12, "max_connections"),
                    (("env", "dev"), "database", "instances", 12, "type"),
                    (("env", "dev"), "database", "instances", 12, "auto_cleanup"),
                    (("env", "dev"), "database", "instances", 12, "reset_schedule"),
                    (("env", "dev"), "database", "instances", 34, "name"),
                    (("env", "dev"), "database", "instances", 34, "max_connections"),
                    (("env", "dev"), "database", "instances", 34, "type"),
                    (("env", "dev"), "database", "instances", 34, "isolation_level"),
                    (("env", "dev"), "database", "instances", 34, "ephemeral"),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "production"),
                        "instances",
                    ),
                    (
                        "global_settings",
                        "networking",
                        "load_balancer",
                        ("env", "dev"),
                        "instances",
                    ),
                ],
            ),
            (
                "sync_lag",
                [(("env", "production"), "database", "instances", 54, "sync_lag")],
            ),
        ],
    )
    def test_key_list(self, smooth_c_sd, key, associated_list):
        assert smooth_c_sd.key_list(key) == associated_list

    @pytest.mark.parametrize(
        "false_key, error, error_msg",
        [
            (
                "network",
                StackedKeyError,
                "Cannot find the key: network in the stacked dictionary (key: network)",
            ),
            (
                ["network"],
                StackedKeyError,
                "This function manages only atomic keys (key: ['network'])",
            ),
            (
                ("network", "test"),
                StackedKeyError,
                "Cannot find the key: ('network', 'test') in the stacked dictionary (key: ('network', 'test'))",
            ),
        ],
    )
    def test_key_list_false(self, smooth_c_sd, false_key, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            smooth_c_sd.key_list(false_key)

    @pytest.mark.parametrize(
        "key, items_list",
        [
            (
                "monitoring",
                [
                    [80, 90, 95],
                    "/var/log/error.log",
                    None,
                    "https://monitoring.company.com",
                    ["slack", "pagerduty"],
                    "1 year",
                    "http://dev-monitoring.internal.com",
                    ["email"],
                    "30 days",
                ],
            ),
            (
                "dashboards",
                [
                    "https://monitoring.company.com",
                    ["slack", "pagerduty"],
                    "1 year",
                    "http://dev-monitoring.internal.com",
                    ["email"],
                    "30 days",
                ],
            ),
            ("error", ["/var/log/error.log"]),
        ],
    )
    def test_items_list(self, smooth_c_sd, key, items_list):
        assert smooth_c_sd.items_list(key) == items_list

    @pytest.mark.parametrize(
        "false_key, error, error_msg",
        [
            (
                "dashboard",
                StackedKeyError,
                "Cannot find the key: dashboard in the stacked dictionary (key: dashboard)",
            ),
            (
                ["monitoring", "dashboard"],
                StackedKeyError,
                "This function manages only atomic keys (key: ['monitoring', 'dashboard'])",
            ),
            (
                None,
                TypeError,
                "items_list() missing 1 required positional argument: 'key'",
            ),
        ],
    )
    def test_items_list_false(self, smooth_c_sd, false_key, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            if false_key:
                smooth_c_sd.items_list(false_key)
            else:
                smooth_c_sd.items_list()

    @pytest.mark.parametrize(
        "value, confirm, first_occurrence",
        [
            ("prod-db.company.com", True, 1),
            (5432, True, 2),
            ("prod-web.compagnie.fr", False, 0),
            ([5, 10, 15], True, 3),
            (5344, False, 0),
            ("us-east", True, 4),
            ("active", True, 5),
            ("slave", False, 0),
            (True, True, 33),
        ],
    )
    def test_leaves(self, smooth_c_sd, value, confirm, first_occurrence):
        if confirm:
            assert value in smooth_c_sd.leaves()
            assert smooth_c_sd.leaves().index(value) == first_occurrence - 1
        else:
            assert value not in smooth_c_sd.leaves()


class TestBuildStrictStackedDict:

    @pytest.mark.parametrize(
        "keys, value",
        [
            ([("env", "production")], {}),
            ([("env", "production"), "database"], {}),
            ([("env", "production"), "database", "host"], "prod-db.company.com"),
            ([("env", "production"), "database", "port"], 5432),
            ([("env", "production"), "database", "pools"], [5, 10, 15]),
            (
                [("env", "production"), "database", "replicas"],
                {
                    1: {"region": "us-east", "status": "active", "id": 42},
                    2: {"region": "eu-west", "status": "standby", "id": 54},
                },
            ),
            ([("env", "production"), "database", "instances"], {}),
            ([("env", "production"), "database", "instances", 42], {}),
            (
                [("env", "production"), "database", "instances", 42, "name"],
                "prod-primary",
            ),
            (
                [("env", "production"), "database", "instances", 42, "max_connections"],
                1000,
            ),
            ([("env", "production"), "database", "instances", 42, "type"], "primary"),
            (
                [
                    ("env", "production"),
                    "database",
                    "instances",
                    42,
                    "maintenance_window",
                ],
                "02:00-04:00 UTC",
            ),
            (
                [("env", "production"), "database", "instances", 54],
                {
                    "name": "prod-secondary",
                    "max_connections": 800,
                    "type": "read_replica",
                    "sync_lag": "< 1s",
                },
            ),
            ([("env", "production"), "api"], {"rate_limit": 10000, "timeout": 30}),
            ([("env", "dev")], {}),
            ([frozenset(["cache", "redis"])], {}),
            (["monitoring"], {}),
            (["global_settings"], {}),
            ([("env", "dev"), "database"], {}),
            ([frozenset(["cache", "redis"]), "nodes"], ["cache-1", "cache-2"]),
            (["monitoring", ("metrics", "cpu")], {}),
            (["global_settings", ("security", "encryption")], {}),
            (
                [("env", "dev"), "api"],
                {"rate_limit": 1000, "timeout": 60, "debug_mode": True},
            ),
            ([("env", "dev"), "features"], {}),
            ([("env", "dev"), "features", "experimental"], ["new_auth", "beta_ui"]),
            (
                [("env", "dev"), "features", "flags"],
                {"enable_logging": True, "mock_external_apis": True},
            ),
            ([("env", "dev"), "database", "host"], "dev-db.internal.com"),
            ([("env", "dev"), "database", "port"], 5433),
            ([("env", "dev"), "database", "pools"], [2, 5, 8]),
            ([("env", "dev"), "database", "replicas"], {}),
            (
                [("env", "dev"), "database", "replicas", 1],
                {"region": "us-east", "status": "active", "id": 12},
            ),
            ([("env", "dev"), "database", "replicas", 2], {}),
            ([("env", "dev"), "database", "replicas", 2, "region"], "eu-west"),
            ([("env", "dev"), "database", "replicas", 2, "status"], "standby"),
            ([("env", "dev"), "database", "replicas", 2, "id"], 34),
            ([("env", "dev"), "database", "backup_frequency"], "daily"),
            ([("env", "dev"), "database", "instances"], {}),
            (
                [("env", "dev"), "database", "instances", 34],
                {
                    "name": "dev-testing",
                    "max_connections": 150,
                    "type": "testing",
                    "isolation_level": "READ_UNCOMMITTED",
                    "ephemeral": True,
                },
            ),
            ([("env", "dev"), "database", "instances", 12], {}),
            ([("env", "dev"), "database", "instances", 12, "name"], "dev-main"),
            ([("env", "dev"), "database", "instances", 12, "max_connections"], 200),
            ([("env", "dev"), "database", "instances", 12, "type"], "development"),
            ([("env", "dev"), "database", "instances", 12, "auto_cleanup"], True),
            ([("env", "dev"), "database", "instances", 12, "reset_schedule"], "weekly"),
            ([frozenset(["cache", "redis"]), "config"], {"ttl": 3600, "memory": "2GB"}),
            ([frozenset(["cache", "redis"]), "environments"], {}),
            (
                [frozenset(["cache", "redis"]), "environments", ("env", "production")],
                {
                    "cluster_size": 6,
                    "persistence": "rdb",
                    "max_memory_policy": "allkeys-lru",
                },
            ),
            ([frozenset(["cache", "redis"]), "environments", ("env", "dev")], {}),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "cluster_size",
                ],
                2,
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "persistence",
                ],
                "none",
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "max_memory_policy",
                ],
                "volatile-lru",
            ),
            (["monitoring", ("logs", "level")], {}),
            (
                ["monitoring", ("metrics", "cpu")],
                [80, 90, 95],
            ),  # reassign a list instead of a dict
            (["monitoring", ("logs", "level"), "error"], "/var/log/error.log"),
            (["monitoring", ("logs", "level"), "debug"], None),
            (["monitoring", "dashboards"], {}),
            (["monitoring", "dashboards", ("env", "production")], {}),
            (
                ["monitoring", "dashboards", ("env", "dev")],
                {
                    "grafana_url": "http://dev-monitoring.internal.com",
                    "alerts": ["email"],
                    "retention": "30 days",
                },
            ),
            (
                ["monitoring", "dashboards", ("env", "production"), "grafana_url"],
                "https://monitoring.company.com",
            ),
            (
                ["monitoring", "dashboards", ("env", "production"), "alerts"],
                ["slack", "pagerduty"],
            ),
            (
                ["monitoring", "dashboards", ("env", "production"), "retention"],
                "1 year",
            ),
            (
                ["global_settings", ("security", "encryption"), "algorithm"],
                "AES-256-GCM",
            ),
            (
                ["global_settings", ("security", "encryption"), "key_rotation"],
                {("env", "production"): 90, ("env", "dev"): 365},
            ),
            (
                ["global_settings", "security"],
                {"encryption": "mandatory", "level": 100},
            ),
            (
                ["global_settings", "networking"],
                {
                    "load_balancer": {
                        ("env", "production"): {
                            "type": "AWS ALB",
                            "instances": 3,
                            "health_check_interval": 30,
                        },
                        ("env", "dev"): {
                            "type": "nginx",
                            "instances": 1,
                            "health_check_interval": 60,
                        },
                    }
                },
            ),
        ],
    )
    def test_build_with_keys(
        self, strict_c_sd, empty_c_strict_sd, standard_strict_c_setup, keys, value
    ):
        stacked_dictionary = empty_c_strict_sd
        d_path = []
        for key in keys[:-1]:
            stacked_dictionary = stacked_dictionary[key]
            d_path.append(key)
        if isinstance(value, dict):
            stacked_dictionary[keys[-1]] = _StackedDict(
                value, default_setup=standard_strict_c_setup
            )
        else:
            stacked_dictionary[keys[-1]] = value
        d_path.append(keys[-1])
        assert empty_c_strict_sd[d_path] == stacked_dictionary[keys[-1]]
        print(empty_c_strict_sd)

    def test_compare(self, strict_c_sd, empty_c_strict_sd):
        assert strict_c_sd == empty_c_strict_sd

    @pytest.mark.parametrize(
        "keys",
        [
            [("env", "dev"), "database", "instances", 12, "max_connections"],
        ],
    )
    def test_delete_with_key(self, empty_c_strict_sd, keys):
        del empty_c_strict_sd[keys]
        with pytest.raises(KeyError):
            assert empty_c_strict_sd[keys]

    @pytest.mark.parametrize(
        "keys, value",
        [
            ([("env", "dev"), "database", "instances", 12, "max_connections"], 200),
        ],
    )
    def test_update_with_key(self, empty_c_strict_sd, strict_c_sd, keys, value):
        empty_c_strict_sd[keys[:-1]].update({keys[-1]: value})
        assert empty_c_strict_sd[keys] == strict_c_sd[keys]

    def test_kwargs_update(self, empty_c_strict_sd, strict_c_sd):
        print(empty_c_strict_sd)
        del empty_c_strict_sd[[("env", "production"), "api", "timeout"]]
        del empty_c_strict_sd[[("env", "production"), "api", "rate_limit"]]
        with pytest.raises(KeyError):
            assert empty_c_strict_sd[[("env", "production"), "api", "timeout"]]
            assert empty_c_strict_sd[[("env", "production"), "api", "rate_limit"]]
        empty_c_strict_sd[[("env", "production")]].update(api={"rate_limit": 10000})
        empty_c_strict_sd[[("env", "production"), "api"]].update(timeout=30)
        assert (
            empty_c_strict_sd[[("env", "production"), "api", "timeout"]]
            == strict_c_sd[[("env", "production"), "api", "timeout"]]
        )

    def test_compare_again(self, empty_c_strict_sd, strict_c_sd):
        assert empty_c_strict_sd == strict_c_sd


class TestBuildSmoothStackedDict:

    @pytest.mark.parametrize(
        "path, value",
        [
            ([("env", "production"), "database", "host"], "prod-db.company.com"),
            ([("env", "production"), "database", "port"], 5432),
            ([("env", "production"), "database", "pools"], [5, 10, 15]),
            ([("env", "production"), "database", "replicas", 1, "region"], "us-east"),
            ([("env", "production"), "database", "replicas", 1, "status"], "active"),
            ([("env", "production"), "database", "replicas", 1, "id"], 42),
            ([("env", "production"), "database", "replicas", 2, "region"], "eu-west"),
            ([("env", "production"), "database", "replicas", 2, "status"], "standby"),
            ([("env", "production"), "database", "replicas", 2, "id"], 54),
            (
                [("env", "production"), "database", "instances", 42, "name"],
                "prod-primary",
            ),
            (
                [("env", "production"), "database", "instances", 42, "max_connections"],
                1000,
            ),
            ([("env", "production"), "database", "instances", 42, "type"], "primary"),
            (
                [
                    ("env", "production"),
                    "database",
                    "instances",
                    42,
                    "maintenance_window",
                ],
                "02:00-04:00 UTC",
            ),
            (
                [("env", "production"), "database", "instances", 54, "name"],
                "prod-secondary",
            ),
            (
                [("env", "production"), "database", "instances", 54, "max_connections"],
                800,
            ),
            (
                [("env", "production"), "database", "instances", 54, "type"],
                "read_replica",
            ),
            ([("env", "production"), "database", "instances", 54, "sync_lag"], "< 1s"),
            ([("env", "production"), "api", "rate_limit"], 10000),
            ([("env", "production"), "api", "timeout"], 30),
            ([frozenset(["cache", "redis"]), "nodes"], ["cache-1", "cache-2"]),
            ([frozenset(["cache", "redis"]), "config", "ttl"], 3600),
            ([frozenset(["cache", "redis"]), "config", "memory"], "2GB"),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "production"),
                    "cluster_size",
                ],
                6,
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "production"),
                    "persistence",
                ],
                "rdb",
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "production"),
                    "max_memory_policy",
                ],
                "allkeys-lru",
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "cluster_size",
                ],
                2,
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "persistence",
                ],
                "none",
            ),
            (
                [
                    frozenset(["cache", "redis"]),
                    "environments",
                    ("env", "dev"),
                    "max_memory_policy",
                ],
                "volatile-lru",
            ),
            ([("env", "dev"), "api", "rate_limit"], 1000),
            ([("env", "dev"), "api", "timeout"], 60),
            ([("env", "dev"), "api", "debug_mode"], True),
            ([("env", "dev"), "features", "experimental"], ["new_auth", "beta_ui"]),
            ([("env", "dev"), "features", "flags", "enable_logging"], True),
            ([("env", "dev"), "features", "flags", "mock_external_apis"], True),
            ([("env", "dev"), "database", "host"], "dev-db.internal.com"),
            ([("env", "dev"), "database", "port"], 5433),
            ([("env", "dev"), "database", "pools"], [2, 5, 8]),
            ([("env", "dev"), "database", "replicas", 1, "region"], "us-east"),
            ([("env", "dev"), "database", "replicas", 1, "status"], "active"),
            ([("env", "dev"), "database", "replicas", 1, "id"], 12),
            ([("env", "dev"), "database", "replicas", 2, "region"], "eu-west"),
            ([("env", "dev"), "database", "replicas", 2, "status"], "standby"),
            ([("env", "dev"), "database", "replicas", 2, "id"], 34),
            ([("env", "dev"), "database", "backup_frequency"], "daily"),
            ([("env", "dev"), "database", "instances", 34, "name"], "dev-testing"),
            ([("env", "dev"), "database", "instances", 34, "max_connections"], 150),
            ([("env", "dev"), "database", "instances", 34, "type"], "testing"),
            (
                [("env", "dev"), "database", "instances", 34, "isolation_level"],
                "READ_UNCOMMITTED",
            ),
            ([("env", "dev"), "database", "instances", 34, "ephemeral"], True),
            ([("env", "dev"), "database", "instances", 12, "name"], "dev-main"),
            ([("env", "dev"), "database", "instances", 12, "max_connections"], 200),
            ([("env", "dev"), "database", "instances", 12, "type"], "development"),
            ([("env", "dev"), "database", "instances", 12, "auto_cleanup"], True),
            ([("env", "dev"), "database", "instances", 12, "reset_schedule"], "weekly"),
            (
                ["monitoring", ("metrics", "cpu")],
                [80, 90, 95],
            ),  # reassign a list instead of a dict
            (["monitoring", ("logs", "level"), "error"], "/var/log/error.log"),
            (["monitoring", ("logs", "level"), "debug"], None),
            (
                ["monitoring", "dashboards", ("env", "dev"), "grafana_url"],
                "http://dev-monitoring.internal.com",
            ),
            (["monitoring", "dashboards", ("env", "dev"), "alerts"], ["email"]),
            (["monitoring", "dashboards", ("env", "dev"), "retention"], "30 days"),
            (
                ["monitoring", "dashboards", ("env", "production"), "grafana_url"],
                "https://monitoring.company.com",
            ),
            (
                ["monitoring", "dashboards", ("env", "production"), "alerts"],
                ["slack", "pagerduty"],
            ),
            (
                ["monitoring", "dashboards", ("env", "production"), "retention"],
                "1 year",
            ),
            (
                ["global_settings", ("security", "encryption"), "algorithm"],
                "AES-256-GCM",
            ),
            (
                [
                    "global_settings",
                    ("security", "encryption"),
                    "key_rotation",
                    ("env", "production"),
                ],
                90,
            ),
            (
                [
                    "global_settings",
                    ("security", "encryption"),
                    "key_rotation",
                    ("env", "dev"),
                ],
                365,
            ),
            (["global_settings", "security", "encryption"], "mandatory"),
            (["global_settings", "security", "level"], 100),
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
                    "health_check_interval",
                ],
                30,
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
                    "health_check_interval",
                ],
                60,
            ),
        ],
    )
    def test_build_with_paths(self, smooth_c_sd, empty_c_smooth_sd, path, value):
        empty_c_smooth_sd[path] = value
        assert smooth_c_sd[path] == empty_c_smooth_sd[path]
        print(empty_c_smooth_sd)

    def test_compare(self, smooth_c_sd, empty_c_smooth_sd):
        assert smooth_c_sd == empty_c_smooth_sd
