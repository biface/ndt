import re

import pytest

from ndict_tools.exception import StackedKeyError, StackedTypeError
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
def test_path_strict_sd(strict_f_sd, path, leaf):
    assert strict_f_sd[path] == leaf


class TestPathStrictSD:

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
        "false_path, error, error_msg",
        [
            ([("env", "production"), "database", "ports"], KeyError, "'ports'"),
            ([frozenset(["cache", "redis"]), "config", "me_ory"], KeyError, "'me_ory'"),
            (["monitoring", ("metrics", "cpus")], KeyError, "('metrics', 'cpus')"),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "types",
                ],
                KeyError,
                "'types'",
            ),
            (
                [frozenset(["cache", "redis"]), ["config", "memory"]],
                StackedTypeError,
                "Nested lists are not allowed as keys in _StackedDict. (expected: str, got: list)",
            ),
        ],
    )
    def test_change_paths_failed(self, strict_c_sd, false_path, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            test = strict_c_sd[false_path]

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

    @pytest.mark.parametrize(
        "path, error_msg",
        [
            (
                ["global-settings", "security"],
                "This function manages only atomic keys (key: ['global-settings', 'security'])",
            ),
            (
                [("env", "dev")],
                "This function manages only atomic keys (key: [('env', 'dev')])",
            ),
        ],
    )
    def test_paths_list_failed(self, strict_c_sd, path, error_msg):
        with pytest.raises(StackedKeyError, match=re.escape(error_msg)):
            strict_c_sd.is_key(path)


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
def test_path_smooth_sd(smooth_f_sd, path, leaf):
    assert smooth_f_sd[path] == leaf

class TestPathSmoothSD:

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
        "false_path, error, error_msg",
        [
            (
                [frozenset(["cache", "redis"]), ["config", "memory"]],
                StackedTypeError,
                "Nested lists are not allowed as keys in _StackedDict. (expected: str, got: list)",
            )
        ],
    )
    def test_change_paths_failed(self, strict_c_sd, false_path, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            test = strict_c_sd[false_path]

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


class TestDictPathsStrictSD:

    @pytest.mark.parametrize("path", [
        [('env', 'production')],
        [('env', 'production'), 'database'],
        [('env', 'production'), 'database', 'host'],
        [('env', 'production'), 'database', 'port'],
        [('env', 'production'), 'database', 'pools'],
        [('env', 'production'), 'database', 'replicas'],
        [('env', 'production'), 'database', 'replicas', 1],
        [('env', 'production'), 'database', 'replicas', 1, 'region'],
        [('env', 'production'), 'database', 'replicas', 1, 'status'],
        [('env', 'production'), 'database', 'replicas', 1, 'id'],
        [('env', 'production'), 'database', 'replicas', 2],
        [('env', 'production'), 'database', 'replicas', 2, 'region'],
        [('env', 'production'), 'database', 'replicas', 2, 'status'],
        [('env', 'production'), 'database', 'replicas', 2, 'id'],
        [('env', 'production'), 'database', 'instances'],
        [('env', 'production'), 'database', 'instances', 42],
        [('env', 'production'), 'database', 'instances', 42, 'name'],
        [('env', 'production'), 'database', 'instances', 42, 'max_connections'],
        [('env', 'production'), 'database', 'instances', 42, 'type'],
        [('env', 'production'), 'database', 'instances', 42, 'maintenance_window'],
        [('env', 'production'), 'database', 'instances', 54],
        [('env', 'production'), 'database', 'instances', 54, 'name'],
        [('env', 'production'), 'database', 'instances', 54, 'max_connections'],
        [('env', 'production'), 'database', 'instances', 54, 'type'],
        [('env', 'production'), 'database', 'instances', 54, 'sync_lag'],
        [('env', 'production'), 'api'],
        [('env', 'production'), 'api', 'rate_limit'],
        [('env', 'production'), 'api', 'timeout'],
        [frozenset({'redis', 'cache'}), 'environments', ('env', 'production')],
        [frozenset({'redis', 'cache'}), 'environments', ('env', 'production'), 'cluster_size'],
        [frozenset({'redis', 'cache'}), 'environments', ('env', 'production'), 'persistence'],
        [frozenset({'redis', 'cache'}), 'environments', ('env', 'production'), 'max_memory_policy'],
        ['monitoring', 'dashboards', ('env', 'production')],
        ['monitoring', 'dashboards', ('env', 'production'), 'grafana_url'],
        ['monitoring', 'dashboards', ('env', 'production'), 'alerts'],
        ['monitoring', 'dashboards', ('env', 'production'), 'retention'],
        ['global_settings', ('security', 'encryption'), 'key_rotation', ('env', 'production')],
        ['global_settings', 'networking', 'load_balancer', ('env', 'production')],
        ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'],
        ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'],
        ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval']
    ])
    def test_path(self, strict_c_sd, path):
        assert path in strict_c_sd.dict_paths()


    def test_eq_path(self, strict_c_sd, smooth_c_sd):
        assert strict_c_sd.dict_paths() == smooth_c_sd.dict_paths()

    def test_neq_path(self, strict_c_sd, standard_strict_c_setup):
        assert strict_c_sd.dict_paths() != _StackedDict({}, default_setup=standard_strict_c_setup)

@pytest.mark.skip(reason="To be refactored and moved as DictPaths in core module")
class TestDictSearch:

    @pytest.mark.parametrize("search", [
        [('env', 'production'), ['database', 'api']],
        [('env', 'dev'), ['database', 'api', 'features']],
        [frozenset({'redis', 'cache'}), ['nodes', 'config', 'environments']],
        ['monitoring', [('metrics', 'cpu'), ('logs', 'level'), 'dashboards']],
        ['global_settings', [('security', 'encryption'), 'security', ('networking', 'load_balancer')]],
        ['global_settings', ('security', 'encryption'), ['algorithm', 'key_rotation']]
    ])
    def test_search_path(self, strict_c_sd, search):
        assert search in strict_c_sd.dict_search()

    @pytest.mark.skip(reason="Refactoring Paths and Searches")
    @pytest.mark.parametrize("search", [
        [('env', 'production'), ['database', 'api']],
        [('env', 'dev'), ['database', 'api', 'features']],
        [frozenset({'redis', 'cache'}), ['nodes', 'config', 'environments']],
        ['monitoring', [('metrics', 'cpu'), ('logs', 'level'), 'dashboards']],
        ['global_settings', [('security', 'encryption'), 'security', ('networking', 'load_balancer')]],
        ['global_settings', ('security', 'encryption'), ['algorithm', 'key_rotation']]
    ])
    def test_search_path_from_paths(self, strict_c_sd, search):
        assert search in strict_c_sd.dict_paths().to_search()

    @pytest.mark.skip(reason="Refactoring Paths and Searches")
    def test_complete_covering_paths(self, strict_c_sd):
        assert strict_c_sd.dict_search().is_complete_coverage(
            strict_c_sd.dict_paths()
        )