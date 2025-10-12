import re

import pytest


class TestHKeyTree:

    def test_init(self, key_tree):
        assert key_tree.has_children()
        assert key_tree.get_statistics() == {
            "total_nodes": 113,
            "leaf_count": 74,
            "max_depth": 5,
            "avg_branching_factor": 2.87,
            "total_paths": 112,
            "levels": 5,
        }

    def test_balance(self, key_tree):
        assert not key_tree.is_balanced()
        assert key_tree.get_balance_factor() != 0

    @pytest.mark.parametrize(
        "keys, children",
        [
            ([("env", "production")], 2),
            ([("env", "production"), "database"], 5),
            ([("env", "production"), "database", "host"], 0),
            ([("env", "production"), "database", "replicas"], 2),
            ([("env", "production"), "database", "replicas", 2], 3),
            ([("env", "production"), "database", "replicas", 2, "status"], 0),
            (["monitoring"], 3),
            (["monitoring", ("logs", "level")], 2),
            (["monitoring", ("logs", "level"), "debug"], 0),
        ],
    )
    def test_get_child(self, key_tree, keys, children):
        hkey = key_tree
        for key in keys:
            hkey = hkey.get_child(key)
        if children > 0:
            assert hkey.has_children()
            assert len(hkey.children) == children
        else:
            assert hkey.is_leaf()

    @pytest.mark.parametrize(
        "path, children",
        [
            ([("env", "dev"), "features", "flags"], 2),
            ([("env", "dev"), "features", "flags", "enable_logging"], 0),
            ([frozenset({"cache", "redis"}), "nodes"], 0),
            ([frozenset({"cache", "redis"}), "config"], 2),
            ([frozenset({"cache", "redis"}), "config", "ttl"], 0),
            ([frozenset({"cache", "redis"}), "config", "memory"], 0),
        ],
    )
    def test_get_children_by_path(self, key_tree, path, children):
        if children > 0:
            assert key_tree.find_by_path(path).has_children()
            assert len(key_tree.find_by_path(path)) == children
        else:
            assert key_tree.find_by_path(path).is_leaf()

    @pytest.mark.parametrize(
        "path",
        [
            [("env", "dev"), "features", "flags"],
            [("env", "dev"), "features", "flags", "enable_logging"],
            [frozenset({"cache", "redis"}), "nodes"],
            [frozenset({"cache", "redis"}), "config"],
            [frozenset({"cache", "redis"}), "config", "ttl"],
            [frozenset({"cache", "redis"}), "config", "memory"],
            [("env", "dev"), "features", "flags"],
            [("env", "dev"), "features", "flags", "enable_logging"],
            [frozenset({"cache", "redis"}), "nodes"],
            [frozenset({"cache", "redis"}), "config"],
            [frozenset({"cache", "redis"}), "config", "ttl"],
            [frozenset({"cache", "redis"}), "config", "memory"],
        ],
    )
    def test_get_path_from_child(self, key_tree, path):
        assert key_tree.find_by_path(path).get_path() == path

    @pytest.mark.parametrize(
        "path, depth, max_depth",
        [
            (None, 0, 5),
            ([("env", "dev")], 0, 4),
            ([("env", "dev"), "features", "flags"], 2, 1),
            ([("env", "dev"), "features", "flags", "enable_logging"], 3, 0),
            ([frozenset({"cache", "redis"}), "nodes"], 1, 0),
            ([frozenset({"cache", "redis"}), "config"], 1, 1),
            ([frozenset({"cache", "redis"}), "config", "ttl"], 2, 0),
            ([frozenset({"cache", "redis"}), "config", "memory"], 2, 0),
        ],
    )
    def test_get_children_by_path(self, key_tree, path, depth, max_depth):
        if path is not None:
            assert key_tree.find_by_path(path).get_depth() == depth
            assert key_tree.find_by_path(path).get_max_depth() == max_depth
        else:
            assert key_tree.get_depth() == depth
            assert key_tree.get_max_depth() == max_depth

    @pytest.mark.parametrize(
        "parent_path, child_key",
        [
            (["global_settings"], "security"),
            (["global_settings", "security"], "encryption"),
            (["global_settings"], ("security", "encryption")),
            (["global_settings", ("security", "encryption")], "algorithm"),
            (["global_settings", ("security", "encryption")], "key_rotation"),
            (
                ["global_settings", ("security", "encryption"), "key_rotation"],
                ("env", "production"),
            ),
            (
                ["global_settings", ("security", "encryption"), "key_rotation"],
                ("env", "dev"),
            ),
        ],
    )
    def test_get_descendants(self, key_tree, parent_path, child_key):
        assert child_key in [
            child.key for child in key_tree.find_by_path(parent_path).get_descendants()
        ]

    @pytest.mark.parametrize(
        "key, children",
        [("global_settings", 3), (("env", "production"), 2), ("monitoring", 3)],
    )
    def test_get_simple_children_by_key(self, key_tree, key, children):
        assert children == len(key_tree.find_by_key(key))

    def test_get_simple_children_by_key_failed(self, key_tree):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "_HKey.find_by_key() missing 1 required positional argument: 'key'"
            ),
        ):
            key_tree.find_by_key()

    @pytest.mark.parametrize(
        "key, children",
        [(("env", "dev"), [3, 3, 3, 0, 3]), (("env", "production"), [2, 3, 3, 0, 3])],
    )
    def test_get_multiple_children_by_key(self, key_tree, key, children):
        for n_child in children:
            assert n_child in [
                len(child.children)
                for child in key_tree.find_by_key(key, find_all=True)
            ]
        assert children == [
            len(child.children) for child in key_tree.find_by_key(key, find_all=True)
        ]
