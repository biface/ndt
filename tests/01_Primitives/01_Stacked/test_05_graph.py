"""
Testing advanced graph and tree functions
"""

import re

import pytest

from ndict_tools import StackedValueError
from ndict_tools.tools import _StackedDict


@pytest.fixture(scope="function")
def stacked_dict():
    return _StackedDict(
        {
            "h": 400,
            "f": {"g": 200},
            "a": {"e": 100, "b": {"c": 42, "d": 84}},
            (1, 2): 450,
            ("i", "j"): 475,
        },
        default_setup={"indent": 2, "default_factory": None},
    )


@pytest.fixture(scope="function")
def sd_dfs(stacked_dict):
    return list(stacked_dict.dfs())


@pytest.fixture(scope="function")
def sd_bfs(stacked_dict):
    return list(stacked_dict.bfs())


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (("h",), 400)),
        (3, (("f", "g"), 200)),
        (5, (("a", "b", "c"), 42)),
    ],
)
def test_bfs(sd_bfs, index, expected):
    assert sd_bfs[index] == expected


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (["h"], 400)),
        (4, (["a", "e"], 100)),
        (6, (["a", "b", "c"], 42)),
    ],
)
def test_dfs(sd_dfs, index, expected):
    assert sd_dfs[index] == expected


@pytest.mark.parametrize(
    "path, index, expected",
    [
        (["a", "b"], 0, (["c"], 42)),
        (["a", "b"], 1, (["d"], 84)),
    ],
)
def test_dfs_node(stacked_dict, sd_dfs, path, index, expected):
    assert list(stacked_dict.dfs(stacked_dict[path]))[index] == expected


@pytest.mark.parametrize(
    "path, index, expected",
    [
        (["k", "l"], 0, (["k", "l", "h"], 400)),
        (["k", "l"], 4, (["k", "l", "a", "e"], 100)),
        (["k", "l"], 6, (["k", "l", "a", "b", "c"], 42)),
    ],
)
def test_dfs(stacked_dict, path, index, expected):
    assert list(stacked_dict.dfs(path=path))[index] == expected


@pytest.mark.parametrize(
    "value, exp_assert",
    [
        (400, True),
        (200, True),
        (43, False),
        (84, True),
        (451, False),
        (475, True),
        (500, False),
    ],
)
def test_leaves(stacked_dict, value, exp_assert):
    assert (value in stacked_dict.leaves()) is exp_assert


def test_height(stacked_dict):
    assert stacked_dict.height() == 3


def test_size(stacked_dict):
    assert stacked_dict.size() == 7


def test_balanced(stacked_dict):
    assert stacked_dict.is_balanced() == False
    assert stacked_dict[["a", "b"]].is_balanced() == True


# Additional tests to cover _StackedDict.is_balanced comprehensively


def test_is_balanced_empty_dict(standard_strict_f_setup):
    nd0 = _StackedDict({}, default_setup=standard_strict_f_setup)
    assert nd0.is_balanced() is True


def test_is_balanced_single_leaf(standard_strict_f_setup):

    nd1 = _StackedDict({"a": 1}, default_setup=standard_strict_f_setup)
    assert nd1.is_balanced() is True


def test_is_balanced_height_diff_one_true(standard_strict_f_setup):

    # Left subtree height=1 (a -> x), right subtree height=2 (b -> y -> z)
    nd2 = _StackedDict(
        {"a": {"x": 1}, "b": {"y": {"z": 2}}}, default_setup=standard_strict_f_setup
    )
    assert nd2.is_balanced() is True


def test_is_balanced_height_diff_gt_one_false(standard_strict_f_setup):
    # Left subtree height=3 (a -> x -> y -> z), right subtree height=0 (b)
    nd3 = _StackedDict(
        {"a": {"x": {"y": {"z": 1}}}, "b": 1}, default_setup=standard_strict_f_setup
    )
    assert nd3.is_balanced() is False


def test_is_balanced_with_tuple_keys(standard_strict_f_setup):
    nd4 = _StackedDict(
        {("env", "dev"): {"a": 1}, ("env", "prod"): {"b": {"c": 2}}},
        default_setup=standard_strict_f_setup,
    )
    # Heights differ by 1 -> balanced
    assert nd4.is_balanced() is True


@pytest.mark.parametrize(
    "value, expected_paths", [(42, ["a", "b"]), (200, ["f"]), (400, [])]
)
def test_ancestors(stacked_dict, value, expected_paths):
    assert stacked_dict.ancestors(value) == expected_paths


def test_ancestors_failed(stacked_dict):
    with pytest.raises(
        StackedValueError,
        match=re.escape("Value 350 not found in the dictionary. (value: 350)"),
    ):
        stacked_dict.ancestors(350)
