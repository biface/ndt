"""
    Testing advanced graph and tree functions
"""

import pytest
from ndict_tools import NestedDictionary

nd = NestedDictionary(
    {
        "h": 400,
        "f": {"g": 200},
        "a": {"e": 100, "b": {"c": 42, "d": 84}},
        (1, 2): 450,
        ("i", "j"): 475,
    },
    indent=2,
    strict=True,
)

dfs = list(nd.dfs())
bfs = list(nd.bfs())


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (("h",), 400)),
        (3, (("f", "g"), 200)),
        (5, (("a", "b", "c"), 42)),
    ],
)
def test_bfs(index, expected):
    assert bfs[index] == expected


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (["h"], 400)),
        (4, (["a", "e"], 100)),
        (6, (["a", "b", "c"], 42)),
    ],
)
def test_dfs(index, expected):
    assert dfs[index] == expected


dfs_n = list(nd.dfs(nd[["a", "b"]]))


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (["c"], 42)),
        (1, (["d"], 84)),
    ],
)
def test_dfs_node(index, expected):
    assert dfs_n[index] == expected


dfs_p = list(nd.dfs(path=["k", "l"]))


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, (["k", "l", "h"], 400)),
        (4, (["k", "l", "a", "e"], 100)),
        (6, (["k", "l", "a", "b", "c"], 42)),
    ],
)
def test_dfs(index, expected):
    assert dfs_p[index] == expected
