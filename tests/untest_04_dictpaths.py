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

paths = list(nd.dict_paths())


@pytest.mark.parametrize(
    "index, expected_path",
    [
        (0, ["h"]),
        (1, ["f"]),
        (2, ["f", "g"]),
        (3, ["a"]),
        (4, ["a", "e"]),
        (5, ["a", "b"]),
        (6, ["a", "b", "c"]),
        (7, ["a", "b", "d"]),
        (8, [(1, 2)]),
        (9, [("i", "j")]),
    ],
)
def test_paths(index, expected_path):
    assert paths[index] == expected_path


@pytest.mark.parametrize(
    "path, expected",
    [
        (["h"], True),
        (["f", "g"], True),
        (["a"], True),
        (["a", "b"], True),
        (["a", "b", "c"], True),
        ([(1, 2)], True),
        ([(1, 2), (3, 4)], False),
        ([("i", "k")], False),
        (["y"], False),
    ],
)
def test_paths_content(path, expected):
    assert nd.dict_paths().__contains__(path) == expected


def test_dictpaths_equality_self_and_order_independence():
    dp = nd.dict_paths()
    # Self equality
    assert dp == dp
    # Build a list of the same paths but shuffled order
    same_paths = list(dp)
    # reverse to ensure order difference
    same_paths_reversed = list(reversed(same_paths))
    assert dp == same_paths_reversed


def test_dictpaths_equality_with_iterables_and_tuple_keys():
    dp = nd.dict_paths()
    # Explicit construction of expected paths (from the fixture above)
    expected_paths = [
        ["h"],
        ["f"],
        ["f", "g"],
        ["a"],
        ["a", "e"],
        ["a", "b"],
        ["a", "b", "c"],
        ["a", "b", "d"],
        [(1, 2)],
        [("i", "j")],
    ]
    # Different ordering must not affect equality
    assert dp == expected_paths


@pytest.mark.parametrize(
    "modifier, expected_equal",
    [
        (lambda paths: paths[:-1], False),  # missing one path
        (lambda paths: paths + [["z"]], False),  # extra path
    ],
)
def test_dictpaths_inequality_cases(modifier, expected_equal):
    dp = nd.dict_paths()
    paths_list = list(dp)
    modified = modifier(paths_list)
    assert (dp == modified) is expected_equal
    assert (dp != modified) is (not expected_equal)


def test_dictpaths_comparison_with_non_iterable():
    dp = nd.dict_paths()

    class NotIterable:
        __iter__ = None

    # Comparing to a non-iterable should not raise and should be unequal
    assert (dp == NotIterable()) is False
    assert (dp != NotIterable()) is True
