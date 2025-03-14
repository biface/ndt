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
