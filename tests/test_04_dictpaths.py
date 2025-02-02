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
        (1, ["f", "g"]),
        (2, ["a", "e"]),
        (3, ["a", "b", "c"]),
        (4, ["a", "b", "d"]),
        (5, [(1, 2)]),
        (6, [("i", "j")]),
    ],
)
def test_paths(index, expected_path):
    assert paths[index] == expected_path


@pytest.mark.parametrize(
    "path, expected",
    [
        (["h"], True),
        (["f", "g"], True),
        ([(1, 2)], True),
        ([("i", "k")], False),
        (["y"], False),
    ],
)
def test_paths_content(path, expected):
    assert nd.dict_paths().__contains__(path) == expected
