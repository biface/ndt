"""
Testing complementary dictionary functions
"""

import pytest

from ndict_tools import NestedDictionary
from ndict_tools.exception import StackedIndexError, StackedKeyError

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


@pytest.mark.parametrize(
    "path, expected", [(["a", "b", "c"], 42), ((1, 2), 450), ("h", 400)]
)
def test_pop_function(path, expected):
    assert nd.pop(path) == expected


@pytest.mark.parametrize(
    "expected",
    [
        ([("i", "j")], 475),
        (["a", "b", "d"], 84),
    ],
)
def test_popitem(expected):
    assert nd.popitem() == expected


def test_pop_function_invalid_path():
    with pytest.raises(StackedKeyError):
        nd.pop(["a", "b", "e"])


def test_pop_function_default():
    assert nd.pop(1234, None) is None


def test_popitem_empty_stack():
    nd.pop(["f"])
    nd.pop(["a"])
    with pytest.raises(StackedIndexError):
        nd.popitem()
