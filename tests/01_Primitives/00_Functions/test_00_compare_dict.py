import pytest

from ndict_tools.tools import compare_dict


@pytest.mark.parametrize(
    "d1, d2, expected",
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"a": {"b": 2}}, {"a": {"b": 2}}, True),
        ({"a": {"b": 2}}, {"a": {"c": 2}}, False),
        ([1, 2], [1, 2, 3], False),
        ([1, 2], {1, 2}, False),
    ],
)
def test_compare_dict(d1, d2, expected):
    assert compare_dict(d1, d2) == expected
