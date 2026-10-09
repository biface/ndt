"""
Argument accepted by ``update()``: the same as ``dict.update`` (#105).
"""

import pytest

from ndict_tools import NestedDictionary, StackedTypeError


class KeysAndGetItem:
    """Has ``keys()`` and ``__getitem__`` but is not a Mapping."""

    def __init__(self, data):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


def test_update_accepts_keys_and_getitem_object():
    nd = NestedDictionary()
    nd.update(KeysAndGetItem({"a": {"b": 1}, "c": 2}))
    assert nd.to_dict() == {"a": {"b": 1}, "c": 2}
    assert isinstance(nd["a"], NestedDictionary)


def test_update_first_argument_is_positional_only():
    nd = NestedDictionary()
    nd.update(m={"x": 1})
    assert nd.to_dict() == {"m": {"x": 1}}


def test_update_rejects_invalid_argument():
    nd = NestedDictionary()
    with pytest.raises(StackedTypeError):
        nd.update(42)
