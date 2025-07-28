"""
Tests for real-world scenarios where exceptions might be raised in the ndict_tools package.
"""

import pytest

from ndict_tools import NestedDictionary
from ndict_tools.exception import (
    StackedAttributeError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)
from ndict_tools.tools import _StackedDict


def test_key_error_in_nested_access():
    """Test StackedKeyError when accessing a non-existent nested key."""
    nd = NestedDictionary(
        {"a": {"b": 1}}, default_setup={"indent": 0, "default_factory": None}
    )

    # Access a non-existent key at the top level
    with pytest.raises(KeyError) as excinfo:
        value = nd["non_existent"]

    assert "non_existent" in str(excinfo.value)

    # Access a non-existent key in a nested dictionary
    with pytest.raises(KeyError) as excinfo:
        value = nd["a"]["non_existent"]

    assert "non_existent" in str(excinfo.value)

    # Access a non-existent nested path
    with pytest.raises(KeyError) as excinfo:
        value = nd["x"]["y"]["z"]

    assert "x" in str(excinfo.value)


def test_attribute_error_in_from_dict():
    """Test StackedAttributeError when setting an invalid attribute."""
    from ndict_tools.tools import from_dict

    # Try to set an attribute that doesn't exist
    with pytest.raises(StackedAttributeError) as excinfo:
        nd = from_dict(
            {"a": 1, "b": 2},
            NestedDictionary,
            default_setup={
                "indent": 2,
                "default_factory": None,
                "non_existent_attr": True,
            },
        )

    assert "non_existent_attr" in str(excinfo.value)
    assert excinfo.value.attribute == "non_existent_attr"


def test_type_error_with_nested_lists():
    """Test StackedTypeError when using nested lists as keys."""
    sd = _StackedDict(default_setup={"indent": 0, "default_factory": None})

    # Try to use a nested list as a key
    with pytest.raises(StackedTypeError) as excinfo:
        sd[[1, [2, 3]]] = "value"

    assert "Nested lists are not allowed" in str(excinfo.value)
    assert excinfo.value.expected_type is str
    assert excinfo.value.actual_type is list

    # Try to access with a nested list
    sd[[1, 2, 3]] = "value"
    with pytest.raises(StackedTypeError) as excinfo:
        value = sd[[1, [2, 3]]]

    assert "Nested lists are not allowed" in str(excinfo.value)


def test_value_error_in_ancestors():
    """Test StackedValueError when searching for a non-existent value."""
    nd = NestedDictionary({"a": {"b": 1}, "c": 2})

    # Try to find ancestors of a non-existent value
    with pytest.raises(StackedValueError) as excinfo:
        ancestors = nd.ancestors(999)

    assert "999" in str(excinfo.value)
    assert excinfo.value.value == 999


def test_index_error_in_popitem():
    """Test StackedIndexError when calling popitem on an empty dictionary."""
    nd = NestedDictionary()

    # Try to pop an item from an empty dictionary
    with pytest.raises(StackedIndexError) as excinfo:
        nd.popitem()

    assert "empty" in str(excinfo.value).lower()


def test_key_error_in_pop():
    """Test StackedKeyError when popping a non-existent key."""
    nd = NestedDictionary({"a": {"b": 1}, "c": 2})

    # Try to pop a non-existent key
    with pytest.raises(StackedKeyError) as excinfo:
        nd.pop(["a", "non_existent"])

    assert "non_existent" in str(excinfo.value)
    assert excinfo.value.key == "non_existent"
    assert excinfo.value.path == ["a"]

    # Try to pop a non-existent path
    with pytest.raises(StackedKeyError) as excinfo:
        nd.pop(["x", "y", "z"])

    assert "x" in str(excinfo.value)


def test_type_error_in_is_key():
    """Test StackedKeyError when using a list in is_key method."""
    nd = NestedDictionary({"a": {"b": 1}, "c": 2})

    # Try to use a list in is_key method
    with pytest.raises(StackedKeyError) as excinfo:
        nd.is_key(["a", "b"])

    assert "atomic keys" in str(excinfo.value).lower()
    assert excinfo.value.key == ["a", "b"]


def test_multiple_exceptions_in_sequence():
    """Test multiple exceptions raised in sequence."""
    nd = NestedDictionary()

    # First exception: StackedIndexError
    with pytest.raises(StackedIndexError):
        nd.popitem()

    # Add some data
    nd["a"] = {"b": 1}

    # Second exception: StackedKeyError
    with pytest.raises(StackedKeyError):
        nd.pop(["a", "non_existent"])

    # Third exception: StackedTypeError
    with pytest.raises(StackedTypeError):
        nd[[1, [2, 3]]] = "value"
