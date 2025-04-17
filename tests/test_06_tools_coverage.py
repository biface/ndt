"""
Tests specifically designed to cover uncovered lines in tools.py
"""

import pytest

from ndict_tools import NestedDictionary
from ndict_tools.exception import StackedKeyError, StackedValueError
from ndict_tools.tools import _StackedDict, from_dict


def test_from_dict_with_init():
    """Test from_dict function with 'init' parameter (lines 74-77)."""
    # Create a dictionary with custom init parameters
    custom_dict = from_dict(
        {"a": 1, "b": {"c": 2}}, _StackedDict, init={"indent": 4, "default": None}
    )

    assert isinstance(custom_dict, _StackedDict)
    assert custom_dict.indent == 4
    assert custom_dict.default_factory is None
    assert custom_dict["a"] == 1
    assert custom_dict["b"]["c"] == 2


def test_setitem_create_intermediate_keys():
    """Test __setitem__ method with intermediate keys that need to be created (lines 227-234)."""
    sd = _StackedDict(indent=2, default=lambda: 0)

    # Set a value with a hierarchical key where intermediate keys don't exist
    sd[["a", "b", "c", "d"]] = "value"

    # Verify that intermediate dictionaries were created
    assert isinstance(sd["a"], _StackedDict)
    assert isinstance(sd["a"]["b"], _StackedDict)
    assert isinstance(sd["a"]["b"]["c"], _StackedDict)
    assert sd["a"]["b"]["c"]["d"] == "value"

    # Verify that default_factory was properly set for intermediate dictionaries
    assert sd["a"].default_factory == sd.default_factory
    assert sd["a"]["b"].default_factory == sd.default_factory


def test_delitem_cleanup_empty_parents():
    """Test __delitem__ method with cleaning up empty parent dictionaries (lines 290-291)."""
    sd = _StackedDict(indent=2, default=None)

    # Create a nested structure
    sd[["a", "b", "c"]] = "value"

    # Delete the leaf node
    del sd[["a", "b", "c"]]

    # Verify that empty parent dictionaries were removed
    assert "a" not in sd

    # Create a nested structure with multiple values
    sd[["x", "y", "z"]] = "value1"
    sd[["x", "y", "w"]] = "value2"

    # Delete one leaf node
    del sd[["x", "y", "z"]]

    # Verify that parent dictionaries with other children were not removed
    assert "x" in sd
    assert "y" in sd["x"]
    assert "w" in sd["x"]["y"]


def test_pop_with_default_and_cleanup():
    """Test pop method with default values and cleanup (lines 380, 393, 397)."""
    sd = _StackedDict(indent=2, default=None)

    # Create a nested structure
    sd[["a", "b", "c"]] = "value"

    # Test popping a non-existent key with default value (line 380)
    result = sd.pop(["a", "non_existent", "key"], "default_value")
    assert result == "default_value"

    # Test popping a leaf node and cleaning up empty parents (line 393)
    result = sd.pop(["a", "b", "c"])
    assert result == "value"
    assert "a" not in sd  # Empty parents should be removed

    # Create a nested structure again
    sd[["a", "b", "c"]] = "value"

    # Test popping a non-existent final key with default value (line 397)
    result = sd.pop(["a", "b", "non_existent"], "default_value")
    assert result == "default_value"


def test_popitem_stack_traversal():
    """Test popitem method with stack-based traversal (lines 424-438)."""
    sd = _StackedDict(indent=2, default=None)

    # Create a nested structure with multiple branches
    sd["a"] = {"b": {"c": "value1"}}
    sd["x"] = {"y": "value2"}
    sd["m"] = "value3"

    # Test popitem on a dictionary with nested structures
    path, value = sd.popitem()

    # The deepest item should be popped first
    assert path == ["a", "b", "c"]
    assert value == "value1"

    # Test popitem again
    path, value = sd.popitem()
    assert path == ["x", "y"]
    assert value == "value2"

    # Test popitem on a dictionary with only flat keys
    path, value = sd.popitem()
    assert path == ["m"]
    assert value == "value3"


def test_update_with_kwargs():
    """Test update method with kwargs (line 494)."""
    sd = _StackedDict(indent=2, default=None)

    # Update with kwargs
    sd.update(a=1, b=2, c={"d": 3})

    assert sd["a"] == 1
    assert sd["b"] == 2
    assert isinstance(sd["c"], _StackedDict)
    assert sd["c"]["d"] == 3


def test_is_balanced_edge_cases():
    """Test is_balanced method with edge cases (lines 679, 682)."""
    # Create an unbalanced dictionary
    sd = _StackedDict(indent=2, default=None)
    sd["a"] = {"b": {"c": {"d": "value1"}}}
    sd["x"] = "value2"

    # Test is_balanced on an unbalanced dictionary (line 679)
    assert sd.is_balanced() is False

    # Create a dictionary with an empty branch (line 682)
    sd2 = _StackedDict(indent=2, default=None)
    sd2["a"] = _StackedDict(indent=2, default=None)  # Empty nested dictionary

    # Test is_balanced on a dictionary with an empty branch
    assert sd2.is_balanced() is True


def test_ancestors():
    """Test ancestors method (line 700)."""
    sd = _StackedDict(indent=2, default=None)

    # Create a nested structure
    sd["a"] = {"b": {"c": "target_value"}}
    sd["x"] = "other_value"

    # Test finding ancestors of a value
    ancestors = sd.ancestors("target_value")
    assert ancestors == ["a", "b"]

    # Test finding ancestors of a value at the top level
    ancestors = sd.ancestors("other_value")
    assert ancestors == []

    # Test finding ancestors of a non-existent value
    with pytest.raises(StackedValueError):
        sd.ancestors("non_existent_value")


def test_dictpaths_iterate_normal_dict():
    """Test DictPaths._iterate_paths method with normal dict conversion (line 742)."""
    # Create a dictionary with a normal dict (not _StackedDict) as a value
    sd = _StackedDict(indent=2, default=None)
    sd["a"] = {"b": {"c": "value"}}  # This is a normal dict, not a _StackedDict

    # Get all paths
    paths = list(sd.dict_paths())

    # Verify that paths include those from the normal dict
    assert ["a"] in paths
    assert ["a", "b"] in paths
    assert ["a", "b", "c"] in paths
