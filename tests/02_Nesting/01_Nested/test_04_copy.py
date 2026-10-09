"""
Copy protocol on the three public variants.

Regression tests for #127: ``copy.deepcopy()`` raised ``TypeError`` because
``__deepcopy__`` did not accept the ``memo`` argument, and the deep copy
shared mutable leaf values with the original.
"""

import copy

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)

VARIANTS = [NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary]


@pytest.fixture(params=VARIANTS, ids=lambda cls: cls.__name__)
def variant(request):
    return request.param


@pytest.fixture
def source(variant):
    return variant({"a": {"b": [1, 2], "c": 3}, "d": {"e": {"f": "g"}}})


def test_deepcopy_keeps_class_and_setup(source):
    result = copy.deepcopy(source)
    assert result == source
    assert result is not source
    assert type(result) is type(source)
    assert type(result["a"]) is type(source["a"])
    assert result.default_setup == source.default_setup


def test_deepcopy_is_independent(source):
    result = copy.deepcopy(source)
    result["a"]["b"].append(3)
    result["d"]["e"]["f"] = "changed"
    assert source["a"]["b"] == [1, 2]
    assert source["d"]["e"]["f"] == "g"


def test_deepcopy_method_matches_protocol(source):
    result = source.deepcopy()
    assert result == copy.deepcopy(source)
    result["a"]["b"].append(3)
    assert source["a"]["b"] == [1, 2]


def test_deepcopy_preserves_shared_substructure(variant):
    shared = variant({"x": [1]})
    source = variant()
    source["left"] = shared
    source["right"] = shared
    assert source["left"] is source["right"]

    result = copy.deepcopy(source)
    assert result["left"] is result["right"]
    assert result["left"] is not shared


def test_deepcopy_handles_self_reference(variant):
    source = variant({"a": 1})
    source["self"] = source

    result = copy.deepcopy(source)
    assert result["self"] is result
    assert result is not source


def test_copy_protocol_is_shallow(source):
    result = copy.copy(source)
    assert type(result) is type(source)
    assert result["a"] is source["a"]
