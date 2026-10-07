"""
The public API names the classes the user works with, never the private
base classes (#138, #155).
"""

import inspect
import re

import pytest

from ndict_tools import (
    CompactPathsView,
    NestedDictionary,
    SmoothNestedDictionary,
    StackedIndexError,
    StackedTypeError,
    StrictNestedDictionary,
)


class Inventory(NestedDictionary):
    """A user subclass: messages follow it too."""


CLASSES = [NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary, Inventory]


@pytest.mark.parametrize("cls", CLASSES)
def test_popitem_empty_names_the_class(cls):
    with pytest.raises(
        StackedIndexError, match=re.escape(f"popitem(): {cls.__name__} is empty")
    ):
        cls().popitem()


@pytest.mark.parametrize("cls", CLASSES)
def test_nested_list_key_names_the_class(cls):
    nd = cls({"a": 1})
    expected = f"Nested lists are not allowed as keys in {cls.__name__}."
    with pytest.raises(StackedTypeError, match=re.escape(expected)):
        nd[["a", ["b"]]] = 1
    with pytest.raises(StackedTypeError, match=re.escape(expected)):
        _ = nd[["a", ["b"]]]


def test_structure_message_names_no_private_class():
    view = NestedDictionary({"a": 1}).compact_paths()
    with pytest.raises(TypeError) as excinfo:
        view.structure = 42
    message = str(excinfo.value)
    assert "Expected a nested dictionary, a dict or a list." in message
    assert "_StackedDict" not in message and "_HKey" not in message


def test_compact_view_to_compact_returns_a_public_view():
    view = NestedDictionary({"a": {"b": 1}, "c": 2}).compact_paths()
    compact = view.to_compact()
    assert type(compact) is CompactPathsView
    assert compact is not view
    assert compact.structure == view.structure


def test_pop_signature_shows_a_readable_default():
    parameter = inspect.signature(NestedDictionary.pop).parameters["default"]
    assert repr(parameter.default) == "<no default>"
    assert "<no default>" in str(inspect.signature(NestedDictionary.pop))
