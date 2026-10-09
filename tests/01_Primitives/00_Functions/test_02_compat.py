"""
Tests for the private ``_compat`` module (PEP 698 ``override`` shim, #103).
"""

import sys
import typing

from ndict_tools._compat import _override, override
from ndict_tools.core import NestedDictionary
from ndict_tools.serialize import NestedDictionaryEncoder
from ndict_tools.tools import _StackedDict


def test_fallback_marks_and_returns_same_object():
    def method(self):
        return 42

    decorated = _override(method)
    assert decorated is method
    assert decorated.__override__ is True
    assert decorated(None) == 42


def test_fallback_tolerates_objects_without_attributes():
    # Built-in functions refuse new attributes: returned unchanged, no marker.
    assert _override(len) is len
    assert not hasattr(len, "__override__")


def test_override_is_standard_version_when_available():
    if sys.version_info >= (3, 12):
        assert override is typing.override
    else:
        assert override is _override


def test_overriding_methods_are_marked():
    assert _StackedDict.update.__override__ is True
    assert _StackedDict.__getitem__.__override__ is True
    assert NestedDictionary.paths.__override__ is True
    assert NestedDictionaryEncoder.default.__override__ is True
