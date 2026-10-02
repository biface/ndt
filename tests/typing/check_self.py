"""
Static check of the ``Self`` return types — issue #75.

This file is not collected by pytest (no ``test_`` prefix). It is checked by
``basedpyright tests/typing`` (tox env ``typing``): ``assert_type`` reports
an error when the inferred type differs, so a regression to a fixed return
type such as ``_StackedDict`` fails the check. Nothing here is executed.
"""

from typing import assert_type

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.tools import _HKey  # pyright: ignore[reportPrivateUsage]


def check_nested(nd: NestedDictionary) -> None:
    _ = assert_type(nd.copy(), NestedDictionary)
    _ = assert_type(nd.deepcopy(), NestedDictionary)
    _ = assert_type(nd.__copy__(), NestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), NestedDictionary)
    _ = assert_type(NestedDictionary.from_dict({}, default_setup={}), NestedDictionary)


def check_strict(nd: StrictNestedDictionary) -> None:
    _ = assert_type(nd.copy(), StrictNestedDictionary)
    _ = assert_type(nd.deepcopy(), StrictNestedDictionary)
    _ = assert_type(nd.__copy__(), StrictNestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), StrictNestedDictionary)
    _ = assert_type(
        StrictNestedDictionary.from_dict({}, default_setup={}), StrictNestedDictionary
    )


def check_smooth(nd: SmoothNestedDictionary) -> None:
    _ = assert_type(nd.copy(), SmoothNestedDictionary)
    _ = assert_type(nd.deepcopy(), SmoothNestedDictionary)
    _ = assert_type(nd.__copy__(), SmoothNestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), SmoothNestedDictionary)
    _ = assert_type(
        SmoothNestedDictionary.from_dict({}, default_setup={}), SmoothNestedDictionary
    )


class _Node(_HKey):
    """Subclass used to check that build_forest returns the calling class."""


def check_build_forest() -> None:
    _ = assert_type(_HKey.build_forest({}), _HKey)
    _ = assert_type(_Node.build_forest({}), _Node)
