"""
Static check of the ``Self`` return types — issue #75.

This file is not collected by pytest (no ``test_`` prefix). It is checked by
``basedpyright tests/typing`` (tox env ``typing``): ``assert_type`` reports
an error when the inferred type differs, so a regression to a fixed return
type such as ``_StackedDict`` fails the check. Nothing here is executed.
"""

from pathlib import Path
from typing import assert_type

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.tools import _HKey  # pyright: ignore[reportPrivateUsage]


def check_nested(nd: NestedDictionary, path: Path) -> None:
    _ = assert_type(nd.copy(), NestedDictionary)
    _ = assert_type(nd.deepcopy(), NestedDictionary)
    _ = assert_type(nd.__copy__(), NestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), NestedDictionary)
    _ = assert_type(NestedDictionary.from_dict({}, default_setup={}), NestedDictionary)
    _ = assert_type(
        NestedDictionary.from_json(path, default_setup={}), NestedDictionary
    )
    _ = assert_type(NestedDictionary.from_pickle(path), NestedDictionary)


def check_strict(nd: StrictNestedDictionary, path: Path) -> None:
    _ = assert_type(nd.copy(), StrictNestedDictionary)
    _ = assert_type(nd.deepcopy(), StrictNestedDictionary)
    _ = assert_type(nd.__copy__(), StrictNestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), StrictNestedDictionary)
    _ = assert_type(
        StrictNestedDictionary.from_dict({}, default_setup={}), StrictNestedDictionary
    )
    _ = assert_type(
        StrictNestedDictionary.from_json(path, default_setup={}), StrictNestedDictionary
    )
    _ = assert_type(StrictNestedDictionary.from_pickle(path), StrictNestedDictionary)


def check_smooth(nd: SmoothNestedDictionary, path: Path) -> None:
    _ = assert_type(nd.copy(), SmoothNestedDictionary)
    _ = assert_type(nd.deepcopy(), SmoothNestedDictionary)
    _ = assert_type(nd.__copy__(), SmoothNestedDictionary)
    _ = assert_type(nd.__deepcopy__({}), SmoothNestedDictionary)
    _ = assert_type(
        SmoothNestedDictionary.from_dict({}, default_setup={}), SmoothNestedDictionary
    )
    _ = assert_type(
        SmoothNestedDictionary.from_json(path, default_setup={}), SmoothNestedDictionary
    )
    _ = assert_type(SmoothNestedDictionary.from_pickle(path), SmoothNestedDictionary)


class _Node(_HKey):
    """Subclass used to check that build_forest returns the calling class."""


def check_build_forest() -> None:
    _ = assert_type(_HKey.build_forest({}), _HKey)
    _ = assert_type(_Node.build_forest({}), _Node)
