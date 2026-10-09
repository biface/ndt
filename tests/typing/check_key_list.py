"""
Static check of the return type of ``key_list`` — issue #143.

Not collected by pytest (no ``test_`` prefix); checked by
``basedpyright tests/typing`` (tox env ``typing``). ``key_list`` returns the
leaf paths that contain the key as tuples, like ``unpacked_keys``.
"""

from typing import Any, assert_type

from ndict_tools import NestedDictionary


def check_key_list(nd: NestedDictionary) -> None:
    # Keys are arbitrary hashables, so the element type is Any by design.
    _ = assert_type(
        nd.key_list("a"),
        list[tuple[Any, ...]],  # pyright: ignore[reportExplicitAny]
    )
