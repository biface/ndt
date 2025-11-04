import re

import pytest

from ndict_tools.exception import NestedDictionaryException, StackedDictionaryError


@pytest.mark.parametrize(
    "msg, code, path, result",
    [
        ("Test error message", 0, [], ["Test error message", 0, []]),
        ("Another test error message", 42, [], ["Another test error message", 42, []]),
        (
            "Test error message with path",
            54,
            ["path", "to", "key"],
            [
                "Test error message with path (at path: path | to | key)",
                54,
                ["path", "to", "key"],
            ],
        ),
        (
            "",
            0,
            ["path", "to", "key"],
            ["Error at path: path | to | key", 0, ["path", "to", "key"]],
        ),
    ],
)
def test_basic_stacked_dictionary_error(msg, code, path, result):
    """Test basic functionality of StackedDictionaryError."""
    # Test with just a message
    error = StackedDictionaryError(msg, code, path)
    assert str(error) == result[0]
    assert error.error_code == result[1]
    assert error.path == result[2]


@pytest.mark.parametrize(
    "msg, code, path, result",
    [
        ("Test error message", 0, [], ["Test error message", 0, []]),
        ("Another test error message", 42, [], ["Another test error message", 42, []]),
        (
            "Test error message with path",
            54,
            ["path", "to", "key"],
            [
                "Test error message with path (at path: path | to | key)",
                54,
                ["path", "to", "key"],
            ],
        ),
        (
            "",
            0,
            ["path", "to", "key"],
            ["Error at path: path | to | key", 0, ["path", "to", "key"]],
        ),
    ],
)
def test_basic_nested_dictionary_exception(msg, code, path, result):
    """Test NestedDictionaryException which inherits from StackedDictionaryError."""
    # Test with just a message
    error = NestedDictionaryException(msg, code, path)
    assert str(error) == result[0]
    assert error.error_code == result[1]
    assert error.path == result[2]
