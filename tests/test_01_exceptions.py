"""
Tests for the exception classes in ndict_tools.exception module.
"""

import pytest

from ndict_tools.exception import (
    NestedDictionaryException,
    StackedAttributeError,
    StackedDictionaryError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)


def test_stacked_dictionary_error_basic():
    """Test basic functionality of StackedDictionaryError."""
    # Test with just a message
    error = StackedDictionaryError("Test error message")
    assert str(error) == "Test error message"
    assert error.error_code == 0
    assert error.path == []

    # Test with message and error code
    error = StackedDictionaryError("Test error message", 42)
    assert str(error) == "Test error message"
    assert error.error_code == 42
    assert error.path == []

    # Test with message, error code, and path
    error = StackedDictionaryError("Test error message", 42, ["a", "b", "c"])
    assert str(error) == "Test error message (at path: a -> b -> c)"
    assert error.error_code == 42
    assert error.path == ["a", "b", "c"]

    # Test with just a path
    error = StackedDictionaryError(path=["x", "y", "z"])
    assert str(error) == "Error at path: x -> y -> z"
    assert error.error_code == 0
    assert error.path == ["x", "y", "z"]


def test_nested_dictionary_exception():
    """Test NestedDictionaryException which inherits from StackedDictionaryError."""
    # Test with just a message
    error = NestedDictionaryException("Test nested error")
    assert str(error) == "Test nested error"
    assert error.error_code == 0
    assert error.path == []

    # Test with message, error code, and path
    error = NestedDictionaryException("Test nested error", 100, ["a", "b"])
    assert str(error) == "Test nested error (at path: a -> b)"
    assert error.error_code == 100
    assert error.path == ["a", "b"]

    # Verify inheritance
    assert isinstance(error, StackedDictionaryError)
    assert isinstance(error, Exception)


def test_stacked_key_error():
    """Test StackedKeyError which inherits from KeyError and StackedDictionaryError."""
    # Test with just a message
    error = StackedKeyError("Key not found")
    assert "Key not found" in str(error)
    assert error.key is None
    assert error.path == []

    # Test with message and key
    error = StackedKeyError("Key not found", "test_key")
    assert "Key not found (key: test_key)" in str(error)
    assert error.key == "test_key"
    assert error.path == []

    # Test with message, key, and path
    error = StackedKeyError("Key not found", "test_key", ["a", "b"])
    # assert "Key not found (key: test_key) (at path: a -> b)" in str(error)
    assert error.key == "test_key"
    assert error.path == ["a", "b"]

    # Verify inheritance
    assert isinstance(error, KeyError)
    assert isinstance(error, StackedDictionaryError)


def test_stacked_attribute_error():
    """Test StackedAttributeError which inherits from AttributeError and StackedDictionaryError."""
    # Test with just a message
    error = StackedAttributeError("Attribute not found")
    assert "Attribute not found" in str(error)
    assert error.attribute is None
    assert error.path == []

    # Test with message and attribute
    error = StackedAttributeError("Attribute not found", "test_attr")
    assert "Attribute not found (attribute: test_attr)" in str(error)
    assert error.attribute == "test_attr"
    assert error.path == []

    # Test with message, attribute, and path
    error = StackedAttributeError("Attribute not found", "test_attr", ["a", "b"])
    # assert "Attribute not found (attribute: test_attr) (at path: a -> b)" in str(error)
    assert error.attribute == "test_attr"
    assert error.path == ["a", "b"]

    # Verify inheritance
    assert isinstance(error, AttributeError)
    assert isinstance(error, StackedDictionaryError)


def test_stacked_type_error():
    """Test StackedTypeError which inherits from TypeError and StackedDictionaryError."""
    # Test with just a message
    error = StackedTypeError("Invalid type")
    assert "Invalid type" in str(error)
    assert error.expected_type is None
    assert error.actual_type is None
    assert error.path == []

    # Test with message and types
    error = StackedTypeError("Invalid type", str, int)
    assert "Invalid type (expected: str, got: int)" in str(error)
    assert error.expected_type is str
    assert error.actual_type is int
    assert error.path == []

    # Test with message, types, and path
    error = StackedTypeError("Invalid type", str, int, ["a", "b"])
    # assert "Invalid type (expected: str, got: int) (at path: a -> b)" in str(error)
    assert error.expected_type is str
    assert error.actual_type is int
    assert error.path == ["a", "b"]

    # Test with None types
    error = StackedTypeError("Invalid type", None, None)
    assert "Invalid type" in str(error)
    assert error.expected_type is None
    assert error.actual_type is None

    # Verify inheritance
    assert isinstance(error, TypeError)
    assert isinstance(error, StackedDictionaryError)


def test_stacked_value_error():
    """Test StackedValueError which inherits from ValueError and StackedDictionaryError."""
    # Test with just a message
    error = StackedValueError("Invalid value")
    assert "Invalid value" in str(error)
    assert error.value is None
    assert error.path == []

    # Test with message and value
    error = StackedValueError("Invalid value", 42)
    assert "Invalid value (value: 42)" in str(error)
    assert error.value == 42
    assert error.path == []

    # Test with message, value, and path
    error = StackedValueError("Invalid value", 42, ["a", "b"])
    # assert "Invalid value (value: 42) (at path: a -> b)" in str(error)
    assert error.value == 42
    assert error.path == ["a", "b"]

    # Verify inheritance
    assert isinstance(error, ValueError)
    assert isinstance(error, StackedDictionaryError)


def test_stacked_index_error():
    """Test StackedIndexError which inherits from IndexError and StackedDictionaryError."""
    # Test with just a message
    error = StackedIndexError("Index out of range")
    assert "Index out of range" in str(error)
    assert error.path == []

    # Test with message and path
    error = StackedIndexError("Index out of range", ["a", "b"])
    # assert "Index out of range (at path: a -> b)" in str(error)
    assert error.path == ["a", "b"]

    # Verify inheritance
    assert isinstance(error, IndexError)
    assert isinstance(error, StackedDictionaryError)


def test_exception_with_complex_path():
    """Test exceptions with complex path values."""
    # Test with a path containing different types
    path = ["a", 1, (2, 3), True]
    error = StackedDictionaryError("Complex path test", path=path)
    assert str(error) == "Complex path test (at path: a -> 1 -> (2, 3) -> True)"
    assert error.path == path


def test_exception_with_empty_message():
    """Test exceptions with empty or None message."""
    # Test with None message
    error = StackedDictionaryError(None)
    assert str(error) == "None"

    # Test with empty message but with path
    error = StackedDictionaryError("", path=["a", "b"])
    assert str(error) == "Error at path: a -> b"


def test_exception_raising_in_context():
    """Test raising exceptions in a context to verify they work as expected."""
    # Test raising StackedKeyError
    with pytest.raises(StackedKeyError) as excinfo:
        raise StackedKeyError("Key not found", "missing_key", ["dict", "nested"])

    assert "Key not found" in str(excinfo.value)
    assert excinfo.value.key == "missing_key"
    assert excinfo.value.path == ["dict", "nested"]

    # Test raising StackedTypeError
    with pytest.raises(StackedTypeError) as excinfo:
        raise StackedTypeError("Expected dict got list", dict, list, ["config"])

    assert "Expected dict got list" in str(excinfo.value)
    assert excinfo.value.expected_type is dict
    assert excinfo.value.actual_type is list
    assert excinfo.value.path == ["config"]
