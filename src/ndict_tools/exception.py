"""
This module provides specific exception classes for nested dictionaries.
These exceptions extend the standard **Exception**, **KeyError** and **AttributeError** classes
to provide more context and better error handling for nested dictionary operations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union


class StackedDictionaryError(Exception):
    """
    Base exception class for all stacked dictionary errors.

    This is the parent class for all exceptions related to stacked dictionaries.
    It provides context about the error including an optional error code.
    """

    def __init__(
        self, message: str = None, error_code: int = 0, path: List[Any] = None
    ) -> None:
        """
        Initialize a StackedDictionaryError.

        :param message: A message describing the error.
        :type message: str
        :param error_code: An integer code identifying the error type.
        :type error_code: int
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        self.error_code = error_code
        self.path = path or []

        # Add path information to the message if available
        if path:
            path_str = " | ".join(str(k) for k in path)
            if message:
                message = f"{message} (at path: {path_str})"
            else:
                message = f"Error at path: {path_str}"

        super().__init__(message)


class NestedDictionaryException(StackedDictionaryError):
    """
    General exception for nested dictionary operations.

    This exception is raised when a nested dictionary operation fails
    but doesn't fall into a more specific error category.
    """

    def __init__(
        self, message: str = None, error_code: int = 0, path: List[Any] = None
    ) -> None:
        """
        Initialize a NestedDictionaryException.

        :param message: A message describing the error.
        :type message: str
        :param error_code: An integer code identifying the error type.
        :type error_code: int
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        super().__init__(message, error_code, path)


class StackedKeyError(KeyError, StackedDictionaryError):
    """
    Exception raised when a key operation fails in a stacked dictionary.

    This exception is raised for key-related errors such as missing keys,
    invalid key types, or operations that cannot be performed on certain keys.
    """

    def __init__(
        self, message: str = None, key: Any = None, path: List[Any] = None
    ) -> None:
        """
        Initialize a StackedKeyError.

        :param message: A message describing the error.
        :type message: str
        :param key: The key that caused the error.
        :type key: Any
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        self.key = key

        # Add key information to the message if available
        if key is not None and message:
            message = f"{message} (key: {key})"

        StackedDictionaryError.__init__(self, message, 0, path)
        KeyError.__init__(self, message)


class StackedAttributeError(AttributeError, StackedDictionaryError):
    """
    Exception raised when an attribute operation fails in a stacked dictionary.

    This exception is raised when attempting to access or modify attributes
    that don't exist or cannot be modified in the current context.
    """

    def __init__(
        self, message: str = None, attribute: str = None, path: List[Any] = None
    ) -> None:
        """
        Initialize a StackedAttributeError.

        :param message: A message describing the error.
        :type message: str
        :param attribute: The attribute that caused the error.
        :type attribute: str
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        self.attribute = attribute

        # Add attribute information to the message if available
        if attribute and message:
            message = f"{message} (attribute: {attribute})"

        StackedDictionaryError.__init__(self, message, 0, path)
        AttributeError.__init__(self, message)


class StackedTypeError(TypeError, StackedDictionaryError):
    """
    Exception raised when a type error occurs in a stacked dictionary operation.

    This exception is raised when an operation receives an argument of the wrong type,
    such as using nested lists as keys or attempting to perform operations on incompatible types.
    """

    def __init__(
        self,
        message: str = None,
        expected_type: Optional[type] = None,
        actual_type: Optional[type] = None,
        path: List[Any] = None,
    ) -> None:
        """
        Initialize a StackedTypeError.

        :param message: A message describing the error.
        :type message: str
        :param expected_type: The expected type for the operation.
        :type expected_type: Optional[type]
        :param actual_type: The actual type that was provided.
        :type actual_type: Optional[type]
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        self.expected_type = expected_type
        self.actual_type = actual_type

        # Add type information to the message if available
        if expected_type and actual_type and message:
            message = f"{message} (expected: {expected_type.__name__}, got: {actual_type.__name__})"

        StackedDictionaryError.__init__(self, message, 0, path)
        TypeError.__init__(self, message)


class StackedValueError(ValueError, StackedDictionaryError):
    """
    Exception raised when a value error occurs in a stacked dictionary operation.

    This exception is raised when an operation receives a value that is semantically
    inappropriate, such as a value that cannot be found in the dictionary.
    """

    def __init__(
        self, message: str = None, value: Any = None, path: List[Any] = None
    ) -> None:
        """
        Initialize a StackedValueError.

        :param message: A message describing the error.
        :type message: str
        :param value: The value that caused the error.
        :type value: Any
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        self.value = value

        # Add value information to the message if available
        if value is not None and message:
            message = f"{message} (value: {value})"

        StackedDictionaryError.__init__(self, message, 0, path)
        ValueError.__init__(self, message)


class StackedIndexError(IndexError, StackedDictionaryError):
    """
    Exception raised when an index error occurs in a stacked dictionary operation.

    This exception is raised when attempting to access an empty dictionary
    or when an operation cannot be performed due to the dictionary being empty.
    """

    def __init__(self, message: str = None, path: List[Any] = None) -> None:
        """
        Initialize a StackedIndexError.

        :param message: A message describing the error.
        :type message: str
        :param path: The path in the nested dictionary where the error occurred.
        :type path: List[Any]
        """
        StackedDictionaryError.__init__(self, message, 0, path)
        IndexError.__init__(self, message)
