"""
This module provides specific exception classes for nested dictionaries.
These exceptions extend the standard **Exception**, **KeyError**, **AttributeError**,
**TypeError**, **ValueError** and **IndexError** classes to provide more context and
better error handling for nested dictionary operations.
"""

from typing import Any


class StackedDictionaryError(Exception):
    """
    Base exception class for all stacked dictionary errors.

    This is the parent class for all exceptions related to stacked dictionaries.
    It provides context about the error including an optional error code.

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    error_code : int, optional
        Integer code identifying the error type, 0 by default.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. ``" (at path: k1 | k2)"`` is appended to the message, or
        the message becomes ``"Error at path: k1 | k2"`` when there is none.
    """

    def __init__(
        self,
        message: str | None = None,
        error_code: int = 0,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        self.error_code: int = error_code
        self.path: list[Any] = path or []

        # Add path information to the message if available
        if path:
            path_str = " | ".join(str(k) for k in path)
            message = (
                f"{message} (at path: {path_str})"
                if message
                else f"Error at path: {path_str}"
            )

        super().__init__(message)


class NestedDictionaryException(StackedDictionaryError):
    """
    General exception for nested dictionary operations.

    This exception is raised when a nested dictionary operation fails
    but doesn't fall into a more specific error category.

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    error_code : int, optional
        Integer code identifying the error type, 0 by default.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. ``" (at path: k1 | k2)"`` is appended to the message, or
        the message becomes ``"Error at path: k1 | k2"`` when there is none.
    """

    def __init__(
        self,
        message: str | None = None,
        error_code: int = 0,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        super().__init__(message, error_code, path)


class StackedKeyError(KeyError, StackedDictionaryError):
    """
    Exception raised when a key operation fails in a stacked dictionary.

    This exception is raised for key-related errors such as missing keys,
    invalid key types, or operations that cannot be performed on certain keys.

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    key : Any, optional
        Key that caused the error. When given with a message,
        ``" (key: <key>)"`` is appended to the message.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. It is not added to the message.
    """

    def __init__(
        self,
        message: str | None = None,
        key: Any | None = None,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        self.key: Any = key

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

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    attribute : str, optional
        Attribute that caused the error. When given with a message,
        ``" (attribute: <attribute>)"`` is appended to the message.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. It is not added to the message.
    """

    def __init__(
        self,
        message: str | None = None,
        attribute: str | None = None,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        self.attribute: str | None = attribute

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

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    expected_type : type, optional
        Type expected by the operation.
    actual_type : type, optional
        Type actually provided. When both types are given with a message,
        ``" (expected: <name>, got: <name>)"`` is appended to the message.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. It is not added to the message.
    """

    def __init__(
        self,
        message: str | None = None,
        expected_type: type | None = None,
        actual_type: type | None = None,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        self.expected_type: type | None = expected_type
        self.actual_type: type | None = actual_type

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

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    value : Any, optional
        Value that caused the error. When given with a message,
        ``" (value: <value>)"`` is appended to the message.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. It is not added to the message.
    """

    def __init__(
        self,
        message: str | None = None,
        value: Any | None = None,
        path: list[Any] | None = None,
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        self.value: Any = value

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

    Parameters
    ----------
    message : str, optional
        Message describing the error.
    path : list[Any], optional
        Path in the nested dictionary where the error occurred, stored in
        ``path``. It is not added to the message.
    """

    def __init__(
        self, message: str | None = None, path: list[Any] | None = None
    ) -> None:
        """
        Initialize the exception; the parameters are described on the class.
        """
        StackedDictionaryError.__init__(self, message, 0, path)
        IndexError.__init__(self, message)
