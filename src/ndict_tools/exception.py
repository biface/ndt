"""
This module provides specific exception classes for nested dictionaries.
These exceptions extend the standard **Exception**, **KeyError** and **AttributeError** classes for future developments.
"""

from __future__ import annotations


class StackedDictionaryError(Exception):
    """
    Exception raised when a stacked dictionary is invalid.
    """

    def __init__(self, message: str = None, error: int = 0) -> None:
        """
        StackedDictionaryError exception class.

        :param message: a message describing the error.
        :type message: str
        :param error: an integer describing the error.
        :type error: int
        :raises: None
        """
        super().__init__(message)
        self.error = error


class NestedDictionaryException(StackedDictionaryError):
    """
    Raised when a nested dictionary is invalid.
    """

    pass


class StackedKeyError(KeyError):
    """
    Exception raised when a key is not compatible with a stacked dictionary.
    """

    pass


class StackedAttributeError(AttributeError):
    """
    This exception is raised when a key is not compatible with stacked dictionary attributes.
    """

    pass
