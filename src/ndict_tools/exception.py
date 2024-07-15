"""
This module provides exception classes.
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


class NestedDirectoryException(StackedDictionaryError):
    """
    Raised when a nested directory is invalid.
    """
    pass


class StackedKeyError(KeyError):
    """
    Exception raised when a key is not a nested directory.
    """
    pass
