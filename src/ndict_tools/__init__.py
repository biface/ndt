from .core import NestedDictionary, SmoothNestedDictionary, StrictNestedDictionary
from .exception import (
    NestedDictionaryException,
    StackedAttributeError,
    StackedDictionaryError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)
from .tools import DictPaths, DictSearch

__version__ = "0.8.0"
__author__ = "biface"

__all__ = [
    "NestedDictionary",
    "SmoothNestedDictionary",
    "StrictNestedDictionary",
    "NestedDictionaryException",
    "StackedDictionaryError",
    "StackedKeyError",
    "StackedAttributeError",
    "StackedTypeError",
    "StackedValueError",
    "StackedIndexError",
    "DictPaths",
    "DictSearch",
]
