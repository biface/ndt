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
from .tools import DictSearch, _Paths

__version__ = "0.9.0"
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
    "_Paths",
    "DictSearch",
]
