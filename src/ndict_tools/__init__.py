from .core import (
    CompactPathsView,
    NestedDictionary,
    PathsView,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from .exception import (
    NestedDictionaryException,
    StackedAttributeError,
    StackedDictionaryError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)

__version__ = "1.2.0rc2"
__author__ = "biface"

__all__ = [
    "PathsView",
    "CompactPathsView",
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
]
