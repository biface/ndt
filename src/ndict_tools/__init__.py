from .core import (
    CompactPathsView,
    DictPaths,
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

__version__ = "0.9.0"
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
