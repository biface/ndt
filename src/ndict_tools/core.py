"""
This module provides tools and class for creating nested dictionaries, since standard python does not have nested
dictionaries.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from ._compat import override
from .tools import _CPaths, _Paths, _StackedDict

"""Classes section"""


class NestedDictionary(_StackedDict):
    """
    Nested dictionary class.

    This class is designed as a stacked dictionary. It represents a nest of dictionaries, that is to say that each
    key is a value or a nested dictionary. And so on...

    Parameters
    ----------
    *args : Mapping or iterable of (key, value) pairs
        Initial data. Nested plain dictionaries are converted recursively.
    default_setup : Mapping[str, Any], optional
        Configuration (keyword-only) with 'indent' and 'default_factory'
        keys. When omitted or empty, ``{'indent': 0, 'default_factory':
        NestedDictionary}`` is used. To get strict behaviour, use
        ``StrictNestedDictionary`` or pass ``'default_factory': None``.
    **kwargs : Any
        Initial data given as keyword arguments. They are data, not settings:
        ``NestedDictionary(indent=2)`` creates a key ``'indent'``.

    Examples
    --------
    >>> from ndict_tools import NestedDictionary
    >>> # From a dictionary
    >>> nd1 = NestedDictionary({'first': 1, 'second': {'1': "2:1", '2': "2:2"}, 'third': 3})
    >>> nd1['second']['2']
    '2:2'

    >>> # From an iterable of (key, value) pairs
    >>> nd2 = NestedDictionary(zip(['first', 'second', 'third'],
    ...                            [1, {'1': "2:1", '2': "2:2"}, 3]))
    >>> nd3 = NestedDictionary([('first', 1), ('second', {'1': "2:1", '2': "2:2"}),
    ...                         ('third', 3)])
    >>> nd1 == nd2 == nd3
    True

    >>> # Nested plain dictionaries become NestedDictionary levels
    >>> type(nd1['second']).__name__
    'NestedDictionary'
    """

    @classmethod
    @override
    def _normalize_setup(
        cls, setup: Mapping[str, Any] | Iterable[tuple[str, Any]] | None
    ) -> dict[str, Any]:
        """
        Supply the default configuration when none is given.

        A missing or empty ``setup`` becomes
        ``{'indent': 0, 'default_factory': NestedDictionary}``. Any other
        configuration is passed on unchanged for validation.
        """
        if not setup:
            setup = {"indent": 0, "default_factory": NestedDictionary}
        return super()._normalize_setup(setup)

    @override
    def paths(self) -> "PathsView":
        """
        Get a view of all hierarchical paths in this dictionary.

        Returns a lazy view over all paths without storing them in memory.
        The view supports iteration, length queries, membership tests, and
        various path operations.

        Returns
        -------
        PathsView
            A lazy view over all paths in the nested dictionary

        Examples
        --------
        >>> from ndict_tools import NestedDictionary
        >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> paths = nd.paths()
        >>> list(paths)
        [['a'], ['a', 'b'], ['a', 'c'], ['d']]

        >>> # Check if path exists
        >>> ['a', 'b'] in paths
        True

        >>> # Get number of paths
        >>> len(paths)
        4

        >>> # Get children of a path
        >>> paths.get_children(['a'])
        ['b', 'c']

        See Also
        --------
        compact_paths : Get compact representation of paths
        PathsView : Documentation of the paths view class
        """
        return PathsView(self)

    @override
    def compact_paths(self) -> "CompactPathsView":
        """
        Get a compact representation of all paths in this dictionary.

        Returns a compact view where the hierarchical structure is represented
        as nested lists, providing a factorized representation of paths.

        Returns
        -------
        CompactPathsView
            A compact view with factorized path structure

        Examples
        --------
        >>> from ndict_tools import NestedDictionary
        >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> cpaths = nd.compact_paths()
        >>> cpaths.structure
        [['a', 'b', 'c'], 'd']

        >>> # Expand to full paths
        >>> cpaths.expand()
        [['a'], ['a', 'b'], ['a', 'c'], ['d']]

        >>> # Check coverage
        >>> cpaths.is_covering(nd)
        True

        See Also
        --------
        paths : Get standard paths view
        CompactPathsView : Documentation of the compact paths view class
        """
        return CompactPathsView(self)


class StrictNestedDictionary(NestedDictionary):
    """
    Strict nested dictionary class.

    This class is designed to implement a non-default answer to an unknown key.

    Parameters
    ----------
    *args : Mapping or iterable of (key, value) pairs
        Initial data, as for ``NestedDictionary``
    default_setup : Mapping[str, Any], optional
        Configuration (keyword-only). Its ``default_factory`` is overridden;
        ``indent`` defaults to 0. The mapping is not modified.
    **kwargs : Any
        Initial data given as keyword arguments

    Notes
    -----
    This class overwrites the default_factory attribute to None, preventing
    automatic creation of nested dictionaries for unknown keys.
    """

    @classmethod
    @override
    def _normalize_setup(
        cls, setup: Mapping[str, Any] | Iterable[tuple[str, Any]] | None
    ) -> dict[str, Any]:
        """
        Force ``default_factory`` to None; ``indent`` defaults to 0.

        Works on a copy: the caller's configuration is never modified.
        """
        normalized = dict(setup) if setup else {}
        normalized.setdefault("indent", 0)
        normalized["default_factory"] = None
        return super()._normalize_setup(normalized)


class SmoothNestedDictionary(NestedDictionary):
    """
    Smooth nested dictionary class.

    This class is designed to implement a default answer as an empty
    SmoothNestedDictionary to an unknown key.

    Parameters
    ----------
    *args : Mapping or iterable of (key, value) pairs
        Initial data, as for ``NestedDictionary``
    default_setup : Mapping[str, Any], optional
        Configuration (keyword-only). Its ``default_factory`` is overridden;
        ``indent`` defaults to 0. The mapping is not modified.
    **kwargs : Any
        Initial data given as keyword arguments

    Notes
    -----
    This class overwrites the default_factory attribute to SmoothNestedDictionary,
    automatically creating nested dictionaries for unknown keys.
    """

    @classmethod
    @override
    def _normalize_setup(
        cls, setup: Mapping[str, Any] | Iterable[tuple[str, Any]] | None
    ) -> dict[str, Any]:
        """
        Force ``default_factory`` to SmoothNestedDictionary; ``indent`` defaults to 0.

        Works on a copy: the caller's configuration is never modified.
        """
        normalized = dict(setup) if setup else {}
        normalized.setdefault("indent", 0)
        normalized["default_factory"] = SmoothNestedDictionary
        return super()._normalize_setup(normalized)


class PathsView(_Paths):
    """
    A view providing access to all hierarchical paths in a nested dictionary.

    Similar to the standard ``dict.keys()`` view, but designed specifically for
    hierarchical paths in nested dictionaries. Provides lazy iteration over all
    paths without storing them in memory.

    This is the public API for working with paths: its conversions return
    public class instances.

    Parameters
    ----------
    stacked_dict : NestedDictionary
        The nested dictionary to create a view for (any class of the
        NestedDictionary family)

    Examples
    --------
    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
    >>> paths = nd.paths()
    >>> type(paths).__name__
    'PathsView'

    >>> # Iterate over paths
    >>> for path in paths:
    ...     print(path)
    ['a']
    ['a', 'b']
    ['a', 'c']
    ['d']

    >>> # Check if path exists
    >>> ['a', 'b'] in paths
    True

    >>> # Get number of paths
    >>> len(paths)
    4

    >>> # Get children of a path
    >>> paths.get_children(['a'])
    ['b', 'c']

    >>> # Get leaf paths only
    >>> paths.get_leaf_paths()
    [['a', 'b'], ['a', 'c'], ['d']]

    >>> # Convert to compact representation
    >>> compact = paths.to_compact()
    >>> type(compact).__name__
    'CompactPathsView'

    See Also
    --------
    CompactPathsView : Compact representation of paths
    NestedDictionary : Nested dictionary with path operations
    """

    @override
    def to_compact(self) -> "CompactPathsView":
        """
        Convert this PathsView to a CompactPathsView.

        Returns
        -------
        CompactPathsView
            Compact representation with the same paths

        Examples
        --------
        >>> from ndict_tools import NestedDictionary
        >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> paths = nd.paths()
        >>> compact = paths.to_compact()
        >>> compact.structure
        [['a', 'b', 'c'], 'd']
        """
        return CompactPathsView(self._stacked_dict)


class CompactPathsView(_CPaths):
    """
    A view providing compact representation of hierarchical paths.

    Provides a factorized/compact representation where the hierarchical structure
    is represented as nested lists. This is useful for efficiently representing
    and manipulating path structures, especially when dealing with large numbers
    of similar paths.

    The compact structure uses nested lists where:

    - Leaf nodes are represented by their key alone
    - Internal nodes are represented as [key, child1, child2, ...]

    A structure built from a dictionary is the canonical form of its paths, and
    conversion works in both directions.

    Parameters
    ----------
    stacked_dict : NestedDictionary
        The nested dictionary to create a compact view for (any class of the
        NestedDictionary family)

    Examples
    --------
    >>> from ndict_tools import NestedDictionary, PathsView
    >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
    >>> cpaths = nd.compact_paths()
    >>> type(cpaths).__name__
    'CompactPathsView'

    >>> # Get compact structure
    >>> cpaths.structure
    [['a', 'b', 'c'], 'd']

    >>> # Expand to full paths
    >>> cpaths.expand()
    [['a'], ['a', 'b'], ['a', 'c'], ['d']]

    >>> # Iterate over the expanded paths, as with PathsView
    >>> list(cpaths)
    [['a'], ['a', 'b'], ['a', 'c'], ['d']]

    >>> # Set custom structure
    >>> cpaths.structure = [['x', 'y', 'z']]
    >>> cpaths.expand()
    [['x'], ['x', 'y'], ['x', 'z']]

    >>> # Check coverage against original dictionary
    >>> cpaths = nd.compact_paths()
    >>> cpaths.is_covering(nd)
    True
    >>> cpaths.coverage(nd)
    1.0

    >>> # Partial structure
    >>> cpaths.structure = [['a', 'b']]
    >>> cpaths.coverage(nd)
    0.5
    >>> cpaths.uncovered_paths(nd)
    [['a', 'c'], ['d']]

    Notes
    -----
    **Compact structure format:**

    - Simple leaf: ``'key'``
    - Node with children: ``['key', child1, child2, ...]``

    **Examples of compact structures:**

    - ``[['a'], ['b']]`` → two independent paths: ``['a']`` and ``['b']``
    - ``[['a', 'b', 'c']]`` → paths: ``['a']``, ``['a', 'b']``, ``['a', 'c']``
    - ``[['a', ['b', 'c']]]`` → equivalent to the above (explicit nesting)

    See Also
    --------
    PathsView : Standard view for iterating over paths
    NestedDictionary : Nested dictionary with path operations
    expand_structure : Static method to expand a compact structure
    """

    def to_paths(self) -> "PathsView":
        """
        Convert this CompactPathsView to a PathsView.

        Returns
        -------
        PathsView
            Standard paths view with the same underlying data

        Examples
        --------
        >>> from ndict_tools import NestedDictionary
        >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> cpaths = nd.compact_paths()
        >>> paths = cpaths.to_paths()
        >>> type(paths).__name__
        'PathsView'
        """
        return PathsView(self._stacked_dict)
