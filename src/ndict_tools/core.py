"""
This module provides tools and class for creating nested dictionaries, since standard python does not have nested
dictionaries.
"""

from __future__ import annotations

import warnings

from .tools import _CPaths, _Paths, _StackedDict

"""Classes section"""


class NestedDictionary(_StackedDict):
    """
    Nested dictionary class.

    This class is designed as a stacked dictionary. It represents a nest of dictionaries, that is to say that each
    key is a value or a nested dictionary. And so on...

    Parameters
    ----------
    *args : Iterable
        The first one of the list must be a dictionary to instantiate an object
    **kwargs : dict
        Enrichments settings:

        - indent : int, optional
            Indentation of the printable nested dictionary (used by json.dumps() function)
        - strict : bool, optional (default=False)
            Strict mode define default answer to unknown key
        - default_setup : dict, optional
            Custom setup for default behavior

    Examples
    --------
    >>> NestedDictionary({'first': 1,'second': {'1': "2:1", '2': "2:2", '3': "3:2"}, 'third': 3, 'fourth': 4})

    >>> NestedDictionary(zip(['first','second', 'third', 'fourth'],
    ...                  [1, {'1': "2:1", '2': "2:2", '3': "3:2"}, 3, 4]))

    >>> NestedDictionary([('first', 1), ('second', {'1': "2:1", '2': "2:2", '3': "3:2"}),
    ...                   ('third', 3), ('fourth', 4)])
    """

    def __init__(self, *args, **kwargs):

        indent = kwargs.pop("indent", 0)
        strict = kwargs.pop("strict", False)
        default_setup = kwargs.pop("default_setup", None)
        default_class = None if strict else NestedDictionary

        if not default_setup:
            default_setup = {"indent": indent, "default_factory": default_class}

        super().__init__(
            *args,
            **kwargs,
            default_setup=default_setup,
        )

    def paths(self) -> PathsView:
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

    def compact_paths(self) -> CompactPathsView:
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
        >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> cpaths = nd.compact_paths()
        >>> cpaths.structure
        [['a', 'b', 'c'], ['d']]

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
    *args : Iterable
        Positional arguments passed to NestedDictionary
    **kwargs : dict
        Keyword arguments passed to NestedDictionary

    Notes
    -----
    This class overwrites the default_factory attribute to None, preventing
    automatic creation of nested dictionaries for unknown keys.
    """

    def __init__(self, *args, **kwargs):

        setup = kwargs.pop("default_setup", None)
        if setup:
            setup["indent"] = setup.pop("indent", 0)
            setup["default_factory"] = None
        else:
            setup = {"indent": 0, "default_factory": None}

        super().__init__(*args, **kwargs, default_setup=setup)


class SmoothNestedDictionary(NestedDictionary):
    """
    Smooth nested dictionary class.

    This class is designed to implement a default answer as an empty
    SmoothNestedDictionary to an unknown key.

    Parameters
    ----------
    *args : Iterable
        Positional arguments passed to NestedDictionary
    **kwargs : dict
        Keyword arguments passed to NestedDictionary

    Notes
    -----
    This class overwrites the default_factory attribute to SmoothNestedDictionary,
    automatically creating nested dictionaries for unknown keys.
    """

    def __init__(self, *args, **kwargs):

        setup = kwargs.pop("default_setup", None)
        if setup:
            setup["indent"] = setup.pop("indent", 0)
            setup["default_factory"] = SmoothNestedDictionary

        else:
            setup = {"indent": 0, "default_factory": SmoothNestedDictionary}

        super().__init__(*args, **kwargs, default_setup=setup)


class PathsView(_Paths):
    """
    A view providing access to all hierarchical paths in a nested dictionary.

    Similar to the standard ``dict.keys()`` view, but designed specifically for
    hierarchical paths in nested dictionaries. Provides lazy iteration over all
    paths without storing them in memory.

    This is the public API for working with paths. It inherits all functionality
    from the internal ``_Paths`` class and ensures that conversions return public
    class instances.

    Parameters
    ----------
    stacked_dict : NestedDictionary or _StackedDict
        The nested dictionary to create a view for

    Examples
    --------
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

    def to_compact(self) -> "CompactPathsView":
        """
        Convert this PathsView to a CompactPathsView.

        Returns
        -------
        CompactPathsView
            Compact representation with the same paths

        Examples
        --------
        >>> paths = nd.paths()
        >>> compact = paths.to_compact()
        >>> compact.structure
        [['a', 'b', 'c'], ['d']]
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

    This class provides a bijective mapping between compact and expanded forms,
    allowing efficient conversion in both directions.

    Parameters
    ----------
    stacked_dict : NestedDictionary or _StackedDict
        The nested dictionary to create a compact view for

    Attributes
    ----------
    structure : List[Any]
        The compact representation as nested lists (lazy-built, read/write)

    Examples
    --------
    >>> nd = NestedDictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
    >>> cpaths = nd.compact_paths()
    >>> type(cpaths).__name__
    'CompactPathsView'

    >>> # Get compact structure
    >>> cpaths.structure
    [['a', 'b', 'c'], ['d']]

    >>> # Expand to full paths
    >>> cpaths.expand()
    [['a'], ['a', 'b'], ['a', 'c'], ['d']]

    >>> # Can still iterate (inherited from PathsView)
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

    def to_paths(self) -> PathsView:
        """
        Convert this CompactPathsView to a PathsView.

        Returns
        -------
        PathsView
            Standard paths view with the same underlying data

        Examples
        --------
        >>> cpaths = nd.compact_paths()
        >>> paths = cpaths.to_paths()
        >>> type(paths).__name__
        'PathsView'
        """
        return PathsView(self._stacked_dict)


class DictPaths(_CPaths):
    """
    Class of compact nested paths.

    .. deprecated:: 0.9.0
        `DictPaths` is deprecated and will be removed in version 1.0.0.
        Use :class:`CompactPathsView` instead.

    Parameters
    ----------
    value : NestedDictionary or _StackedDict
        The nested dictionary to create a compact view for

    Notes
    -----
    **Migration guide:**

    - Replace ``DictPaths(nd)`` with ``nd.compact_paths()``
    - Or use ``CompactPathsView(nd)`` directly

    Examples
    --------
    >>> # Old way (deprecated)
    >>> dpaths = DictPaths(nd)  # doctest: +SKIP

    >>> # New way (recommended)
    >>> cpaths = nd.compact_paths()

    >>> # Or directly
    >>> cpaths = CompactPathsView(nd)

    See Also
    --------
    CompactPathsView : The replacement class for DictPaths
    NestedDictionary.compact_paths : Method to create a CompactPathsView
    """

    def __init__(self, value):
        warnings.warn(
            "DictPaths is deprecated since version 0.9.0 and will be removed in version 1.0.0. "
            "Use CompactPathsView or NestedDictionary.compact_paths() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(value)
