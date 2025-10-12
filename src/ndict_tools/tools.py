"""
This module provides the **core technical infrastructure** for manipulating nested dictionaries. While hidden from the
package's public API, it serves as the foundation for all nested dictionary operations.

* **_StackedDict**: Base class for nested dictionary structures
* **_HKey**: Tree node representing hierarchical keys (private, optimized with tuples)
* **DictPaths**: View of all paths in a nested dictionary
* **DictSearch**: Factorized search structure for nested dictionaries

The module enables efficient navigation, querying, and factorization of deeply
nested dictionary structures with support for various key types.

The ``_StackedDict`` class is the **central engine** of the ``ndict_tools`` package. It implements all the fundamental
attributes, methods, and logic required to initialize, manage, and manipulate nested dictionaries. This class is
designed to:

- Orchestrate the **basic building blocks** of nested dictionary functionality
- Provide the **complete toolset** for dictionary nesting, key management, and hierarchical data operations
- Serve as a **versatile base** for current and future dictionary implementations

"""

from __future__ import annotations

from collections import defaultdict, deque
from textwrap import indent
from typing import Any, Generator, List, Dict, Optional, Iterator, Tuple, Set, Callable, Union

from .exception import (
    StackedAttributeError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)

"""Internal functions"""


def compare_dict(d1, d2) -> bool:
    """
    Compare two (possibly nested) structures: dicts, lists, tuples, sets, or scalars.
    Returns True if they are the same type and have equal content recursively.
    """
    if type(d1) != type(d2):
        return False
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            return False
        for k in d1:
            if not compare_dict(d1[k], d2[k]):
                return False
        return True
    elif isinstance(d1, (list, tuple, set)):
        if len(d1) != len(d2):
            return False
        return all(compare_dict(x, y) for x, y in zip(d1, d2))
    else:
        return d1 == d2


def unpack_items(dictionary: dict) -> Generator:
    """
    This function de-stacks items from a nested dictionary.

    :param dictionary: Dictionary to unpack.
    :type dictionary: dict
    :return: Generator that yields items from a nested dictionary.
    :rtype: Generator
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            if not value:  # Handle empty dictionaries
                yield (key,), value
            else:  # Recursive case for non-empty dictionaries
                for stacked_key, stacked_value in unpack_items(value):
                    yield (key,) + stacked_key, stacked_value
        else:  # Base case for non-dictionary values
            yield (key,), value


def from_dict(dictionary: dict, class_name: object, **class_options) -> _StackedDict:
    """
    This recursive function is used to transform a dictionary into a stacked dictionary.

    This function enhances and replaces the previous from_dict() function in the core module of this package.
    It allows you to create an object subclass of a _StackedDict with initialization options if requested and
    attributes to be set.

    :param dictionary: The dictionary to transform
    :type dictionary: dict
    :param class_name: name of the class to return
    :type class_name: object
    :param class_options: default settings to pass to class instance to be set up.
    :type class_options: dict
    :return: stacked dictionary or of subclasses of _StackedDict
    :rtype: _StackedDict
    :raise StackedKeyError: if attribute called is not an attribute of the class hierarchy.
    """

    if "default_setup" in class_options:
        dict_object = class_name(**class_options)
    else:
        raise StackedKeyError(
            f"The key 'default_setup' must be present in class options : {class_options}",
            key="default_setup",
        )

    for key, value in dictionary.items():
        if isinstance(value, _StackedDict):
            dict_object[key] = value
        elif isinstance(value, dict):
            dict_object[key] = from_dict(value, class_name, **class_options)
        else:
            dict_object[key] = value

    return dict_object


"""Private Classes section"""


class _HKey:
    """
    Private tree node representing a hierarchical key in a nested dictionary.

    Each ``_HKey`` instance represents a single key in a nested dictionary structure,
    forming a tree where:

    * Each node holds a key value from the dictionary
    * Children nodes (stored as immutable tuples) represent keys in nested dictionaries
    * Parent references enable path reconstruction from any node

    The use of immutable tuples for children optimizes memory usage and iteration
    performance for tree traversal algorithms (DFS, BFS).

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code. Access it through ``DictSearch`` or ``NestedDictionary``.

    Parameters
    ----------
    key : Any
        The key value this node represents
    parent : Optional[_HKey], optional
        Reference to parent node, None for root nodes
    is_root : bool, optional
        Whether this node is a root node, by default False

    Attributes
    ----------
    key : Any
        The key value this node represents
    children : Tuple[_HKey, ...]
        Immutable tuple of child nodes
    parent : Optional[_HKey]
        Reference to parent node (None for root)
    is_root : bool
        True if this is a root node

    Examples
    --------
    >>> root = _HKey('a')
    >>> child_b = root.add_child('b')
    >>> child_c = root.add_child('c')
    >>> root.get_child_keys()
    ['b', 'c']
    >>> child_b.get_path()
    ['a', 'b']

    See Also
    --------
    DictSearch : Uses _HKey internally for tree representation
    """

    __slots__ = ('key', 'children', 'parent', 'is_root')

    def __init__(self, key: Any, parent: Optional['_HKey'] = None, is_root: bool = False) -> None:
        self.key: Any = key
        self.children: Tuple[_HKey, ...] = ()
        self.parent: Optional[_HKey] = parent
        self.is_root: bool = is_root

    @classmethod
    def build_forest(cls, stacked_dict: Dict) -> '_HKey':
        """
        Build a forest of _HKey trees from a nested dictionary.

        Creates a root node containing all top-level keys as children,
        recursively building the tree structure for nested dictionaries.

        Parameters
        ----------
        stacked_dict : dict
            A dictionary (or _StackedDict) to build the tree from

        Returns
        -------
        _HKey
            Root node (with is_root=True, key=None) containing the forest

        Examples
        --------
        >>> data = {'a': 1, 'b': {'c': 2}}
        >>> forest = _HKey.build_forest(data)
        >>> forest.get_child_keys()
        ['a', 'b']
        >>> forest.is_root
        True
        """
        root: _HKey = cls(None, is_root=True)
        root._build_from_dict(stacked_dict)
        return root

    def _build_from_dict(self, current_dict: Dict) -> None:
        """
        Recursively build tree structure from a dictionary.

        Parameters
        ----------
        current_dict : dict
            Dictionary to process at current level
        """
        children_list: List[_HKey] = []

        for key, value in current_dict.items():
            child: _HKey = _HKey(key, parent=self)
            children_list.append(child)

            if isinstance(value, dict):
                child._build_from_dict(value)

        self.children = tuple(children_list)

    def add_child(self, key: Any) -> '_HKey':
        """
        Add a child node with the given key.

        If a child with this key already exists, returns the existing child.
        Creates a new tuple with the additional child (O(n) operation).

        .. note::
           For adding multiple children, use :meth:`add_children` for better performance.

        Parameters
        ----------
        key : Any
            Key for the new child node

        Returns
        -------
        _HKey
            The child node (newly created or existing)

        Examples
        --------
        >>> node = _HKey('parent')
        >>> child = node.add_child('child')
        >>> child.key
        'child'
        >>> child.parent.key
        'parent'
        """
        for child in self.children:
            if child.key == key:
                return child

        new_child: _HKey = _HKey(key, parent=self)
        self.children = self.children + (new_child,)
        return new_child

    def add_children(self, keys: List[Any]) -> Tuple['_HKey', ...]:
        """
        Add multiple children at once (more efficient than repeated add_child).

        Parameters
        ----------
        keys : List[Any]
            List of keys to add as children

        Returns
        -------
        Tuple[_HKey, ...]
            Tuple of newly created child nodes

        Examples
        --------
        >>> node = _HKey('parent')
        >>> children = node.add_children(['a', 'b', 'c'])
        >>> len(node.children)
        3
        """
        new_children: List[_HKey] = []

        for key in keys:
            exists = any(child.key == key for child in self.children)
            if not exists:
                new_child = _HKey(key, parent=self)
                new_children.append(new_child)

        self.children = self.children + tuple(new_children)
        return tuple(new_children)

    def get_child(self, key: Any) -> Optional['_HKey']:
        """
        Get a child node by key.

        Parameters
        ----------
        key : Any
            Key to look up

        Returns
        -------
        Optional[_HKey]
            Child node if found, None otherwise

        Examples
        --------
        >>> node = _HKey('parent')
        >>> node.add_child('child')
        >>> child = node.get_child('child')
        >>> child.key
        'child'
        >>> node.get_child('nonexistent') is None
        True
        """
        for child in self.children:
            if child.key == key:
                return child
        return None

    def get_child_keys(self) -> List[Any]:
        """
        Get list of all child keys.

        Returns
        -------
        List[Any]
            List of keys for all children

        Examples
        --------
        >>> node = _HKey('parent')
        >>> node.add_child('a')
        >>> node.add_child('b')
        >>> sorted(node.get_child_keys())
        ['a', 'b']
        """
        return [child.key for child in self.children]

    def has_children(self) -> bool:
        """
        Check if this node has any children.

        Returns
        -------
        bool
            True if node has at least one child
        """
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        """
        Check if this is a leaf node (no children).

        Returns
        -------
        bool
            True if node has no children
        """
        return not self.has_children()

    def get_path(self) -> List[Any]:
        """
        Get the path from root to this node.

        Traverses parent references to reconstruct the full path.

        Returns
        -------
        List[Any]
            List of keys from root to this node

        Examples
        --------
        >>> root = _HKey('a')
        >>> child = root.add_child('b')
        >>> grandchild = child.add_child('c')
        >>> grandchild.get_path()
        ['a', 'b', 'c']
        """
        path: List[Any] = []
        current: Optional[_HKey] = self

        while current is not None and not current.is_root:
            path.append(current.key)
            current = current.parent

        return list(reversed(path))

    def find_by_path(self, path: List[Any]) -> Optional['_HKey']:
        """
        Find a node by following a path from this node.

        Parameters
        ----------
        path : List[Any]
            List of keys to follow

        Returns
        -------
        Optional[_HKey]
            Node at end of path, or None if path doesn't exist

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> node = root.find_by_path(['a', 'b', 'c'])
        >>> node.key
        'c'
        >>> root.find_by_path(['a', 'x']) is None
        True
        """
        current: _HKey = self

        for key in path:
            child: Optional[_HKey] = current.get_child(key)
            if child is None:
                return None
            current = child

        return current

    def get_all_paths(self) -> List[List[Any]]:
        """
        Get all paths from this node to all descendants.

        Recursively collects paths to every node in the subtree,
        including intermediate nodes and leaves.

        Returns
        -------
        List[List[Any]]
            List of all paths in the subtree

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> paths = root.get_all_paths()
        >>> len(paths)
        3
        >>> sorted([tuple(p) for p in paths])
        [('a',), ('a', 'b'), ('a', 'c')]
        """
        paths: List[List[Any]] = []
        base_path: List[Any] = self.get_path() if not self.is_root else []

        def collect_paths(node: _HKey, current_path: List[Any]) -> None:
            if not node.is_root:
                node_path: List[Any] = current_path + [node.key]
                paths.append(node_path)

                for child in node.children:
                    collect_paths(child, node_path)
            else:
                for child in node.children:
                    collect_paths(child, current_path)

        collect_paths(self, base_path)
        return paths

    def get_descendants(self) -> List['_HKey']:
        """
        Get all descendant nodes (children, grandchildren, etc.).

        Returns
        -------
        List[_HKey]
            List of all descendant nodes in DFS order
        """
        descendants: List[_HKey] = []

        def collect(node: _HKey) -> None:
            for child in node.children:
                descendants.append(child)
                collect(child)

        collect(self)
        return descendants

    def get_depth(self) -> int:
        """
        Get the depth of this node (distance from root).

        Returns
        -------
        int
            Depth level (0 for direct children of root)
        """
        depth: int = 0
        current: Optional[_HKey] = self.parent

        while current is not None and not current.is_root:
            depth += 1
            current = current.parent

        return depth

    def get_max_depth(self) -> int:
        """
        Get the maximum depth of the subtree rooted at this node.

        Returns
        -------
        int
            Maximum depth (0 for leaf nodes)
        """
        if not self.children:
            return 0

        return 1 + max(child.get_max_depth() for child in self.children)

    def iter_children(self) -> Iterator['_HKey']:
        """
        Iterate over direct child nodes.

        Yields
        ------
        _HKey
            Each child node
        """
        return iter(self.children)

    def iter_leaves(self) -> Iterator['_HKey']:
        """
        Iterate over all leaf nodes in the subtree.

        Yields
        ------
        _HKey
            Each leaf node (nodes with no children)
        """
        if self.is_leaf():
            yield self
        else:
            for child in self.children:
                yield from child.iter_leaves()

    def to_dict(self) -> Dict[Any, Any]:
        """
        Convert the tree structure back to a nested dict.

        Returns
        -------
        Dict[Any, Any]
            Nested dictionary representation of the tree
        """
        result: Dict[Any, Any] = {}

        for child in self.children:
            if child.has_children():
                result[child.key] = child.to_dict()
            else:
                result[child.key] = {}

        return result

    # ========================================================================
    # Tree Traversal Algorithms
    # ========================================================================

    def dfs_preorder(self, visit: Optional[Callable[['_HKey'], None]] = None) -> Iterator['_HKey']:
        """
        Depth-First Search traversal in pre-order (node, then children).

        Pre-order: Visit current node before its children.
        Order: Root → Left subtree → Right subtree

        Parameters
        ----------
        visit : Optional[Callable[[_HKey], None]], optional
            Optional callback function called on each node

        Yields
        ------
        _HKey
            Each node in pre-order

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> keys = [node.key for node in root.dfs_preorder() if not node.is_root]
        >>> keys
        ['a', 'b', 'c']

        See Also
        --------
        dfs_postorder : Post-order DFS traversal
        bfs : Breadth-first traversal
        """
        if visit:
            visit(self)
        yield self

        for child in self.children:
            yield from child.dfs_preorder(visit)

    def dfs_postorder(self, visit: Optional[Callable[['_HKey'], None]] = None) -> Iterator['_HKey']:
        """
        Depth-First Search traversal in post-order (children, then node).

        Post-order: Visit children before current node.
        Order: Left subtree → Right subtree → Root
        Useful for deletion or bottom-up calculations.

        Parameters
        ----------
        visit : Optional[Callable[[_HKey], None]], optional
            Optional callback function called on each node

        Yields
        ------
        _HKey
            Each node in post-order

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> keys = [node.key for node in root.dfs_postorder() if not node.is_root]
        >>> keys
        ['b', 'c', 'a']

        See Also
        --------
        dfs_preorder : Pre-order DFS traversal
        """
        for child in self.children:
            yield from child.dfs_postorder(visit)

        if visit:
            visit(self)
        yield self

    def bfs(self, visit: Optional[Callable[['_HKey'], None]] = None) -> Iterator['_HKey']:
        """
        Breadth-First Search (level-order) traversal.

        BFS explores all nodes at depth N before moving to depth N+1.
        Uses a queue (deque) for optimal O(1) operations.

        Parameters
        ----------
        visit : Optional[Callable[[_HKey], None]], optional
            Optional callback function called on each node

        Yields
        ------
        _HKey
            Each node in level-order

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> keys = [node.key for node in root.bfs() if not node.is_root]
        >>> keys
        ['a', 'b', 'c']

        >>> # BFS with depth tracking
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> for node in root.bfs():
        ...     if not node.is_root:
        ...         print(f"Depth {node.get_depth()}: {node.key}")
        Depth 0: a
        Depth 0: d
        Depth 1: b
        Depth 1: c

        See Also
        --------
        dfs_preorder : Depth-first traversal
        iter_by_level : Get nodes grouped by level
        """
        queue: deque = deque([self])

        while queue:
            node = queue.popleft()
            if visit:
                visit(node)
            yield node

            queue.extend(node.children)

    def dfs_find(self, predicate: Callable[['_HKey'], bool]) -> Optional['_HKey']:
        """
        Find first node matching predicate using DFS.

        Performs depth-first search and returns the first node for which
        the predicate returns True. Returns None if no match found.

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that returns True for the target node

        Returns
        -------
        Optional[_HKey]
            First matching node, or None if not found

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}, 'c': 2})
        >>> node = root.dfs_find(lambda n: n.key == 'b')
        >>> node.key
        'b'
        >>> root.dfs_find(lambda n: n.key == 'z') is None
        True

        See Also
        --------
        bfs_find : BFS-based search
        find_all : Find all matching nodes
        """
        if predicate(self):
            return self

        for child in self.children:
            result = child.dfs_find(predicate)
            if result is not None:
                return result

        return None

    def bfs_find(self, predicate: Callable[['_HKey'], bool]) -> Optional['_HKey']:
        """
        Find first node matching predicate using BFS.

        Performs breadth-first search and returns the first node for which
        the predicate returns True. Finds nodes at shallower depths first.

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that returns True for the target node

        Returns
        -------
        Optional[_HKey]
            First matching node, or None if not found

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> # BFS finds 'b' before 'c' (closer to root)
        >>> node = root.bfs_find(lambda n: n.key in ['b', 'c'])
        >>> node.key
        'b'

        See Also
        --------
        dfs_find : DFS-based search
        """
        for node in self.bfs():
            if predicate(node):
                return node
        return None

    def find_all(self, predicate: Callable[['_HKey'], bool]) -> List['_HKey']:
        """
        Find all nodes matching predicate.

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that returns True for target nodes

        Returns
        -------
        List[_HKey]
            List of all matching nodes

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}, 'c': {'b': 2}})
        >>> nodes = root.find_all(lambda n: n.key == 'b')
        >>> len(nodes)
        2
        >>> [n.get_path() for n in nodes]
        [['a', 'b'], ['c', 'b']]
        """
        results: List[_HKey] = []

        for node in self.dfs_preorder():
            if predicate(node):
                results.append(node)

        return results

    def find_by_key(self, key: Any, find_all: bool = False) -> Optional['_HKey'] | List['_HKey']:
        """
        Find node(s) with specific key value.

        Convenience method for finding nodes by their key value.

        Parameters
        ----------
        key : Any
            Key value to search for
        find_all : bool, optional
            If True, return all matches; if False, return first match

        Returns
        -------
        Optional[_HKey] or List[_HKey]
            Single node if find_all=False, list of nodes if find_all=True

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}, 'c': {'b': 2}})
        >>> node = root.find_by_key('b')
        >>> node.get_path()
        ['a', 'b']
        >>> nodes = root.find_by_key('b', find_all=True)
        >>> len(nodes)
        2
        """
        if find_all:
            return self.find_all(lambda n: n.key == key)
        else:
            return self.dfs_find(lambda n: n.key == key)

    def iter_by_level(self) -> Iterator[Tuple[int, List['_HKey']]]:
        """
        Iterate over nodes grouped by depth level.

        Yields tuples of (depth, nodes_at_depth) for each level of the tree.

        Yields
        ------
        Tuple[int, List[_HKey]]
            (depth_level, list_of_nodes_at_that_level)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> for depth, nodes in root.iter_by_level():
        ...     keys = [n.key for n in nodes if not n.is_root]
        ...     if keys:
        ...         print(f"Level {depth}: {keys}")
        Level 0: ['a']
        Level 1: ['b', 'c']

        See Also
        --------
        bfs : Breadth-first traversal
        get_nodes_at_depth : Get nodes at specific depth
        """
        levels: Dict[int, List[_HKey]] = defaultdict(list)

        for node in self.bfs():
            depth = node.get_depth() if not node.is_root else -1
            if depth >= 0:
                levels[depth].append(node)

        for depth in sorted(levels.keys()):
            yield depth, levels[depth]

    def get_nodes_at_depth(self, target_depth: int) -> List['_HKey']:
        """
        Get all nodes at a specific depth.

        Parameters
        ----------
        target_depth : int
            Depth level to query (0 = root's children)

        Returns
        -------
        List[_HKey]
            All nodes at the specified depth

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> nodes = root.get_nodes_at_depth(1)
        >>> [n.key for n in nodes]
        ['b']

        See Also
        --------
        iter_by_level : Iterate all levels
        """
        return [node for node in self.bfs()
                if not node.is_root and node.get_depth() == target_depth]

    def filter_paths(self, predicate: Callable[[List[Any]], bool]) -> List[List[Any]]:
        """
        Filter paths based on a predicate function.

        Parameters
        ----------
        predicate : Callable[[List[Any]], bool]
            Function that takes a path and returns True to include it

        Returns
        -------
        List[List[Any]]
            Filtered list of paths

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> # Get paths longer than 1
        >>> paths = root.filter_paths(lambda p: len(p) > 1)
        >>> sorted([tuple(p) for p in paths])
        [('a', 'b'), ('a', 'c')]

        >>> # Get paths containing specific key
        >>> paths = root.filter_paths(lambda p: 'b' in p)
        >>> paths
        [['a', 'b']]
        """
        all_paths = self.get_all_paths()
        return [path for path in all_paths if predicate(path)]

    def map_nodes(self, func: Callable[['_HKey'], Any]) -> List[Any]:
        """
        Apply a function to all nodes and collect results.

        Parameters
        ----------
        func : Callable[[_HKey], Any]
            Function to apply to each node

        Returns
        -------
        List[Any]
            List of results from applying func to each node

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}})
        >>> # Get all keys with their depths
        >>> results = root.map_nodes(lambda n: (n.key, n.get_depth()) if not n.is_root else None)
        >>> [r for r in results if r is not None]
        [('a', 0), ('b', 1)]
        """
        return [func(node) for node in self.dfs_preorder()]

    def prune(self, predicate: Callable[['_HKey'], bool]) -> '_HKey':
        """
        Create a new tree with nodes filtered by predicate.

        Returns a new tree containing only nodes (and their ancestors) that
        match the predicate. This is useful for creating filtered views.

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that returns True for nodes to keep

        Returns
        -------
        _HKey
            New pruned tree (root node)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> # Keep only paths containing 'b'
        >>> pruned = root.prune(lambda n: n.key == 'b' or n.is_root or n.key == 'a')
        >>> pruned.get_all_paths()
        [['a'], ['a', 'b']]

        Notes
        -----
        The pruned tree maintains the hierarchical structure but excludes
        branches that don't lead to matching nodes.
        """
        new_root = _HKey(self.key, is_root=self.is_root)

        def copy_if_match(source: _HKey, target: _HKey) -> bool:
            """Recursively copy matching nodes."""
            has_matching_child = False

            for child in source.children:
                if predicate(child):
                    new_child = target.add_child(child.key)
                    copy_if_match(child, new_child)
                    has_matching_child = True
                elif copy_if_match(child, target):
                    # Child doesn't match but has matching descendants
                    new_child = target.add_child(child.key)
                    copy_if_match(child, new_child)
                    has_matching_child = True

            return has_matching_child or predicate(source)

        copy_if_match(self, new_root)
        return new_root

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the tree.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing various tree statistics

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}, 'd': 2})
        >>> stats = root.get_statistics()
        >>> stats['total_nodes']
        4
        >>> stats['max_depth']
        2
        >>> stats['leaf_count']
        2
        """
        all_nodes = list(self.dfs_preorder())
        leaves = list(self.iter_leaves())

        # Calculate branching factors
        non_leaf_nodes = [n for n in all_nodes if n.has_children()]
        avg_branching = (sum(len(n.children) for n in non_leaf_nodes) / len(non_leaf_nodes)
                         if non_leaf_nodes else 0)

        return {
            'total_nodes': len(all_nodes),
            'leaf_count': len(leaves),
            'max_depth': self.get_max_depth(),
            'avg_branching_factor': round(avg_branching, 2),
            'total_paths': len(self.get_all_paths()),
            'levels': self.get_max_depth() + 1 if not self.is_root else self.get_max_depth()
        }

    # ========================================================================
    # Graph Theory & Structure Validation
    # ========================================================================

    def has_cycles(self) -> Tuple[bool, Optional[List['_HKey']]]:
        """
        Check if the tree contains cycles (should not in a proper tree).

        Detects cycles by tracking visited nodes during DFS. A cycle exists
        if we encounter a node that's already in the current path (back edge).

        Returns
        -------
        Tuple[bool, Optional[List[_HKey]]]
            (has_cycle, cycle_path) where cycle_path is the nodes forming the cycle

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}})
        >>> has_cycle, path = root.has_cycles()
        >>> has_cycle
        False

        >>> # Manually create a cycle (should not happen normally)
        >>> root = _HKey('a')
        >>> child = root.add_child('b')
        >>> # In a proper tree, this wouldn't happen

        Notes
        -----
        In a properly constructed _HKey tree, this should always return False.
        This method is useful for validation and debugging.

        See Also
        --------
        is_valid_tree : Complete tree validation
        is_dag : Check if structure is a Directed Acyclic Graph
        """
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        cycle_path: List[_HKey] = []

        def dfs_cycle_detect(node: _HKey, path: List[_HKey]) -> bool:
            node_id = id(node)
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node)

            for child in node.children:
                child_id = id(child)

                if child_id not in visited:
                    if dfs_cycle_detect(child, path):
                        return True
                elif child_id in rec_stack:
                    # Cycle detected
                    cycle_start = next(i for i, n in enumerate(path) if id(n) == child_id)
                    cycle_path.extend(path[cycle_start:] + [child])
                    return True

            path.pop()
            rec_stack.remove(node_id)
            return False

        has_cycle = dfs_cycle_detect(self, [])
        return has_cycle, cycle_path if has_cycle else None

    def is_dag(self) -> bool:
        """
        Check if the structure is a Directed Acyclic Graph (DAG).

        A DAG is a directed graph with no cycles. All trees are DAGs,
        but not all DAGs are trees (DAGs can have multiple parents per node).

        Returns
        -------
        bool
            True if structure is acyclic (is a DAG)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> root.is_dag()
        True

        See Also
        --------
        has_cycles : Detect cycles with path information
        is_valid_tree : Check if it's a valid tree structure
        """
        has_cycle, _ = self.has_cycles()
        return not has_cycle

    def is_valid_tree(self) -> Tuple[bool, List[str]]:
        """
        Validate that this is a proper tree structure.

        Checks multiple tree properties:

        * No cycles (acyclic)
        * Each non-root node has exactly one parent
        * Single root (or forest with explicit root marker)
        * All nodes reachable from root

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues) where issues describes any problems found

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}})
        >>> is_valid, issues = root.is_valid_tree()
        >>> is_valid
        True
        >>> issues
        []

        See Also
        --------
        has_cycles : Check for cycles
        check_parent_consistency : Verify parent references
        """
        issues: List[str] = []

        # Check for cycles
        has_cycle, cycle = self.has_cycles()
        if has_cycle:
            issues.append(f"Cycle detected: {[n.key for n in cycle] if cycle else 'unknown'}")

        # Check parent consistency
        parent_issues = self.check_parent_consistency()
        issues.extend(parent_issues)

        # Check single root
        if not self.is_root:
            if self.parent is None:
                issues.append("Non-root node has no parent")

        # Check all nodes have valid parent references
        for node in self.dfs_preorder():
            if not node.is_root and node.parent is None:
                issues.append(f"Node {node.key} has no parent but is not marked as root")

            # Verify parent's children contain this node
            if node.parent and not node.is_root:
                if node not in node.parent.children:
                    issues.append(f"Node {node.key} not in parent's children list")

        return len(issues) == 0, issues

    def check_parent_consistency(self) -> List[str]:
        """
        Check that parent-child relationships are consistent.

        Verifies that:

        * Each child's parent reference points to the correct parent
        * Each parent's children list contains the child

        Returns
        -------
        List[str]
            List of inconsistency messages (empty if consistent)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}})
        >>> issues = root.check_parent_consistency()
        >>> len(issues)
        0
        """
        issues: List[str] = []

        for node in self.dfs_preorder():
            for child in node.children:
                if child.parent != node:
                    issues.append(
                        f"Inconsistent parent: child {child.key} has parent "
                        f"{child.parent.key if child.parent else 'None'} but is child of {node.key}"
                    )

        return issues

    def is_complete_tree(self) -> bool:
        """
        Check if this is a complete tree.

        A complete tree is a tree where all levels are fully filled except
        possibly the last level, which is filled from left to right.

        Returns
        -------
        bool
            True if tree is complete

        Examples
        --------
        >>> # Complete tree: all levels filled
        >>> root = _HKey('a')
        >>> root.add_child('b')
        >>> root.add_child('c')
        >>> root.is_complete_tree()
        True

        >>> # Incomplete: last level not filled left-to-right
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> c.add_child('d')  # Only right child has children
        >>> root.is_complete_tree()
        False

        Notes
        -----
        This uses BFS to check level-by-level filling.

        See Also
        --------
        is_perfect_tree : Check if perfectly balanced
        is_balanced : Check if height-balanced
        """
        if not self.has_children():
            return True

        queue: deque = deque([self])
        found_incomplete = False

        while queue:
            node = queue.popleft()

            for child in node.children:
                if found_incomplete:
                    # After finding a node that's not full, no nodes should have children
                    if child.has_children():
                        return False
                queue.append(child)

            # If this node doesn't have maximum children, mark as incomplete
            if not node.is_root and len(node.children) < 2:
                found_incomplete = True

        return True

    def is_perfect_tree(self) -> bool:
        """
        Check if this is a perfect tree (all leaves at same depth, all internal nodes have 2 children).

        A perfect tree is both complete and full:

        * All leaves are at the same depth
        * All internal nodes have exactly 2 children

        Returns
        -------
        bool
            True if tree is perfect

        Examples
        --------
        >>> # Perfect tree with 2 children per node
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> b.add_child('d')
        >>> b.add_child('e')
        >>> c.add_child('f')
        >>> c.add_child('g')
        >>> root.is_perfect_tree()
        True

        Notes
        -----
        This assumes binary tree structure. For n-ary trees, this checks
        if all internal nodes have the same number of children and all
        leaves are at the same depth.

        See Also
        --------
        is_complete_tree : Less strict completeness check
        is_balanced : Height-balanced check
        """
        leaves = list(self.iter_leaves())
        if not leaves:
            return True

        # All leaves should be at the same depth
        first_leaf_depth = leaves[0].get_depth()
        if not all(leaf.get_depth() == first_leaf_depth for leaf in leaves):
            return False

        # All internal nodes should have the same number of children
        internal_nodes = [n for n in self.dfs_preorder() if n.has_children() and not n.is_root]
        if not internal_nodes:
            return True

        first_children_count = len(internal_nodes[0].children)
        return all(len(n.children) == first_children_count for n in internal_nodes)

    def is_balanced(self, threshold: int = 1) -> bool:
        """
        Check if the tree is height-balanced.

        A balanced tree is one where the heights of the two subtrees of any node
        differ by at most the threshold value.

        Parameters
        ----------
        threshold : int, optional
            Maximum allowed height difference between subtrees, by default 1

        Returns
        -------
        bool
            True if tree is balanced within the threshold

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> root.is_balanced()
        True

        >>> # Unbalanced tree
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> b.add_child('c').add_child('d').add_child('e')
        >>> root.add_child('f')
        >>> root.is_balanced(threshold=1)
        False

        See Also
        --------
        get_balance_factor : Calculate balance factor for a node
        is_perfect_tree : Check perfect balance
        """

        def check_balance(node: _HKey) -> Tuple[bool, int]:
            """Returns (is_balanced, height)"""
            if not node.has_children():
                return True, 0

            child_results = [check_balance(child) for child in node.children]

            # Check if all children are balanced
            if not all(balanced for balanced, _ in child_results):
                return False, 0

            heights = [height for _, height in child_results]
            max_height = max(heights)
            min_height = min(heights)

            # Check if this node is balanced
            if max_height - min_height > threshold:
                return False, 0

            return True, max_height + 1

        balanced, _ = check_balance(self)
        return balanced

    def get_balance_factor(self) -> int:
        """
        Get the balance factor of this node.

        Balance factor = max_child_height - min_child_height

        Returns
        -------
        int
            Balance factor (0 means perfectly balanced)

        Examples
        --------
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> b.add_child('d').add_child('e')  # Deep subtree
        >>> root.get_balance_factor()
        2

        See Also
        --------
        is_balanced : Check if tree is balanced
        """
        if not self.has_children():
            return 0

        child_depths = [child.get_max_depth() for child in self.children]
        return max(child_depths) - min(child_depths)

    def count_nodes_by_degree(self) -> Dict[int, int]:
        """
        Count nodes by their out-degree (number of children).

        Returns
        -------
        Dict[int, int]
            Dictionary mapping degree to count of nodes with that degree

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> root.count_nodes_by_degree()
        {2: 1, 0: 2}  # One node with 2 children, two nodes with 0 children

        Notes
        -----
        Degree 0 nodes are leaves. This is useful for analyzing tree branching structure.

        See Also
        --------
        get_statistics : Comprehensive tree statistics
        """
        degree_counts: Dict[int, int] = defaultdict(int)

        for node in self.dfs_preorder():
            if not node.is_root:
                degree = len(node.children)
                degree_counts[degree] += 1

        return dict(degree_counts)

    def is_binary_tree(self) -> bool:
        """
        Check if this is a binary tree (all nodes have at most 2 children).

        Returns
        -------
        bool
            True if all nodes have 0, 1, or 2 children

        Examples
        --------
        >>> root = _HKey('a')
        >>> root.add_child('b')
        >>> root.add_child('c')
        >>> root.is_binary_tree()
        True

        >>> root.add_child('d')  # Now has 3 children
        >>> root.is_binary_tree()
        False
        """
        for node in self.dfs_preorder():
            if len(node.children) > 2:
                return False
        return True

    def is_full_tree(self, n: Optional[int] = None) -> bool:
        """
        Check if this is a full tree (all nodes have 0 or n children).

        A full tree (also called proper or plane tree) has all internal nodes
        with the same number of children.

        Parameters
        ----------
        n : Optional[int], optional
            Expected number of children for internal nodes. If None, uses the
            number from the first internal node found.

        Returns
        -------
        bool
            True if tree is full

        Examples
        --------
        >>> # Full binary tree (0 or 2 children)
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> b.add_child('d')
        >>> b.add_child('e')
        >>> root.is_full_tree(n=2)
        True

        See Also
        --------
        is_binary_tree : Check if binary
        is_perfect_tree : Check if perfect
        """
        internal_nodes = [node for node in self.dfs_preorder()
                          if node.has_children() and not node.is_root]

        if not internal_nodes:
            return True

        if n is None:
            n = len(internal_nodes[0].children)

        return all(len(node.children) == n for node in internal_nodes)

    def __len__(self) -> int:
        """Return the number of direct children."""
        return len(self.children)

    def __contains__(self, key: Any) -> bool:
        """Check if a child with given key exists."""
        return any(child.key == key for child in self.children)

    def __getitem__(self, key: Any) -> '_HKey':
        """Get child by key (dict-like access)."""
        for child in self.children:
            if child.key == key:
                return child
        raise KeyError(key)

    def __iter__(self) -> Iterator['_HKey']:
        """Iterate over children."""
        return iter(self.children)

    def __repr__(self) -> str:
        if self.is_root:
            return f"_HKey(ROOT, children={len(self.children)})"
        return f"_HKey(key={self.key!r}, children={len(self.children)})"


class _StackedDict(defaultdict):
    """
    This class is an internal technical class for stacking nested dictionaries. This class is technical and is used to
    manage the processing of nested dictionaries. It inherits from defaultdict.

     .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code. Access it through ``NestedDictionary``.

       However, it could be used by developers as described in
       :doc:`usage`
    """

    def __init__(self, *args, **kwargs):
        """
        At instantiation, it has two mandatory parameters for its creation:

            * **indent**, which is used to format the object's display.
            * **default_factory**, which initializes the ``default_factory`` attribute of its parent class ``defaultdict``.
            * these mandatory parameters are stored in ``default_setup`` attribute to be propagated.

        These parameters are passed using the kwargs dictionary.

        :param args:
        :type args: iterator
        :param kwargs:
        :type kwargs: dict
        """

        ind: int = 0
        default = None
        setup = set()

        # Initialize instance attributes

        self.indent: int = 0
        "indent is used to print the dictionary with json indentation"
        self._default_setup: set = set()
        "default_setup is ued to disseminate default parameters to stacked objects"

        # Manage init parameters
        settings = kwargs.pop("default_setup", None)

        if settings is None:
            if "indent" not in kwargs:
                raise StackedKeyError("Missing 'indent' arguments", key="indent")
            else:
                ind = kwargs.pop("indent")

            if "default" not in kwargs:
                default = None
            else:
                default = kwargs.pop("default")
            setup = {("indent", ind), ("default_factory", default)}
        else:
            if not "indent" in settings.keys():
                print("verifed")
                raise StackedKeyError(
                    "Missing 'indent' argument in default settings", key="indent"
                )
            if not "default_factory" in settings.keys():
                raise StackedKeyError(
                    "Missing 'default_factory' argument in default settings",
                    key="default_factory",
                )

            for key, value in settings.items():
                setup.add((key, value))

        # Initializing instance

        super().__init__()
        self._default_setup = setup
        for key, value in self._default_setup:
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                # You cannot initialize undefined attributes
                raise StackedAttributeError(
                    f"The key {key} is not an attribute of the {self.__class__} class.",
                    attribute=key,
                )

        # Update dictionary

        if len(args):
            for item in args:
                if isinstance(item, self.__class__):
                    nested = item.deepcopy()
                elif isinstance(item, dict):
                    nested = from_dict(
                        item, self.__class__, default_setup=dict(self.default_setup)
                    )
                else:
                    nested = from_dict(
                        dict(item),
                        self.__class__,
                        default_setup=dict(self.default_setup),
                    )
                self.update(nested)

        if kwargs:
            nested = from_dict(
                kwargs,
                self.__class__,
                default_setup=dict(self.default_setup),
            )
            self.update(nested)

    @property
    def default_setup(self) -> list:
        """Return a deterministic, list-based view of the internal setup set.
        Order: 'indent', 'default_factory', then other keys sorted alphabetically.
        """
        priority = ["indent", "default_factory"]
        # Convert internal set of tuples to dict to deduplicate and access by key
        d = {k: v for (k, v) in self._default_setup}
        ordered: list = []
        for p in priority:
            if p in d:
                ordered.append((p, d[p]))
        remaining = sorted(
            [(k, v) for (k, v) in self._default_setup if k not in priority],
            key=lambda kv: kv[0],
        )
        ordered.extend(remaining)
        return ordered

    @default_setup.setter
    def default_setup(self, value) -> None:
        """Accept dict, list[tuple], or set[tuple] and store internally as a set."""
        if isinstance(value, dict):
            items = value.items()
        else:
            items = value
        self._default_setup = set(items)

    def __str__(self, padding=0) -> str:
        """
        Override __str__ to converts a nested dictionary to a string in json like format

        :param padding: whitespace indentation of dictionary content
        :type padding: int
        :return: a string in json like format
        :rtype: str
        """

        d_str = "{\n"
        padding += self.indent

        for key, value in self.items():
            if isinstance(value, _StackedDict):
                d_str += indent(
                    str(key) + " : " + value.__str__(padding), padding * " "
                )
            else:
                d_str += indent(str(key) + " : " + str(value), padding * " ")
            d_str += ",\n"

        d_str += "}"

        return d_str

    def __copy__(self) -> _StackedDict:
        """
        Override __copy__ to create a shallow copy of a stacked dictionary.

        :return: a shallow copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        new = self.__class__(default_setup=dict(self.default_setup))
        for key, value in self.items():
            new[key] = value
        return new

    def __deepcopy__(self) -> _StackedDict:
        """
        Override __deepcopy__ to create a complete copy of a stacked dictionary.

        :return: a complete copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        return from_dict(
            self.to_dict(), self.__class__, default_setup=dict(self.default_setup)
        )

    def __setitem__(self, key, value) -> None:
        """
        Override __setitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :param value: value to set
        :type value: object
        :return: None
        :rtype: None
        :raises StackedTypeError: if a nested list is found within the key
        """
        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise StackedTypeError(
                        "Nested lists are not allowed as keys in _StackedDict.",
                        expected_type=str,
                        actual_type=list,
                        path=key[: key.index(sub_key)],
                    )

            # Handle hierarchical keys
            current = self
            for sub_key in key[:-1]:  # Traverse the hierarchy
                if sub_key not in current or not isinstance(
                    current[sub_key], _StackedDict
                ):
                    current[sub_key] = self.__class__(
                        default_setup=dict(self.default_setup)
                    )
                current = current[sub_key]
            current[key[-1]] = value
        else:
            # Flat keys are handled as usual
            super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Override __getitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :return: value
        :rtype: object
        :raises StackedTypeError: if a nested list is found within the key
        """

        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise StackedTypeError(
                        "Nested lists are not allowed as keys in _StackedDict.",
                        expected_type=str,
                        actual_type=list,
                        path=key[: key.index(sub_key)],
                    )

            # Handle hierarchical keys
            current = self
            for sub_key in key:
                current = current[sub_key]
            return current

        # if isinstance(key, str) and key in self.__dict__.keys():
        #    return self.__getattribute__(key)
        # else:
        return super().__getitem__(key)

    def __delitem__(self, key):
        """
        Override __delitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :return: None
        :rtype: None
        """
        if isinstance(
            key, list
        ):  # Une liste est interprétée comme une hiérarchie de clés
            current = self
            parents = []
            for sub_key in key[:-1]:  # Parcourt tous les sous-clés sauf la dernière
                parents.append(
                    (current, sub_key)
                )  # Garde une trace des parents pour nettoyer ensuite
                current = current[sub_key]
            del current[key[-1]]  # Supprime la dernière clé
            # Nettoie les parents s'ils deviennent vides
            for parent, sub_key in reversed(parents):
                if not parent[sub_key]:
                    del parent[sub_key]
        else:  # Autres types traités comme des clés simples
            super().__delitem__(key)

    def __eq__(self, other):
        """
        Override __eq__ to compare only structural content, not configuration.
        Two instances are equal if they are the same class and their expanded dict
        representations are equal, regardless of indent/default_factory settings.
        """
        if not isinstance(other, type(self)):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def __ne__(self, other):
        """
        Override __ne__ to handle hierarchical keys.
        """
        return not self.__eq__(other)

    def similar(self, other):
        """
        Override __eq__ to handle hierarchical keys.
        """
        if not isinstance(other, _StackedDict):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def isomorph(self, other):
        """
        Override __eq__ to handle hierarchical keys.
        """
        if not isinstance(other, (dict, _StackedDict)):
            return False
        return compare_dict(self.to_dict(), dict(other))

    def unpacked_items(self) -> Generator:
        """
        This method de-stacks items from a nested dictionary. It calls internal unpack_items() function.

        :return: generator that yields items from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield key, value

    def unpacked_keys(self) -> Generator:
        """
        This method de-stacks keys from a nested dictionary and return them as keys. It calls internal unpack_items()
        function.

        :return: generator that yields keys from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield key

    def unpacked_values(self) -> Generator:
        """
        This method de-stacks values from a nested dictionary and return them as values. It calls internal
        unpack_items() function.

        :return: generator that yields values from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield value

    def to_dict(self) -> dict:
        """
        This method converts a nested dictionary to a classical dictionary

        :return: a dictionary
        :rtype: dict
        """
        unpacked_dict = {}
        for key in self.keys():
            if isinstance(self[key], _StackedDict):
                unpacked_dict[key] = self[key].to_dict()
            else:
                unpacked_dict[key] = self[key]
        return unpacked_dict

    def copy(self) -> _StackedDict:
        """
        This method copies stacked dictionaries to a copy of the dictionary.
        :return: a shallow copy of the dictionary
        :rtype: _StackedDict: a _StackedDict of subclasses of _StackedDict
        """
        return self.__copy__()

    def deepcopy(self) -> _StackedDict:
        """
        This method copies a stacked dictionaries to a deep copy of the dictionary.

        :return: a deep copy of the dictionary
        :rtype: _StackedDict: a _StackedDict of subclasses of _StackedDict
        """

        return self.__deepcopy__()

    def pop(self, key: Union[Any, List[Any]], default=None) -> Any:
        """
        Removes the specified key (or hierarchical key) and returns its value.
        If the key does not exist, returns the default value if provided, or raises a KeyError.

        :param key: The key or hierarchical key to remove.
        :type key: Union[Any, List[Any]]
        :param default: The value to return if the key does not exist.
        :type default: Any
        :return: The value associated with the removed key.
        :rtype: Any
        :raises StackedKeyError: If the key does not exist and no default is provided.
        """
        if isinstance(key, list):
            # Handle hierarchical keys
            current = self
            parents = []  # Track parent dictionaries for cleanup
            for sub_key in key[:-1]:  # Traverse up to the last key
                if sub_key not in current:
                    if default is not None:
                        return default
                    raise StackedKeyError(
                        f"Key path {key} does not exist.", key=key, path=key[:-1]
                    )
                parents.append((current, sub_key))
                current = current[sub_key]

            # Pop the final key
            if key[-1] in current:
                value = current.pop(key[-1])
                # Clean up empty parents
                for parent, sub_key in reversed(parents):
                    if not parent[sub_key]:  # Remove empty dictionaries
                        parent.pop(sub_key)
                return value
            else:
                if default is not None:
                    return default
                raise StackedKeyError(
                    f"Key path {key} does not exist.", key=key[-1], path=key[:-1]
                )
        else:
            # Handle flat keys
            return super().pop(key, default)

    def popitem(self):
        """
        Removes and returns the last item in the most deeply nested dictionary as a (path, value) pair.
        The path is represented as a list of keys leading to the value.
        If the dictionary is empty, raises a KeyError.

        The method follows a depth-first search (DFS) traversal to locate the last item,
        removing it from the nested structure before returning.

        :return: A tuple containing the hierarchical path (list of keys) and the value.
        :rtype: tuple
        :raises StackedIndexError: If the dictionary is empty.
        """
        if not self:  # Handle empty dictionary
            raise StackedIndexError("popitem(): _StackedDict is empty")

        # Initialize a stack to traverse the dictionary
        stack = [(self, [])]  # Each entry is (current_dict, current_path)

        while stack:
            current, path = stack.pop()  # Get the current dictionary and path

            if isinstance(current, dict):  # Ensure we are at a dictionary level
                keys = list(current.keys())
                if keys:  # If there are keys in the current dictionary
                    key = keys[-1]  # Select the last key
                    new_path = path + [key]  # Update the path
                    stack.append((current[key], new_path))  # Continue with this branch
            else:
                # If the current value is not a dictionary, we have reached a leaf
                break

        # Remove the item from the dictionary using the found path
        container = self  # Start from the root dictionary
        for key in path[:-1]:  # Traverse to the parent of the target key
            container = container[key]
        value = container.pop(path[-1])  # Remove the last key-value pair

        return path, value

    def update(self, dictionary: dict = None, **kwargs) -> None:
        """
        Updates a stacked dictionary with key/value pairs from a dictionary or keyword arguments.

        :param dictionary: A dictionary with key/value pairs to update.
        :type dictionary: dict
        :param kwargs: Additional key/value pairs to update.
        :type kwargs: dict
        :return: None
        """
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, _StackedDict):
                    value.indent = self.indent
                    value.default_factory = self.default_factory
                    self[key] = value
                elif isinstance(value, dict):
                    nested_dict = from_dict(
                        value, self.__class__, default_setup=dict(self.default_setup)
                    )
                    self[key] = nested_dict
                else:
                    self[key] = value

        for key, value in kwargs.items():
            if isinstance(value, _StackedDict):
                value.indent = self.indent
                value.default_factory = self.default_factory
                self[key] = value
            elif isinstance(value, dict):
                nested_dict = from_dict(
                    value, self.__class__, default_setup=dict(self.default_setup)
                )
                self[key] = nested_dict
            else:
                self[key] = value

    def is_key(self, key: Any) -> bool:
        """
        Checks if a key exists at any level in the _StackedDict hierarchy using unpack_items().
        This works for both flat keys (e.g., 1) and hierarchical keys (e.g., [1, 2, 3]).

        :param key: A key to check. Can be a single key or a part of a hierarchical path.
        :return: True if the key exists at any level, False otherwise.
        """
        # Normalize the key (convert lists to tuples for uniform comparison)
        if isinstance(key, list):
            raise StackedKeyError("This function manages only atomic keys", key=key)

        # Check directly if the key exists in unpacked keys
        return any(key in keys for keys in self.unpacked_keys())

    def occurrences(self, key: Any) -> int:
        """
        Returns the Number of occurrences of a key in a stacked dictionary including 0 if the key is not a keys in a
        stacked dictionary.

        :param key: A possible key in a stacked dictionary.
        :type key: Any
        :return: Number of occurrences or 0
        :rtype: int
        """
        __occurrences = 0
        for stacked_keys in self.unpacked_keys():
            if key in stacked_keys:
                for occ in stacked_keys:
                    if occ == key:
                        __occurrences += 1
        return __occurrences

    def key_list(self, key: Any) -> list:
        """
        returns the list of unpacked keys containing the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).

        :param key: a possible key in a stacked dictionary.
        :type key: Any
        :return: A list of unpacked keys containing the key from the stacked dictionary.
        :rtype: list
        :raise StackedKeyError: if a key is not in a stacked dictionary.
        """
        __key_list = []

        if self.is_key(key):
            for keys in self.unpacked_keys():
                if key in keys:
                    __key_list.append(keys)
        else:
            raise StackedKeyError(
                f"Cannot find the key: {key} in the stacked dictionary", key=key
            )

        return __key_list

    def items_list(self, key: Any) -> list:
        """
        returns the list of unpacked items associated to the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).

        :param key: a possible key in a stacked dictionary.
        :type key: Any
        :return: A list of unpacked items associated the key from the stacked dictionary.
        :rtype: list
        :raise StackedKeyError: if a key is not in a stacked dictionary.
        """
        __items_list = []

        if self.is_key(key):
            for items in self.unpacked_items():
                if key in items[0]:
                    __items_list.append(items[1])
        else:
            raise StackedKeyError(
                f"Cannot find the key: {key} in the stacked dictionary", key=key
            )

        return __items_list

    def dict_paths(self) -> _Paths:
        """
        Returns a view object for all hierarchical paths in the _StackedDict.

        .. deprecated:: 0.9

            This function will be removed in a future release.
            Use :meth:`paths` instead.
        """
        return _Paths(self)

    def paths(self) -> _Paths:
        """
        Returns a view object for all hierarchical paths in the _StackedDict.
        """
        return _Paths(self)

    def dict_search(self) -> DictSearch:
        """
        Returns a DictSearch object factorizing the hierarchical paths of the _StackedDict.
        """
        return DictSearch.from_dict_paths(self.dict_paths())

    def dfs(self, node=None, path=None) -> Generator[Tuple[List, Any], None, None]:
        """
        Depth-First Search (DFS) traversal of the stacked dictionary.

        This method recursively traverses the dictionary in a depth-first manner.
        It yields each hierarchical path as a list and its corresponding value.

        :param node: The current dictionary node being traversed. Defaults to the root if None.
        :type node: Optional[dict]
        :param path: The current hierarchical path being constructed. Defaults to an empty list if None.
        :type path: Optional[List]
        :return: A generator that yields tuples of hierarchical paths and their corresponding values.
        :rtype: Generator[Tuple[List, Any], None, None]
        """
        if node is None:
            node = self
        if path is None:
            path = []

        for key, value in node.items():
            current_path = path + [key]
            yield (current_path, value)
            if isinstance(value, dict):  # Check if the value is a nested dictionary
                yield from self.dfs(
                    value, current_path
                )  # Recursively traverse the nested dictionary

    def bfs(self) -> Generator[Tuple[Tuple, Any], None, None]:
        """
        Breadth-First Search (BFS) traversal of the stacked dictionary.

        This method iteratively traverses the dictionary in a breadth-first manner.
        It uses a queue to ensure that all nodes at a given depth are visited before moving deeper.

        :return: A generator that yields tuples of hierarchical paths (as tuples) and their corresponding values.
        :rtype: Generator[Tuple[Tuple, Any], None, None]
        """
        queue = deque(
            [((), self)]
        )  # Start with an empty path and the top-level dictionary
        while queue:
            path, current_dict = queue.popleft()  # Dequeue the first dictionary
            for key, value in current_dict.items():
                new_path = path + (key,)  # Extend the path with the current key
                if isinstance(
                    value, _StackedDict
                ):  # Check if the value is a nested _StackedDict
                    queue.append(
                        (new_path, value)
                    )  # Enqueue the nested dictionary with its path
                else:
                    yield new_path, value  # Yield the current path and value

    def height(self) -> int:
        """
        Computes the height of the _StackedDict, defined as the length of the longest path.

        :return: The height of the dictionary.
        :rtype: int
        """
        return max((len(path) for path in self.dict_paths()), default=0)

    def size(self) -> int:
        """
        Computes the size of the _StackedDict, defined as the total number of keys (nodes) in the structure.

        :return: The total number of nodes in the dictionary.
        :rtype: int
        """
        return sum(1 for _ in self.unpacked_items())

    def leaves(self) -> list:
        """
        Extracts the leaf nodes of the _StackedDict.

        :return: A list of leaf values.
        :rtype: list
        """
        return [value for _, value in self.dfs() if not isinstance(value, _StackedDict)]

    def is_balanced(self) -> bool:
        """
        Checks if the _StackedDict is balanced.
        A balanced dictionary is one where the height difference between any two subtrees is at most 1.

        :return: True if balanced, False otherwise.
        :rtype: bool
        """

        def check_balance(node):
            if not isinstance(node, _StackedDict) or not node:
                return 0, True  # Height, is_balanced
            heights = []
            for key in node:
                height, balanced = check_balance(node[key])
                if not balanced:
                    return 0, False
                heights.append(height)
            if not heights:
                return 1, True
            return max(heights) + 1, max(heights) - min(heights) <= 1

        _, balanced = check_balance(self)
        return balanced

    def ancestors(self, value):
        """
        Finds the ancestors (keys) of a given value in the nested dictionary.

        :param value: The value to search for in the nested dictionary.
        :type value: Any
        :return: A list of keys representing the path to the value.
        :rtype: List[Any]
        :raises ValueError: If the value is not found in the dictionary.
        """
        for path, val in self.dfs():
            if val == value:
                return path[
                    :-1
                ]  # Return all keys except the last one (the direct key of the value)

        raise StackedValueError(
            f"Value {value} not found in the dictionary.", value=value
        )


"""Public class section"""

class _Paths:
    """
    A lazy view providing access to all hierarchical paths in a nested dictionary.

    Similar to the standard ``dict_keys`` view, but designed specifically for
    hierarchical paths in nested dictionaries. Uses an internal ``_HKey`` tree
    for efficient path generation and querying.

    The view provides lazy iteration over all paths without storing them in memory,
    and leverages the optimized tree structure of ``_HKey`` for fast operations.

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code. Access it through ``NestedDictionary``

    Parameters
    ----------
    stacked_dict : _StackedDict
        The nested dictionary to create a view for

    Attributes
    ----------
    _stacked_dict : _StackedDict
        Reference to the source dictionary
    _hkey : _HKey
        Internal tree structure for efficient path operations (lazy-built)

    Examples
    --------
    >>> data = _StackedDict({'a': {'b': 1}, 'c': 2})
    >>> paths = _Paths(data)
    >>> list(paths)
    [['a'], ['a', 'b'], ['c']]
    >>> ['a', 'b'] in paths
    True
    >>> len(paths)
    3

    See Also
    --------
    DictSearch : Factorized representation of paths
    _StackedDict.paths : building paths for nested dictionaries.
    """

    def __init__(self, stacked_dict: _StackedDict):
        self._stacked_dict = stacked_dict
        self._hkey: Optional[_HKey] = None  # Lazy initialization

    def _ensure_hkey(self) -> _HKey:
        """
        Ensure _HKey tree is built (lazy initialization).

        Returns
        -------
        _HKey
            The built tree structure
        """
        if self._hkey is None:
            self._hkey = _HKey.build_forest(self._stacked_dict)
        return self._hkey

    def __iter__(self) -> Iterator[List[Any]]:
        """
        Iterate over all hierarchical paths.

        Yields
        ------
        List[Any]
            Each path as a list of keys from root to node

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}}))
        >>> for path in paths:
        ...     print(path)
        ['a']
        ['a', 'b']
        """
        hkey = self._ensure_hkey()
        return iter(hkey.get_all_paths())

    def __len__(self) -> int:
        """
        Return the number of paths.

        Returns
        -------
        int
            Total number of hierarchical paths

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}, 'c': 2}))
        >>> len(paths)
        3
        """
        hkey = self._ensure_hkey()
        return len(hkey.get_all_paths())

    def __contains__(self, path: List[Any]) -> bool:
        """
        Check if a path exists.

        Uses the optimized tree structure for O(n) lookup where n is path length,
        much faster than iterating all paths.

        Parameters
        ----------
        path : List[Any]
            Path to verify

        Returns
        -------
        bool
            True if path exists, False otherwise

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}}))
        >>> ['a', 'b'] in paths
        True
        >>> ['a', 'c'] in paths
        False
        """
        hkey = self._ensure_hkey()
        return hkey.find_by_path(path) is not None

    def __eq__(self, other: Any) -> bool:
        """
        Compare two DictPaths for set-wise equality (order-independent).

        Parameters
        ----------
        other : Any
            Another DictPaths or iterable of paths

        Returns
        -------
        bool
            True if both contain the same paths (regardless of order)

        Examples
        --------
        >>> paths1 = _Paths(_StackedDict({'a': 1}))
        >>> paths2 = _Paths(_StackedDict({'a': 1}))
        >>> paths1 == paths2
        True
        """
        try:
            self_set = {tuple(p) for p in self}
            other_iter = other if not isinstance(other, _Paths) else iter(other)
            other_set = {tuple(p) for p in other_iter}
            return self_set == other_set
        except TypeError:
            return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """
        Check inequality between DictPaths objects.

        Parameters
        ----------
        other : Any
            Another DictPaths or iterable

        Returns
        -------
        bool
            True if not equal
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String showing class name and paths
        """
        return f"{self.__class__.__name__}({list(self)})"


    def get_children(self, path: List[Any]) -> List[Any]:
        """
        Get child keys at the next level after the given path.

        Parameters
        ----------
        path : List[Any]
            Path to query

        Returns
        -------
        List[Any]
            List of child keys, empty if path not found

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1, 'c': 2}}))
        >>> paths.get_children(['a'])
        ['b', 'c']
        >>> paths.get_children(['a', 'b'])
        []

        See Also
        --------
        has_children : Check if path has children
        """
        hkey = self._ensure_hkey()
        node = hkey.find_by_path(path)
        if node is None:
            return []
        return node.get_child_keys()

    def has_children(self, path: List[Any]) -> bool:
        """
        Check if a path has any children.

        Parameters
        ----------
        path : List[Any]
            Path to check

        Returns
        -------
        bool
            True if path has children

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}}))
        >>> paths.has_children(['a'])
        True
        >>> paths.has_children(['a', 'b'])
        False
        """
        hkey = self._ensure_hkey()
        node = hkey.find_by_path(path)
        if node is None:
            return False
        return node.has_children()

    def get_subtree_paths(self, prefix: List[Any]) -> List[List[Any]]:
        """
        Get all paths that start with the given prefix.

        Parameters
        ----------
        prefix : List[Any]
            Prefix to filter by

        Returns
        -------
        List[List[Any]]
            List of paths with this prefix

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': {'c': 1}, 'd': 2}}))
        >>> paths.get_subtree_paths(['a'])
        [['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'd']]

        See Also
        --------
        filter_paths : Filter paths by predicate
        """
        hkey = self._ensure_hkey()
        node = hkey.find_by_path(prefix)
        if node is None:
            return []

        subtree_paths = node.get_all_paths()
        if not node.is_root:
            base_path = node.get_path()
            return [base_path] + subtree_paths if subtree_paths else [base_path]

        return subtree_paths

    def filter_paths(self, predicate: Callable[[List[Any]], bool]) -> List[List[Any]]:
        """
        Filter paths based on a predicate function.

        Parameters
        ----------
        predicate : Callable[[List[Any]], bool]
            Function that returns True for paths to include

        Returns
        -------
        List[List[Any]]
            Filtered list of paths

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}, 'c': 2}))
        >>> # Get paths longer than 1
        >>> paths.filter_paths(lambda p: len(p) > 1)
        [['a', 'b']]

        See Also
        --------
        get_subtree_paths : Filter by prefix
        """
        hkey = self._ensure_hkey()
        return hkey.filter_paths(predicate)

    def get_depth(self) -> int:
        """
        Get the maximum depth of paths.

        Returns
        -------
        int
            Maximum path length

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': {'c': 1}}}))
        >>> paths.get_depth()
        3
        """
        hkey = self._ensure_hkey()
        return hkey.get_max_depth()

    def get_leaf_paths(self) -> List[List[Any]]:
        """
        Get all leaf paths (paths with no children).

        Returns
        -------
        List[List[Any]]
            List of leaf paths

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}, 'c': 2}))
        >>> paths.get_leaf_paths()
        [['a', 'b'], ['c']]
        """
        hkey = self._ensure_hkey()
        return [node.get_path() for node in hkey.iter_leaves()]

    def to_search(self) -> 'DictSearch':
        """
        Convert this DictPaths to a DictSearch.

        Returns
        -------
        DictSearch
            DictSearch with complete coverage of these paths

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}}))
        >>> search = paths.to_search()
        >>> search.to_factorized()
        [['a', ['b']]]

        See Also
        --------
        DictSearch.from_dict_paths : Alternative construction method
        """
        hkey = self._ensure_hkey()
        return DictSearch(hkey)


class DictSearch:
    """
    Factorized representation of hierarchical dictionary paths.

    ``DictSearch`` provides a compact, factorized representation of nested dictionary
    paths by grouping sibling keys under their common parent. It maintains both a
    public factorized structure for manual construction and an internal ``_HKey`` tree
    for efficient operations.

    **Factorization Features:**

    * **Path compression**: Chains with single children become tuples: ``('env', 'production')``
    * **Sibling grouping**: Children are grouped in lists: ``['a', ['b', 'c']]``
    * **Manual construction**: Can be created directly with factorized structure
    * **Automatic construction**: Built from _Paths or _StackedDict with optimization

    The relationship between :class:`_Paths` and ``DictSearch`` is bijective when
    constructed from a complete :class:`_Paths`, meaning they can be converted
    back and forth without information loss. However, ``DictSearch`` can also represent
    partial coverage for selective tree traversal operations.

    Parameters
    ----------
    structure : Optional[List[Any]], optional
        Factorized structure as nested lists for manual construction

    Attributes
    ----------
    structure : List[Any]
        Public factorized representation (readable/writable for manual use)
    _hkey : Optional[_HKey]
        Internal tree structure (private, lazy-built from structure when needed)

    Examples
    --------
    Manual construction with simple structure:

    >>> search = DictSearch([['a'], ['b', ['c', 'd']]])
    >>> search.structure
    [['a'], ['b', ['c', 'd']]]
    >>> search.get_all_paths()
    [['a'], ['b'], ['b', 'c'], ['b', 'd']]

    Manual construction with tuple compression:

    >>> search = DictSearch([[('env', 'prod'), ['db', 'cache']]])
    >>> search.get_all_paths()
    [[('env', 'prod')], [('env', 'prod'), 'db'], [('env', 'prod'), 'cache']]

    Automatic construction from _StackedDict:

    >>> data = _StackedDict({'a': 1, 'b': {'c': 2, 'd': 3}})
    >>> search = DictSearch.from_stacked_dict(data)
    >>> search.structure
    [['a'], ['b', ['c', 'd']]]

    Automatic construction with chain compression:

    >>> data = _StackedDict({'env': {'prod': {'db': 1}}})
    >>> search = DictSearch.from_stacked_dict(data)
    >>> search.structure
    [[('env', 'prod', 'db')]]

    Partial coverage:

    >>> paths = data.paths()
    >>> partial = DictSearch([['b', ['c']]])
    >>> partial.is_partial_coverage(paths)
    True

    See Also
    --------
    _Paths : View of all paths in a nested dictionary
    _HKey : Internal tree node structure (private)
    _StackedDict : Base dictionary class with search() method
    """

    __slots__ = ('structure', '_hkey', '_structure_dirty')

    def __init__(self, structure: Optional[List[Any]] = None) -> None:
        """
        Initialize a DictSearch with a factorized structure.

        Parameters
        ----------
        structure : Optional[List[Any]], optional
            Factorized structure as nested lists. If None, creates empty structure.

        Examples
        --------
        >>> search = DictSearch([['a'], ['b', ['c', 'd']]])
        >>> len(search)
        2
        """
        self.structure: List[Any] = structure if structure is not None else []
        self._hkey: Optional[_HKey] = None  # Lazy-built from structure
        self._structure_dirty: bool = False  # Track if structure changed after _hkey built

    @classmethod
    def from_stacked_dict(cls, stacked_dict: Dict) -> 'DictSearch':
        """
        Create a DictSearch from a nested dictionary with automatic optimization.

        This method builds an optimized factorized structure by:

        1. Creating an _HKey tree from the dictionary
        2. Compressing single-child chains into tuples: ``(key1, key2, key3)``
        3. Grouping sibling keys into lists: ``[parent, [child1, child2]]``

        This is the most efficient construction method.

        Parameters
        ----------
        stacked_dict : Dict
            A _StackedDict or dict instance

        Returns
        -------
        DictSearch
            New DictSearch with optimized factorization

        Examples
        --------
        Chain compression:

        >>> data = _StackedDict({'env': {'prod': {'db': 1}}})
        >>> search = DictSearch.from_stacked_dict(data)
        >>> search.structure
        [[('env', 'prod', 'db')]]

        Sibling grouping:

        >>> data2 = _StackedDict({'a': {'b': 1, 'c': 2}})
        >>> search2 = DictSearch.from_stacked_dict(data2)
        >>> search2.structure
        [['a', ['b', 'c']]]

        See Also
        --------
        from_dict_paths : Construct from _Paths
        from_paths_list : Construct from plain list of paths
        """
        if not isinstance(stacked_dict, dict) or len(stacked_dict) == 0:
            return cls([])

        # Build _HKey tree
        root = _HKey.build_forest(stacked_dict)

        def compress_chain(node: _HKey) -> Tuple[Any, _HKey]:
            """
            Compress single-child chains into tuples.

            Returns
            -------
            Tuple[Any, _HKey]
                (compressed_token, last_node_in_chain)
            """
            keys = [node.key]
            current = node

            while len(current.children) == 1:
                child = current.children[0]
                keys.append(child.key)
                current = child

            if len(keys) == 1:
                return keys[0], current
            else:
                return tuple(keys), current

        structure: List[Any] = []

        def walk(start: _HKey, prefix_tokens: List[Any]) -> None:
            """Walk the tree and build factorized structure."""
            # Compress from this node
            token, last = compress_chain(start)
            new_prefix = prefix_tokens + [token]

            # Get immediate children (compressed individually)
            children_tokens = []
            for child in last.children:
                child_token, _ = compress_chain(child)
                children_tokens.append(child_token)

            # Add factorized element
            if children_tokens:
                structure.append(new_prefix + [children_tokens])
            else:
                structure.append(new_prefix)

            # Recurse on each child
            for child in last.children:
                walk(child, new_prefix)

        # Start from each root child
        for child in root.children:
            walk(child, [])

        return cls(structure)

    @classmethod
    def from_dict_paths(cls, dict_paths: '_Paths') -> 'DictSearch':
        """
        Create a DictSearch from a _Paths object with automatic optimization.

        This uses the underlying _StackedDict to build an optimized structure
        with chain compression and sibling grouping.

        Parameters
        ----------
        dict_paths : _Paths
            _Paths instance to build from

        Returns
        -------
        DictSearch
            New DictSearch with complete coverage and optimization

        Examples
        --------
        >>> data = _StackedDict({'a': {'b': {'c': 1}}})
        >>> paths = data.paths()
        >>> search = DictSearch.from_dict_paths(paths)
        >>> search.structure
        [[('a', 'b', 'c')]]

        See Also
        --------
        from_stacked_dict : Direct construction from dictionary
        """
        try:
            stacked = dict_paths._stacked_dict
        except AttributeError:
            # Fallback: treat as _StackedDict directly
            stacked = dict_paths

        return cls.from_stacked_dict(stacked)

    @classmethod
    def from_paths_list(cls, paths: List[List[Any]], compress: bool = True) -> 'DictSearch':
        """
        Create a DictSearch from a plain list of paths.

        Parameters
        ----------
        paths : List[List[Any]]
            List of paths to include
        compress : bool, optional
            Whether to apply chain compression, by default True

        Returns
        -------
        DictSearch
            New DictSearch containing the specified paths

        Examples
        --------
        With compression (creates tuples for single-child chains):

        >>> paths = [['a', 'b', 'c'], ['a', 'b', 'd'], ['e']]
        >>> search = DictSearch.from_paths_list(paths)
        >>> search.structure
        [[('a', 'b'), ['c', 'd']], ['e']]

        Without compression:

        >>> search_no_compress = DictSearch.from_paths_list(paths, compress=False)
        >>> search_no_compress.structure
        [['a', ['b']], ['b', ['c', 'd']], ['e']]

        See Also
        --------
        build_search_from_paths : Module-level utility function
        """
        if compress:
            # Build tree, then use optimized from_stacked_dict logic
            root = _HKey(None, is_root=True)
            for path in paths:
                if not path:
                    continue
                current = root
                for key in path:
                    child = current.get_child(key)
                    if child is None:
                        child = current.add_child(key)
                    current = child

            # Convert tree back to dict and use from_stacked_dict
            tree_dict = root.to_dict()
            return cls.from_stacked_dict(tree_dict)
        else:
            # Simple factorization without compression
            return cls._factorize_simple(paths)

    @classmethod
    def _factorize_simple(cls, paths: List[List[Any]]) -> 'DictSearch':
        """
        Simple factorization without chain compression.

        Groups sibling keys but doesn't create tuples for chains.

        Parameters
        ----------
        paths : List[List[Any]]
            List of paths to factorize

        Returns
        -------
        DictSearch
            Factorized structure without tuple compression
        """
        factorized: List[Any] = []
        processed: Set[Tuple] = set()
        sorted_paths = sorted(paths, key=len)

        for path in sorted_paths:
            path_tuple = tuple(path)
            if path_tuple in processed:
                continue

            # Find direct children (paths extending this by exactly one key)
            children = []
            for other_path in sorted_paths:
                other_tuple = tuple(other_path)
                if (other_tuple != path_tuple and
                        len(other_path) == len(path) + 1 and
                        other_path[:len(path)] == path):

                    child_key = other_path[len(path)]
                    if child_key not in children:
                        children.append(child_key)
                    processed.add(other_tuple)

            # Build factorized entry
            if children:
                factorized.append(path + [children])
            else:
                factorized.append(path)

            processed.add(path_tuple)

        return cls(factorized)

    def _ensure_hkey(self) -> _HKey:
        """
        Ensure _HKey tree is built from structure (lazy initialization).

        Builds the tree by expanding the factorized structure and creating
        nodes for each path. Handles tuple decompression.

        Returns
        -------
        _HKey
            The built tree structure
        """
        if self._hkey is None or self._structure_dirty:
            # Build tree from factorized structure
            expanded = self.get_all_paths()
            root = _HKey(None, is_root=True)

            for path in expanded:
                if not path:
                    continue
                current = root

                for key in path:
                    # Handle tuple compression: expand tuple into chain
                    if isinstance(key, tuple):
                        for k in key:
                            child = current.get_child(k)
                            if child is None:
                                child = current.add_child(k)
                            current = child
                    else:
                        child = current.get_child(key)
                        if child is None:
                            child = current.add_child(key)
                        current = child

            self._hkey = root
            self._structure_dirty = False

        return self._hkey

    def get_all_paths(self) -> List[List[Any]]:
        """
        Expand the factorized structure to a flat list of all paths.

        This method handles:

        * Tuple decompression: ``('a', 'b')`` → ``['a'], ['a', 'b']``
        * Sibling expansion: ``['a', ['b', 'c']]`` → ``['a'], ['a', 'b'], ['a', 'c']``

        Returns
        -------
        List[List[Any]]
            List of all expanded paths

        Examples
        --------
        Simple sibling grouping:

        >>> search = DictSearch([['a'], ['b', ['c', 'd']]])
        >>> search.get_all_paths()
        [['a'], ['b'], ['b', 'c'], ['b', 'd']]

        With tuple compression:

        >>> search2 = DictSearch([[('a', 'b'), ['c', 'd']]])
        >>> search2.get_all_paths()
        [['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'd']]

        See Also
        --------
        to_dict_paths_list : Alias for this method
        """
        paths = []
        seen = set()  # Avoid duplicates

        def expand_token(token: Any, current_path: List[Any]) -> List[List[Any]]:
            """Expand a token (key or tuple) into intermediate paths."""
            if isinstance(token, tuple):
                # Expand tuple: (a, b, c) → [a], [a, b], [a, b, c]
                intermediate = []
                for i in range(len(token)):
                    path = current_path + list(token[:i + 1])
                    intermediate.append(path)
                return intermediate
            else:
                # Single key
                return [current_path + [token]]

        def expand_structure(item: Any, current_path: List[Any] = []) -> None:
            if not isinstance(item, list) or len(item) == 0:
                return

            # Check if last element is a list of children
            if isinstance(item[-1], list):
                # Node with children: ['a', 'b', ['c', 'd']]
                prefix_tokens = item[:-1]
                children = item[-1]

                # Expand prefix tokens
                expanded_prefix = current_path[:]
                for token in prefix_tokens:
                    token_paths = expand_token(token, expanded_prefix)
                    for tp in token_paths:
                        tp_tuple = tuple(tp)
                        if tp_tuple not in seen:
                            paths.append(tp)
                            seen.add(tp_tuple)
                    expanded_prefix = token_paths[-1]  # Last path is the full prefix

                # Expand each child
                for child in children:
                    child_paths = expand_token(child, expanded_prefix)
                    for cp in child_paths:
                        cp_tuple = tuple(cp)
                        if cp_tuple not in seen:
                            paths.append(cp)
                            seen.add(cp_tuple)
            else:
                # Simple path: ['a', 'b', 'c']
                expanded_path = current_path[:]
                for token in item:
                    token_paths = expand_token(token, expanded_path)
                    for tp in token_paths:
                        tp_tuple = tuple(tp)
                        if tp_tuple not in seen:
                            paths.append(tp)
                            seen.add(tp_tuple)
                    expanded_path = token_paths[-1]

        for item in self.structure:
            expand_structure(item)

        return paths

    def to_dict_paths_list(self) -> List[List[Any]]:
        """
        Alias for get_all_paths() for backward compatibility.

        Returns
        -------
        List[List[Any]]
            List of all expanded paths

        See Also
        --------
        get_all_paths : Primary method
        """
        return self.get_all_paths()

    def contains_path(self, path: List[Any]) -> bool:
        """
        Check if a specific path exists in this DictSearch.

        Parameters
        ----------
        path : List[Any]
            Path to check

        Returns
        -------
        bool
            True if path exists, False otherwise

        Examples
        --------
        >>> search = DictSearch([['a', 'b']])
        >>> search.contains_path(['a'])
        True
        >>> search.contains_path(['a', 'b'])
        True
        >>> search.contains_path(['a', 'c'])
        False
        """
        path_tuple = tuple(path)
        expanded = self.get_all_paths()
        return path_tuple in {tuple(p) for p in expanded}

    def covers_path(self, path: List[Any]) -> bool:
        """
        Alias for contains_path() for backward compatibility.

        Parameters
        ----------
        path : List[Any]
            Path to check

        Returns
        -------
        bool
            True if path is covered

        See Also
        --------
        contains_path : Primary method
        """
        return self.contains_path(path)

    def is_complete_coverage(self, dict_paths: '_Paths') -> bool:
        """
        Check if this DictSearch provides complete coverage of the given _Paths.

        Complete coverage means this DictSearch contains exactly the same
        set of paths as the _Paths, confirming a bijective relationship.

        Parameters
        ----------
        dict_paths : _Paths
            _Paths instance to compare against

        Returns
        -------
        bool
            True if coverage is complete (bijective), False otherwise

        Examples
        --------
        >>> data = _StackedDict({'a': {'b': 1}})
        >>> paths = data.paths()
        >>> complete = DictSearch.from_dict_paths(paths)
        >>> complete.is_complete_coverage(paths)
        True
        >>> partial = DictSearch([['a']])
        >>> partial.is_complete_coverage(paths)
        False
        """
        our_paths = set(tuple(p) for p in self.get_all_paths())
        their_paths = set(tuple(p) for p in dict_paths)
        return our_paths == their_paths

    def is_partial_coverage(self, dict_paths: '_Paths') -> bool:
        """
        Check if this DictSearch represents partial coverage of the given _Paths.

        Partial coverage means this DictSearch covers a proper subset of the
        paths in the _Paths.

        Parameters
        ----------
        dict_paths : _Paths
            _Paths instance to compare against

        Returns
        -------
        bool
            True if coverage is partial (proper subset), False otherwise

        Examples
        --------
        >>> data = _StackedDict({'a': 1, 'b': 2, 'c': 3})
        >>> paths = data.paths()
        >>> partial = DictSearch([['a'], ['b']])
        >>> partial.is_partial_coverage(paths)
        True
        >>> complete = DictSearch.from_dict_paths(paths)
        >>> complete.is_partial_coverage(paths)
        False
        """
        our_paths = set(tuple(p) for p in self.get_all_paths())
        their_paths = set(tuple(p) for p in dict_paths)
        return our_paths.issubset(their_paths) and our_paths != their_paths

    def __len__(self) -> int:
        """
        Return the number of top-level factorized elements.

        Returns
        -------
        int
            Number of structure elements (not total paths)

        Examples
        --------
        >>> search = DictSearch([['a'], ['b', ['c', 'd']]])
        >>> len(search)  # 2 structure elements
        2
        >>> len(search.get_all_paths())  # 4 total paths
        4
        """
        return len(self.structure)

    def __contains__(self, path: List[Any]) -> bool:
        """
        Check if a path exists in this DictSearch.

        Parameters
        ----------
        path : List[Any]
            Path to check

        Returns
        -------
        bool
            True if path exists

        Examples
        --------
        >>> search = DictSearch([['a', ['b']]])
        >>> ['a'] in search
        True
        >>> ['a', 'b'] in search
        True
        >>> ['c'] in search
        False
        """
        return self.contains_path(path)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over factorized structure elements.

        Yields
        ------
        Any
            Each factorized structure element

        Examples
        --------
        >>> search = DictSearch([['a'], ['b', ['c']]])
        >>> for item in search:
        ...     print(item)
        ['a']
        ['b', ['c']]
        """
        return iter(self.structure)

    def __getitem__(self, index: int) -> Any:
        """
        Get factorized structure element at index.

        Parameters
        ----------
        index : int
            Index to retrieve

        Returns
        -------
        Any
            Structure element at index

        Examples
        --------
        >>> search = DictSearch([['a'], ['b', ['c']]])
        >>> search[0]
        ['a']
        >>> search[1]
        ['b', ['c']]
        """
        return self.structure[index]

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String showing factorized structure

        Examples
        --------
        >>> search = DictSearch([['a'], ['b', ['c']]])
        >>> repr(search)
        "DictSearch([['a'], ['b', ['c']]])"
        """
        return f"DictSearch({self.structure})"
