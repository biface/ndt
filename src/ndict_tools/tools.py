"""
This module provides the **core technical infrastructure** for manipulating nested dictionaries. While hidden from the
package's public API, it serves as the foundation for all nested dictionary operations.

* **_StackedDict**: Base class for nested dictionary structures
* **_HKey**: Tree node representing hierarchical keys (private, optimized with tuples)
* **_Paths**: View of all paths in a nested dictionary
* **_CPaths**: Compact/factorized representation of paths in nested dictionaries

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

import warnings
from collections import defaultdict, deque
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .exception import (
    StackedAttributeError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)

MAX_DEPTH = 100

"""Internal functions"""


def compare_dict(d1, d2) -> bool:
    """
    Recursively compare two potentially nested structures for equality.

    Performs deep comparison of dictionaries, lists, tuples, sets, and scalar values.
    Two structures are considered equal if they have the same type and equal content
    at all nesting levels.

    Parameters
    ----------
    d1 : Any
        First structure to compare
    d2 : Any
        Second structure to compare

    Returns
    -------
    bool
        True if structures are identical in type and content

    Examples
    --------
    >>> compare_dict({'a': 1}, {'a': 1})
    True
    >>> compare_dict({'a': {'b': 1}}, {'a': {'b': 1}})
    True
    >>> compare_dict({'a': 1}, {'a': 2})
    False
    >>> compare_dict([1, 2], (1, 2))
    False

    Notes
    -----
    - Type checking is strict: [1, 2] != (1, 2) even with same values
    - Dictionary key sets must match exactly
    - Recursively handles nested structures of arbitrary depth
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
    Recursively flatten a nested dictionary into (path, value) pairs.

    Traverses a nested dictionary structure and yields each terminal value
    along with its hierarchical path represented as a tuple of keys.
    Empty dictionaries are preserved and yielded with their path.

    Parameters
    ----------
    dictionary : dict
        Dictionary to unpack (may be nested)

    Yields
    ------
    tuple
        (path_tuple, value) where path_tuple is a tuple of keys leading to value

    Examples
    --------
    >>> list(unpack_items({'a': 1, 'b': {'c': 2}}))
    [(('a',), 1), (('b', 'c'), 2)]
    >>> list(unpack_items({'a': {}}))
    [(('a',), {})]
    >>> list(unpack_items({'a': {'b': {'c': 1}}}))
    [(('a', 'b', 'c'), 1)]

    Notes
    -----
    - Uses depth-first traversal
    - Empty dictionaries are treated as terminal values
    - Paths are represented as immutable tuples for hashability

    See Also
    --------
    _StackedDict.unpacked_items : Method wrapper for this function
    _StackedDict.dfs : Depth-first traversal alternative
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
    Recursively convert a standard dictionary to a _StackedDict or subclass.

    This function transforms a regular nested dictionary into a _StackedDict-based
    structure, preserving the hierarchical organization while adding the enhanced
    functionality of _StackedDict. It can instantiate any _StackedDict subclass
    with custom initialization options.

    Parameters
    ----------
    dictionary : dict
        The dictionary to transform (may be nested)
    class_name : type
        The _StackedDict class (or subclass) to instantiate
    **class_options : dict
        Initialization options for the class instances.
        Must contain 'default_setup' key with configuration dict.

    Returns
    -------
    _StackedDict
        New instance of class_name containing the dictionary structure

    Raises
    ------
    StackedKeyError
        If 'default_setup' key is missing from class_options

    Examples
    --------
    >>> setup = {'default_setup': {'indent': 2, 'default_factory': None}}
    >>> sdict = from_dict({'a': {'b': 1}}, _StackedDict, **setup)
    >>> type(sdict)
    <class '_StackedDict'>
    >>> sdict['a']['b']
    1

    Notes
    -----
    - Already-instantiated _StackedDict values are preserved as-is
    - Regular dict values are recursively converted
    - Non-dict values are assigned directly
    - All created instances share the same class_options

    See Also
    --------
    _StackedDict.__init__ : Constructor that uses this function
    _StackedDict.to_dict : Inverse operation (convert back to dict)
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
    * Parent's references enable path reconstruction from any node

    The use of immutable tuples for children optimizes memory usage and iteration
    performance for tree traversal algorithms (DFS, BFS).

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code. Access it through ``CompactPathsView`` or ``NestedDictionary``.

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
    _CPaths : Uses _HKey internally for tree representation
    """

    __slots__ = ("key", "children", "parent", "is_root")

    def __init__(
        self, key: Any, parent: Optional["_HKey"] = None, is_root: bool = False
    ) -> None:
        self.key: Any = key
        self.children: Tuple[_HKey, ...] = ()
        self.parent: Optional[_HKey] = parent
        self.is_root: bool = is_root

    @classmethod
    def build_forest(cls, stacked_dict: Dict) -> "_HKey":
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

    def add_child(self, key: Any) -> "_HKey":
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

    def add_children(self, keys: List[Any]) -> Tuple["_HKey", ...]:
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

    def get_child(self, key: Any) -> Optional["_HKey"]:
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
        return not (self.is_root or self.has_children())

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

    def find_by_path(self, path: List[Any]) -> Optional["_HKey"]:
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

    def get_descendants(self) -> List["_HKey"]:
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

    def iter_children(self) -> Iterator["_HKey"]:
        """
        Iterate over direct child nodes.

        Yields
        ------
        _HKey
            Each child node
        """
        return iter(self.children)

    def iter_leaves(self) -> Iterator["_HKey"]:
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

    # def to_dict(self) -> Dict[Any, Any]:
    #    """
    #    Convert the tree structure back to a nested dict.

    #    Returns
    #    -------
    #    Dict[Any, Any]
    #        Nested dictionary representation of the tree
    #   """
    #    result: Dict[Any, Any] = {}

    #    for child in self.children:
    #        if child.has_children():
    #            result[child.key] = child.to_dict()
    #        else:
    #            result[child.key] = {}

    #    return result

    # ========================================================================
    # Tree Traversal Algorithms
    # ========================================================================

    def _dfs_traverse(
        self,
        visit: Optional[Callable[["_HKey"], None]] = None,
        *,
        preorder: bool = True,
    ) -> Iterator["_HKey"]:
        """
        Internal helper generator for DFS traversals with cycle protection.

        Parameters
        ----------
        visit : Optional[Callable[["_HKey"], None]]
            Optional callback invoked for each yielded node.
        preorder : bool
            If True, yield node before children (pre-order). If False, yield
            after children (post-order).
        """
        seen: Set[int] = set()

        def traverse(node: "_HKey") -> Iterator["_HKey"]:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)

            if preorder:
                if visit:
                    visit(node)
                yield node

            for child in node.children:
                yield from traverse(child)

            if not preorder:
                if visit:
                    visit(node)
                yield node

        yield from traverse(self)

    def dfs_preorder(
        self, visit: Optional[Callable[["_HKey"], None]] = None
    ) -> Iterator["_HKey"]:
        """
        Depth-First Search traversal in pre-order (node, then children).

        Pre-order: Visit current node before its children.
        Order: Root → Left subtree → Right subtree

        This traversal is cycle-safe: if the underlying structure contains
        cycles (which should not happen in a valid tree), nodes that have
        already been seen will be skipped to prevent infinite loops.

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
        yield from self._dfs_traverse(visit, preorder=True)

    def dfs_postorder(
        self, visit: Optional[Callable[["_HKey"], None]] = None
    ) -> Iterator["_HKey"]:
        """
        Depth-First Search traversal in post-order (children, then node).

        Post-order: Visit children before current node.
        Order: Left subtree → Right subtree → Root
        Useful for deletion or bottom-up calculations.

        This traversal is cycle-safe: previously visited nodes are skipped to
        prevent infinite recursion if a cycle is present.

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
        yield from self._dfs_traverse(visit, preorder=False)

    def bfs(
        self, visit: Optional[Callable[["_HKey"], None]] = None
    ) -> Iterator["_HKey"]:
        """
        Breadth-First Search (level-order) traversal.

        BFS explores all nodes at depth N before moving to depth N+1.
        Uses a queue (deque) for optimal O(1) operations.

        This traversal is cycle-safe: nodes already seen are not enqueued
        again, preventing infinite loops in the presence of cycles.

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
        seen: Set[int] = {id(self)}

        while queue:
            node = queue.popleft()
            if visit:
                visit(node)
            yield node

            for child in node.children:
                cid = id(child)
                if cid not in seen:
                    seen.add(cid)
                    queue.append(child)

    def dfs_find(self, predicate: Callable[["_HKey"], bool]) -> Optional["_HKey"]:
        """
        Find first node matching predicate using DFS.

        Performs depth-first search and returns the first node for which
        the predicate returns True. Returns None if no match found.
        This method is cycle-safe as it leverages `dfs_preorder` which
        guards against revisiting nodes.

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
        for node in self.dfs_preorder():
            if predicate(node):
                return node
        return None

    def bfs_find(self, predicate: Callable[["_HKey"], bool]) -> Optional["_HKey"]:
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

    def find_all(self, predicate: Callable[["_HKey"], bool]) -> List["_HKey"]:
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

    def find_by_key(
        self, key: Any, find_all: bool = False
    ) -> Optional["_HKey"] | List["_HKey"]:
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

    def iter_by_level(self) -> Iterator[Tuple[int, List["_HKey"]]]:
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

    def get_nodes_at_depth(self, target_depth: int) -> List["_HKey"]:
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
        return [
            node
            for node in self.bfs()
            if not node.is_root and node.get_depth() == target_depth
        ]

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

    def map_nodes(self, func: Callable[["_HKey"], Any]) -> List[Any]:
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

    def prune(self, predicate: Callable[["_HKey"], bool]) -> "_HKey":
        """
        Create a new tree containing only nodes matching the predicate and their ancestors.

        This method implements "pruning with path preservation", which filters the tree
        to keep only branches that lead to nodes satisfying the predicate. All ancestor
        nodes are automatically preserved to maintain tree connectivity and hierarchical
        structure.

        The algorithm:

        1. Identifies all nodes matching the predicate
        2. Preserves all ancestors of matching nodes to maintain paths
        3. Removes all other branches
        4. Returns a new tree (original is unchanged)

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that takes an _HKey node and returns True if the node should
            be kept in the pruned tree. The function is called on each node during
            traversal.

        Returns
        -------
        _HKey
            A new pruned tree (root node) containing only the filtered branches.
            The original tree remains unchanged. The returned tree preserves the
            same root properties (key, is_root flag).

        Examples
        --------
        >>> # Keep only nodes with key 'b' at depth 2
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}, 'x': {'b': {'z': 1}}}})
        >>> pruned = root.prune(lambda n: n.key == 'b' and n.get_depth() == 2)
        >>> pruned.get_all_paths()
        [['a'], ['a', 'b'], ['a', 'x'], ['a', 'x', 'b']]

        >>> # Keep only leaf nodes
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> pruned = root.prune(lambda n: n.is_leaf())
        >>> pruned.get_all_paths()
        [['a'], ['a', 'b'], ['a', 'c']]

        >>> # Keep nodes with specific key value
        >>> root = _HKey.build_forest({'a': {'target': 1, 'other': 2}, 'b': 3})
        >>> pruned = root.prune(lambda n: n.key == 'target')
        >>> pruned.get_all_paths()
        [['a'], ['a', 'target']]

        Notes
        -----
        - The pruned tree maintains hierarchical connectivity by preserving ancestor paths
        - This is a non-destructive operation; the original tree is not modified
        - If no nodes match the predicate, returns a tree with only the root node
        - The predicate is evaluated on every node in the tree (O(n) complexity)
        - All intermediate nodes required to reach matching nodes are preserved,
          even if they don't match the predicate themselves

        See Also
        --------
        filter_paths : Filter paths based on a predicate function
        find_all : Find all nodes matching a predicate
        dfs_find : Find first node matching a predicate using DFS
        get_all_paths : Get all paths in the tree
        """
        new_root = _HKey(self.key, is_root=self.is_root)

        def has_matching_descendant(node: _HKey) -> bool:
            """
            Check if node or any of its descendants matches the predicate.

            Parameters
            ----------
            node : _HKey
                Node to check

            Returns
            -------
            bool
                True if node or any descendant matches predicate
            """
            if predicate(node):
                return True
            for child in node.children:
                if has_matching_descendant(child):
                    return True
            return False

        def copy_matching_subtree(source: _HKey, target: _HKey) -> None:
            """
            Recursively copy nodes that match or have matching descendants.

            This function traverses the source tree and copies nodes to the target
            tree only if they match the predicate or have descendants that match.
            This preserves the hierarchical path to all matching nodes.

            Parameters
            ----------
            source : _HKey
                Source node to copy from
            target : _HKey
                Target node to copy to
            """
            for child in source.children:
                # Only process this child if it or its descendants match
                if has_matching_descendant(child):
                    new_child = target.add_child(child.key)
                    # Recursively copy the subtree
                    copy_matching_subtree(child, new_child)

        copy_matching_subtree(self, new_root)
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
        avg_branching = (
            sum(len(n.children) for n in non_leaf_nodes) / len(non_leaf_nodes)
            if non_leaf_nodes
            else 0
        )

        return {
            "total_nodes": len(all_nodes),
            "leaf_count": len(leaves),
            "max_depth": self.get_max_depth(),
            "avg_branching_factor": round(avg_branching, 2),
            "total_paths": len(self.get_all_paths()),
            "levels": (
                self.get_max_depth() + 1 if not self.is_root else self.get_max_depth()
            ),
        }

    # ========================================================================
    # Graph Theory & Structure Validation
    # ========================================================================

    def has_cycles(self) -> Tuple[bool, Optional[List["_HKey"]]]:
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
                    cycle_start = next(
                        i for i, n in enumerate(path) if id(n) == child_id
                    )
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
            issues.append(
                f"Cycle detected: {[n.key for n in cycle] if cycle else 'unknown'}"
            )

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
                issues.append(
                    f"Node {node.key} has no parent but is not marked as root"
                )

            # Verify parent's children contain this node
            if node.parent and not node.is_root:
                print("->")
                print(node, ":", node.parent, ":", node.parent.children)
                if node not in node.parent.children:
                    print("-->")
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
        internal_nodes = [
            n for n in self.dfs_preorder() if n.has_children() and not n.is_root
        ]
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
        internal_nodes = [
            node
            for node in self.dfs_preorder()
            if node.has_children() and not node.is_root
        ]

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

    def __getitem__(self, key: Any) -> "_HKey":
        """Get child by key (dict-like access)."""
        for child in self.children:
            if child.key == key:
                return child
        raise KeyError(key)

    def __iter__(self) -> Iterator["_HKey"]:
        """Iterate over children."""
        return iter(self.children)

    def __repr__(self) -> str:
        if self.is_root:
            return f"_HKey(ROOT, children={len(self.children)})"
        return f"_HKey(key={self.key!r}, children={len(self.children)})"


class _StackedDict(defaultdict):
    """
    Internal base class for hierarchical nested dictionary structures.

    ``_StackedDict`` is the **central engine** providing the foundation for all
    nested dictionary operations in the ndict_tools package. It extends Python's
    ``defaultdict`` with hierarchical key support, path-based access, and specialized
    traversal methods.

    Key features:

    * **Hierarchical keys**: Use lists as keys to access nested values: ``d[['a', 'b', 'c']]``
    * **Automatic nesting**: Missing intermediate levels are created automatically
    * **Path views**: Access all paths via ``paths()`` and ``compact_paths()``
    * **Tree traversal**: DFS and BFS algorithms for navigation
    * **Deep operations**: Specialized copy, equality, and conversion methods

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code. Access it through ``NestedDictionary`` or subclasses.

       However, it can be used directly by developers for custom implementations
       as described in the usage documentation.

    Parameters
    ----------
    *args : iterable, optional
        Dictionaries or iterables to initialize from
    **kwargs : dict, optional
        Either initialization data OR configuration via 'default_setup'

    Attributes
    ----------
    indent : int
        Indentation level for string representation (default: 2)
    default_factory : callable or None
        Factory function for missing keys (inherited from defaultdict)
    _default_setup : set
        Internal storage for configuration as set of (key, value) tuples

    Examples
    --------
    >>> setup = {'indent': 2, 'default_factory': None}
    >>> sd = _StackedDict(default_setup=setup)
    >>> sd['a']['b']['c'] = 1  # Automatic nesting
    >>> sd[['a', 'b', 'c']]
    1
    >>> # Initialize with data
    >>> sd = _StackedDict({'a': {'b': 1}}, default_setup=setup)
    >>> list(sd.paths())
    [['a'], ['a', 'b']]
    >>> # Hierarchical key access
    >>> sd[['a', 'b']] = 2
    >>> sd['a']['b']
    2

    Notes
    -----
    The class maintains two key invariants:

    1. All nested dictionaries are _StackedDict instances (or subclass)
    2. All instances share the same default_setup configuration

    See Also
    --------
    _Paths : View object for accessing all paths
    _CPaths : Compact representation of paths
    _HKey : Internal tree structure for path operations
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a new _StackedDict with configuration and optional data.

        The constructor requires configuration via the 'default_setup' parameter,
        which must contain at least 'indent' and 'default_factory' keys. Additional
        initialization data can be provided through args or kwargs.

        Parameters
        ----------
        *args : iterable
            Dictionaries, _StackedDict instances, or iterables of (key, value) pairs
        **kwargs : dict
            Either:
            - 'default_setup': dict with 'indent' and 'default_factory' keys
            - Direct key-value pairs to initialize (requires default_setup in kwargs)

        Raises
        ------
        StackedKeyError
            If 'indent' or 'default_factory' is missing from configuration
        StackedAttributeError
            If default_setup contains keys that aren't valid attributes

        Examples
        --------
        >>> setup = {'indent': 2, 'default_factory': None}
        >>> sd = _StackedDict(default_setup=setup)
        >>> # Initialize with data
        >>> sd = _StackedDict({'a': 1}, default_setup=setup)
        >>> # Copy another _StackedDict
        >>> sd2 = _StackedDict(sd, default_setup=setup)

        Notes
        -----
        - Args are processed sequentially, later values override earlier ones
        - _StackedDict args are deep-copied
        - Regular dicts are converted to _StackedDict recursively
        - All configuration is propagated to nested instances
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
            warnings.warn(
                "indent and default parameters are obsolete since version 0.8.0"
                "and will be remove in version 1.2.0. Use kwargs default_setup dictionary instead",
                DeprecationWarning,
                stacklevel=2,
            )
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
        """
        Get configuration as an ordered list of (key, value) tuples.

        Returns a deterministic view of the internal configuration with
        priority ordering: 'indent', 'default_factory', then alphabetically.

        Returns
        -------
        list of tuple
            Configuration as [(key, value), ...] in priority order

        Examples
        --------
        >>> sd = _StackedDict(default_setup={'indent': 2, 'default_factory': None})
        >>> sd.default_setup
        [('indent', 2), ('default_factory', None)]

        See Also
        --------
        default_setup.setter : Set new configuration
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
        """
        Update configuration from dict, list, or set of tuples.

        Parameters
        ----------
        value : dict, list of tuple, or set of tuple
            New configuration to apply

        Examples
        --------
        >>> sd = _StackedDict(default_setup={'indent': 2, 'default_factory': None})
        >>> sd.default_setup = {'indent': 4, 'default_factory': int}
        >>> sd.indent
        4
        """
        if isinstance(value, dict):
            items = value.items()
        else:
            items = value
        self._default_setup = set(items)

    def __str__(self, padding=0) -> str:
        """ "
        Convert to JSON-like formatted string representation.

        Creates a human-readable string with proper indentation showing
        the nested structure. Uses the configured indent level.

        Parameters
        ----------
        padding : int, optional
            Current indentation level (used internally for recursion)

        Returns
        -------
        str
            JSON-like formatted string

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> print(sd)
        {
          a : {
            b : 1,
          },
        }

        Notes
        -----
        - Uses recursive formatting for nested dictionaries
        - Trailing commas are included for consistency
        - Empty dictionaries shown as {}
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
        Create a shallow copy of the _StackedDict.

        Creates a new _StackedDict with the same keys and values, but values
        are not recursively copied. Nested _StackedDict instances are referenced,
        not duplicated.

        Returns
        -------
        _StackedDict
            Shallow copy with same configuration

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd2 = sd.__copy__()
        >>> sd2 is sd
        False
        >>> sd2['a'] is sd['a']  # Nested dicts are referenced
        True

        See Also
        --------
        __deepcopy__ : Create complete independent copy
        copy : Public method wrapper
        """

        new = self.__class__(default_setup=dict(self.default_setup))
        for key, value in self.items():
            new[key] = value
        return new

    def __deepcopy__(self) -> _StackedDict:
        """
        Create a deep copy of the _StackedDict.

        Creates a completely independent copy where all nested structures
        are recursively duplicated. Changes to the copy will not affect
        the original.

        Returns
        -------
        _StackedDict
            Complete independent copy

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd2 = sd.__deepcopy__()
        >>> sd2['a']['b'] = 2
        >>> sd['a']['b']
        1

        Notes
        -----
        - Uses to_dict() → from_dict() pipeline for copying
        - Preserves class type (works with subclasses)
        - All configuration is transferred to the copy

        See Also
        --------
        __copy__ : Shallow copy alternative
        deepcopy : Public method wrapper
        """

        return from_dict(
            self.to_dict(), self.__class__, default_setup=dict(self.default_setup)
        )

    def __setitem__(self, key, value) -> None:
        """
        Set item with support for hierarchical keys.

        Supports both flat keys and hierarchical paths (as lists).
        For hierarchical keys, automatically creates intermediate
        _StackedDict levels as needed.

        Parameters
        ----------
        key : Any or List[Any]
            Single key or list representing hierarchical path
        value : Any
            Value to assign

        Raises
        ------
        StackedTypeError
            If key list contains nested lists

        Examples
        --------
        >>> sd = _StackedDict(default_setup={'indent': 2, 'default_factory': None})
        >>> sd['a'] = 1  # Flat key
        >>> sd[['b', 'c', 'd']] = 2  # Hierarchical key
        >>> sd['b']['c']['d']
        2

        >>> # Nested lists not allowed
        >>> sd[['a', ['b']]] = 1  # Raises StackedTypeError

        Notes
        -----
        - Creates intermediate levels automatically
        - Overwrites existing values at the target path
        - All created levels use the same default_setup

        See Also
        --------
        __getitem__ : Get items with hierarchical keys
        __delitem__ : Delete items with hierarchical keys
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
        Get item with support for hierarchical keys.

        Supports both flat keys and hierarchical paths (as lists).
        For hierarchical keys, traverses the nested structure to
        retrieve the value at the specified path.

        Parameters
        ----------
        key : Any or List[Any]
            Single key or list representing hierarchical path

        Returns
        -------
        Any
            Value at the specified key/path

        Raises
        ------
        StackedTypeError
            If key list contains nested lists
        KeyError
            If key or path doesn't exist

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd['a']
        <_StackedDict: {'b': 1}>
        >>> sd[['a', 'b']]
        1

        Notes
        -----
        - Flat keys behave like standard dict access
        - List keys traverse the hierarchy
        - Raises KeyError if path doesn't exist

        See Also
        --------
        __setitem__ : Set items with hierarchical keys
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
        Delete item with support for hierarchical keys and cleanup.

        Deletes the item at the specified key or hierarchical path.
        For hierarchical keys, automatically removes empty parent
        dictionaries after deletion.

        Parameters
        ----------
        key : Any or List[Any]
            Single key or list representing hierarchical path

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> del sd[['a', 'b', 'c']]
        >>> 'b' in sd['a']  # Empty 'b' was removed
        False

        Notes
        -----
        - Automatically cleans up empty parent dictionaries
        - Preserves non-empty parent levels
        - Works with both flat and hierarchical keys

        See Also
        --------
        pop : Delete and return value
        popitem : Remove and return last item
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
        Check equality: same class, configuration, and content.

        Two _StackedDict instances are equal if they have:
        1. Identical class type (exact match, not subclasses)
        2. Identical default_setup configuration
        3. Identical dictionary structure and values

        Parameters
        ----------
        other : Any
            Object to compare with

        Returns
        -------
        bool
            True if all conditions met

        Examples
        --------
        >>> setup = {'indent': 2, 'default_factory': None}
        >>> sd1 = _StackedDict({'a': 1}, default_setup=setup)
        >>> sd2 = _StackedDict({'a': 1}, default_setup=setup)
        >>> sd1 == sd2
        True

        >>> # Different setup
        >>> sd3 = _StackedDict({'a': 1}, default_setup={'indent': 4, 'default_factory': None})
        >>> sd1 == sd3
        False

        See Also
        --------
        __ne__ : Inequality check
        similar : Compare content only (ignore class/setup)
        isomorph : Compare as plain dicts
        """

        if not isinstance(other, type(self)):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def __ne__(self, other):
        """
        Check inequality (negation of __eq__).

        Returns
        -------
        bool
            True if not equal

        See Also
        --------
        __eq__ : Equality check
        """

        return not self.__eq__(other)

    def similar(self, other):
        """
        Check if two structures share the same content (ignoring setup).

        Two structures are similar if they:
        1. Are both _StackedDict instances (any subclass)
        2. Have identical dictionary content (keys and values)

        Configuration differences are ignored.

        Parameters
        ----------
        other : Any
            Object to compare with

        Returns
        -------
        bool
            True if both are _StackedDict with same content

        Examples
        --------
        >>> setup1 = {'indent': 2, 'default_factory': None}
        >>> setup2 = {'indent': 4, 'default_factory': _StackedDict}
        >>> sd1 = _StackedDict({'a': 1}, default_setup=setup1)
        >>> sd2 = _StackedDict({'a': 1}, default_setup=setup2)
        >>> sd1 == sd2
        False
        >>> sd1.similar(sd2)
        True

        See Also
        --------
        __eq__ : Strict equality (includes setup)
        isomorph : Compare as plain dicts
        """
        if not isinstance(other, _StackedDict):
            return False

        return compare_dict(self.to_dict(), other.to_dict())

    def isomorph(self, other):
        """
        Check if structures are isomorphic (same keys/values, any dict type).

        Two structures are isomorphic if they represent the same nested
        dictionary structure, regardless of whether they're _StackedDict,
        plain dict, or any other dict-like type.

        Parameters
        ----------
        other : dict or _StackedDict
            Dictionary to compare with

        Returns
        -------
        bool
            True if structure-preserving mapping exists

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> regular_dict = {'a': {'b': 1}}
        >>> sd.isomorph(regular_dict)
        True

        >>> sd.isomorph({'a': {'b': 2}})
        False

        Notes
        -----
        Checks if sd[k1]...[kn] == other[k1]...[kn] for all paths.
        This is the most permissive comparison method.

        See Also
        --------
        __eq__ : Strict equality
        similar : Compare _StackedDict instances
        """

        if not isinstance(other, (dict, _StackedDict)):
            return False
        elif isinstance(other, _StackedDict):
            return compare_dict(self.to_dict(), other.to_dict())
        else:
            return compare_dict(self.to_dict(), dict(other))

    def unpacked_items(self) -> Generator:
        """
        Generate all (path, value) pairs from nested structure.

        Yields terminal values along with their hierarchical paths as tuples.
        This provides a flattened view of the entire nested dictionary.

        Yields
        ------
        tuple
            (path_tuple, value) for each terminal value

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.unpacked_items())
        [(('a', 'b'), 1), (('c',), 2)]

        >>> # Empty dict as value
        >>> sd = _StackedDict({'a': {}}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.unpacked_items())
        [(('a',), {})]

        Notes
        -----
        - Uses depth-first traversal
        - Empty dictionaries are treated as terminal values
        - Paths are immutable tuples for hashability

        See Also
        --------
        unpacked_keys : Get only the paths
        unpacked_values : Get only the values
        dfs : Depth-first traversal alternative returning lists
        """

        for key, value in unpack_items(self):
            yield key, value

    def unpacked_keys(self) -> Generator:
        """
        Generate all hierarchical paths (keys) from nested structure.

        Yields the path to each terminal value as a tuple of keys,
        providing access to all navigable paths in the dictionary.

        Yields
        ------
        tuple
            Path as tuple of keys

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.unpacked_keys())
        [('a', 'b')]

        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3}, default_setup={'indent': 2, 'default_factory': None})
        >>> sorted(sd.unpacked_keys())
        [('a', 'b'), ('a', 'c'), ('d',)]

        See Also
        --------
        unpacked_items : Get (path, value) pairs
        paths : Get paths as _Paths view object
        """

        for key, value in unpack_items(self):
            yield key

    def unpacked_values(self) -> Generator:
        """
        Generate all terminal values from nested structure.

        Yields only the leaf values, discarding path information.
        Useful for collecting all data values regardless of structure.

        Yields
        ------
        Any
            Each leaf value in the nested dictionary

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.unpacked_values())
        [1, 2]

        >>> # Empty dict is a value
        >>> sd = _StackedDict({'a': {}, 'b': 1}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.unpacked_values())
        [{}, 1]

        See Also
        --------
        unpacked_items : Get (path, value) pairs
        leaves : Alternative method returning list
        """
        for key, value in unpack_items(self):
            yield value

    def to_dict(self) -> dict:
        """
        Convert to a standard nested dictionary.

        Recursively converts the _StackedDict and all nested _StackedDict
        instances to regular Python dictionaries, removing all special
        functionality but preserving the structure.

        Returns
        -------
        dict
            Regular nested dictionary with same structure

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> regular = sd.to_dict()
        >>> type(regular)
        <class 'dict'>
        >>> regular
        {'a': {'b': 1}}

        Notes
        -----
        - All _StackedDict instances are converted recursively
        - Other value types are preserved as-is
        - Inverse operation of from_dict()

        See Also
        --------
        from_dict : Convert dict to _StackedDict
        __deepcopy__ : Create _StackedDict copy
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
        Create a shallow copy of the _StackedDict.

        Returns
        -------
        _StackedDict
            Shallow copy with same configuration

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd2 = sd.copy()
        >>> sd2['a'] is sd['a']
        True

        See Also
        --------
        __copy__ : Internal implementation
        deepcopy : Create independent copy
        """

        return self.__copy__()

    def deepcopy(self) -> _StackedDict:
        """
        Create a deep copy of the _StackedDict.

        Creates a completely independent copy where all nested structures
        are recursively duplicated.

        Returns
        -------
        _StackedDict
            Complete independent copy

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd2 = sd.deepcopy()
        >>> sd2['a']['b'] = 999
        >>> sd['a']['b']
        1

        See Also
        --------
        __deepcopy__ : Internal implementation
        copy : Shallow copy alternative
        """

        return self.__deepcopy__()

    def pop(self, key: Union[Any, List[Any]], default=None) -> Any:
        """
        Remove and return value at key or hierarchical path.

        Removes the specified key (flat or hierarchical) and returns its value.
        Automatically cleans up empty parent dictionaries after removal.
        If the key doesn't exist, returns the default value or raises an error.

        Parameters
        ----------
        key : Any or List[Any]
            Single key or hierarchical path to remove
        default : Any, optional
            Value to return if key doesn't exist

        Returns
        -------
        Any
            The value that was removed

        Raises
        ------
        StackedKeyError
            If key doesn't exist and no default provided

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.pop('c')
        2
        >>> 'c' in sd
        False

        >>> # Hierarchical key
        >>> sd.pop(['a', 'b'])
        1
        >>> 'a' in sd  # Empty 'a' was removed
        False

        >>> # With default
        >>> sd.pop('nonexistent', 'default_value')
        'default_value'

        See Also
        --------
        popitem : Remove and return last item
        __delitem__ : Delete without returning value
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
        Remove and return the last item as (path, value) pair.

        Removes the last item in the most deeply nested dictionary,
        returning its full hierarchical path and value. Uses depth-first
        traversal to locate the deepest rightmost item.

        Returns
        -------
        tuple
            (path_list, value) where path_list is the hierarchical path

        Raises
        ------
        StackedIndexError
            If the dictionary is empty

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> path, value = sd.popitem()
        >>> path
        ['a', 'c']
        >>> value
        2

        >>> # Empty dictionary
        >>> sd = _StackedDict(default_setup={'indent': 2, 'default_factory': None})
        >>> sd.popitem()  # Raises StackedIndexError

        Notes
        -----
        - Follows DFS to find the last (rightmost, deepest) item
        - Cleans up empty parent dictionaries automatically
        - Path is returned as a list of keys

        See Also
        --------
        pop : Remove item by key
        unpacked_items : View all (path, value) pairs
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
        Update _StackedDict with key/value pairs from dict or kwargs.

        Merges the provided dictionary or keyword arguments into this
        _StackedDict, converting regular dicts to _StackedDict instances
        recursively while preserving existing _StackedDict values.

        Parameters
        ----------
        dictionary : dict, optional
            Dictionary with key/value pairs to merge
        **kwargs : dict
            Additional key/value pairs to merge

        Examples
        --------
        >>> sd = _StackedDict({'a': 1}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.update({'b': 2, 'c': {'d': 3}})
        >>> sd['c']['d']
        3

        >>> # Using kwargs
        >>> sd.update(e=4, f={'g': 5})
        >>> sd['f']['g']
        5

        Notes
        -----
        - Regular dicts are converted to _StackedDict recursively
        - _StackedDict values are accepted directly
        - Configuration is synchronized across all nested instances
        - Later values override earlier ones for duplicate keys

        See Also
        --------
        __init__ : Initialization with data
        __setitem__ : Set individual items
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
        Check if an atomic key exists at any level in the hierarchy.

        Searches through all hierarchical paths to determine if the given
        atomic key appears anywhere in the nested structure. Does not accept
        hierarchical paths (lists).

        Parameters
        ----------
        key : Any
            Atomic key to search for (not a list)

        Returns
        -------
        bool
            True if key exists at any level

        Raises
        ------
        StackedKeyError
            If key is a list (hierarchical keys not allowed)

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.is_key('b')
        True
        >>> sd.is_key('c')
        False

        >>> # Lists not allowed
        >>> sd.is_key(['a', 'b'])  # Raises StackedKeyError

        Notes
        -----
        - Only searches for atomic keys
        - Checks all nesting levels
        - O(n) complexity where n is total number of keys

        See Also
        --------
        occurrences : Count how many times key appears
        key_list : Get all paths containing the key
        """

        # Normalize the key (convert lists to tuples for uniform comparison)
        if isinstance(key, list):
            raise StackedKeyError("This function manages only atomic keys", key=key)

        # Check directly if the key exists in unpacked keys
        return any(key in keys for keys in self.unpacked_keys())

    def occurrences(self, key: Any) -> int:
        """
        Count occurrences of an atomic key throughout the hierarchy.

        Returns the total number of times a key appears in the nested
        structure, counting each occurrence in every hierarchical path.

        Parameters
        ----------
        key : Any
            Atomic key to count

        Returns
        -------
        int
            Number of occurrences (0 if key doesn't exist)

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': {'b': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.occurrences('b')
        2
        >>> sd.occurrences('a')
        1
        >>> sd.occurrences('z')
        0

        >>> # Key appearing multiple times in same path
        >>> sd = _StackedDict({'a': {'a': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.occurrences('a')
        2

        See Also
        --------
        is_key : Check if key exists
        key_list : Get all paths containing the key
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
        Get all hierarchical paths containing a specific key.

        Returns a list of all complete paths (as tuples) that contain
        the specified atomic key at any position in the path.

        Parameters
        ----------
        key : Any
            Atomic key to search for

        Returns
        -------
        list
            List of paths (as tuples) containing the key

        Raises
        ------
        StackedKeyError
            If key doesn't exist in the dictionary

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': {'b': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.key_list('b')
        [('a', 'b'), ('c', 'b')]

        >>> sd.key_list('a')
        [('a', 'b')]

        >>> sd.key_list('nonexistent')  # Raises StackedKeyError

        See Also
        --------
        is_key : Check if key exists
        items_list : Get values at paths containing key
        occurrences : Count occurrences
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
        Get all values associated with paths containing a specific key.

        Returns a list of all terminal values whose hierarchical paths
        contain the specified atomic key at any position.

        Parameters
        ----------
        key : Any
            Atomic key to search for

        Returns
        -------
        list
            List of values from paths containing the key

        Raises
        ------
        StackedKeyError
            If key doesn't exist in the dictionary

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': {'b': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.items_list('b')
        [1, 2]

        >>> sd.items_list('a')
        [1]

        >>> sd.items_list('nonexistent')  # Raises StackedKeyError

        Notes
        -----
        - Returns values in the order they're encountered during traversal
        - Multiple values returned if key appears in multiple paths
        - Only returns terminal (leaf) values

        See Also
        --------
        key_list : Get paths containing the key
        unpacked_items : Get all (path, value) pairs
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
        Get a view object for all hierarchical paths.

        .. deprecated:: 0.9
            This method is deprecated and will be removed in a future release.
            Use :meth:`paths` instead.

        Returns
        -------
        _Paths
            View object providing access to all paths

        See Also
        --------
        paths : Recommended replacement method
        """

        return _Paths(self)

    def paths(self) -> _Paths:
        """
        Get a view object for all hierarchical paths in the dictionary.

        Returns a lazy view similar to dict.keys() that provides access to
        all hierarchical paths without materializing them in memory. The view
        supports iteration, length queries, and membership testing.

        Returns
        -------
        _Paths
            Lazy view object for all hierarchical paths

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> paths = sd.paths()
        >>> len(paths)
        3
        >>> ['a', 'b'] in paths
        True
        >>> list(paths)
        [['a'], ['a', 'b'], ['c']]

        Notes
        -----
        - Paths are generated lazily during iteration
        - Internal _HKey tree is built on first access
        - View updates automatically if dictionary changes
        - More efficient than unpacked_keys() for large structures

        See Also
        --------
        compact_paths : Get factorized/compact representation
        unpacked_keys : Generate paths as tuples
        _Paths : View class documentation
        """

        return _Paths(self)

    def compact_paths(self) -> _CPaths:
        """
        Get a compact/factorized representation of all paths.

        Returns a view that represents the hierarchical structure in a
        factorized form where common prefixes are shared. This provides
        a more concise representation useful for visualization and analysis.

        Returns
        -------
        _CPaths
            Compact view with factorized path structure

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> cpaths = sd.compact_paths()
        >>> cpaths.structure
        [['a', 'b', 'c']]

        >>> # Expand back to full paths
        >>> cpaths.expand()
        [['a'], ['a', 'b'], ['a', 'c']]

        Notes
        -----
        - Provides bijective mapping between expanded and compact forms
        - Useful for path coverage analysis
        - More compact representation for deeply nested structures

        See Also
        --------
        paths : Get full paths view
        _CPaths : Compact view class documentation
        """

        return _CPaths(self)

    def dfs(self, node=None, path=None) -> Generator[Tuple[List, Any], None, None]:
        """
        Depth-First Search traversal of the nested dictionary.

        Recursively traverses the dictionary in depth-first order, yielding
        each hierarchical path (as list) and its corresponding value. This
        includes both intermediate nodes and leaf values.

        Parameters
        ----------
        node : dict, optional
            Current node being traversed (defaults to root/self)
        path : list, optional
            Current path being constructed (defaults to empty list)

        Yields
        ------
        tuple
            (path_list, value) for each node in DFS order

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> for path, value in sd.dfs():
        ...     print(f"{path} -> {value}")
        ['a'] -> <_StackedDict>
        ['a', 'b'] -> 1
        ['c'] -> 2

        Notes
        -----
        - Visits nodes before their children (pre-order)
        - Returns paths as mutable lists (unlike unpacked_keys)
        - Includes intermediate dictionary nodes
        - Useful for tree-based operations

        See Also
        --------
        bfs : Breadth-first traversal
        unpacked_items : Similar but only terminal values
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
        Breadth-First Search traversal of the nested dictionary.

        Iteratively traverses the dictionary level by level, visiting all
        nodes at depth N before moving to depth N+1. Uses a queue (deque)
        for efficient FIFO operations.

        Yields
        ------
        tuple
            (path_tuple, value) for each node in BFS order

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> for path, value in sd.bfs():
        ...     print(f"{path} -> {value}")
        ('a',) -> <_StackedDict>
        ('a', 'b') -> <_StackedDict>
        ('a', 'b', 'c') -> 1

        >>> # Only leaf values
        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3}, default_setup={'indent': 2, 'default_factory': None})
        >>> leaves = [(p, v) for p, v in sd.bfs() if not isinstance(v, _StackedDict)]
        >>> leaves
        [(('a', 'b'), 1), ('a', 'c'), 2), (('d',), 3)]

        Notes
        -----
        - Visits all nodes at same depth before going deeper
        - Returns paths as immutable tuples
        - Includes all nodes (intermediate and terminal)
        - More memory efficient than collecting all paths first

        See Also
        --------
        dfs : Depth-first traversal
        _HKey.bfs : Tree-based BFS traversal
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
        Compute the height (maximum depth) of the nested structure.

        The height is defined as the length of the longest path from root
        to any leaf node. An empty dictionary has height 0.

        Returns
        -------
        int
            Maximum path length in the dictionary

        Examples
        --------
        >>> sd = _StackedDict({'a': 1}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.height()
        1

        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.height()
        3

        >>> sd = _StackedDict(default_setup={'indent': 2, 'default_factory': None})
        >>> sd.height()
        0

        Notes
        -----
        - Empty dictionary has height 0
        - Single-level dictionary has height 1
        - O(n) complexity where n is number of paths

        See Also
        --------
        size : Count total number of keys
        leaves : Get all leaf values
        """

        return max((len(path) for path in self.dict_paths()), default=0)

    def size(self) -> int:
        """
        Compute the total number of keys (nodes) in the structure.

        Counts all keys at all nesting levels, including both intermediate
        dictionary keys and terminal value keys.

        Returns
        -------
        int
            Total number of keys across all levels

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.size()
        2

        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.size()
        4

        Notes
        -----
        - Counts all keys at all levels
        - Equivalent to number of nodes in tree representation
        - Different from len() which only counts top-level keys

        See Also
        --------
        height : Get maximum depth
        __len__ : Get top-level key count
        """

        return sum(1 for _ in self.unpacked_items())

    def leaves(self) -> list:
        """
        Extract all leaf (terminal) values from the nested structure.

        Returns a list of all values that are not themselves nested
        dictionaries, i.e., all terminal nodes in the tree structure.

        Returns
        -------
        list
            List of all leaf values

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.leaves()
        [1, 2]

        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.leaves()
        [1]

        >>> # Empty dict as leaf value
        >>> sd = _StackedDict({'a': {}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.leaves()
        [{}]

        Notes
        -----
        - Returns values in DFS order
        - Empty dictionaries are considered leaf values
        - Plain dicts (not _StackedDict) are also leaves

        See Also
        --------
        unpacked_values : Generator alternative
        dfs : Traversal including intermediate nodes
        """

        return [value for _, value in self.dfs() if not isinstance(value, _StackedDict)]

    def is_balanced(self) -> bool:
        """
        Check if the nested structure is height-balanced.

        A balanced dictionary is one where the height difference between
        any two subtrees at the same level differs by at most 1. This
        indicates a relatively uniform distribution of nesting depth.

        Returns
        -------
        bool
            True if structure is balanced, False otherwise

        Examples
        --------
        >>> # Balanced
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': {'d': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.is_balanced()
        True

        >>> # Unbalanced
        >>> sd = _StackedDict({'a': {'b': {'c': 1}}, 'd': 2}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.is_balanced()
        False

        Notes
        -----
        - Uses recursive height calculation
        - Checks balance at every node
        - Empty dictionary is considered balanced
        - Useful for identifying skewed structures

        See Also
        --------
        height : Get maximum depth
        _HKey.is_balanced : Tree-based balance check
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
        Find the hierarchical path (ancestors) leading to a specific value.

        Searches the nested structure for the given value and returns
        the complete path of keys leading to it, excluding the final key.
        This returns the "ancestry" of the value in the tree.

        Parameters
        ----------
        value : Any
            Value to search for in the nested dictionary

        Returns
        -------
        list
            List of keys forming the path to the value (excluding final key)

        Raises
        ------
        StackedValueError
            If value is not found in the dictionary

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.ancestors(1)
        ['a', 'b']

        >>> sd = _StackedDict({'a': {'b': 1}, 'c': {'d': 2}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.ancestors(2)
        ['c']

        >>> sd.ancestors(999)  # Raises StackedValueError

        Notes
        -----
        - Returns path excluding the final key (direct parent of value)
        - Uses DFS to find the first occurrence
        - If value appears multiple times, returns first found path
        - Empty list returned for top-level values

        See Also
        --------
        dfs : Depth-first traversal
        unpacked_items : Get all (path, value) pairs
        """

        for path, val in self.dfs():
            if val == value:
                return path[
                    :-1
                ]  # Return all keys except the last one (the direct key of the value)

        raise StackedValueError(
            f"Value {value} not found in the dictionary.", value=value
        )


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
    _CPaths : Factorized representation of paths
    _StackedDict.paths : building paths for nested dictionaries.
    """

    def __init__(self, stacked_dict: _StackedDict = None):
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
        if self._stacked_dict is not None and self._hkey is None:
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

    def to_compact(self) -> _CPaths:
        """
        Convert this _Paths to a _CPaths.

        Returns
        -------
        _CPaths
            Compact representation with the same paths
        """
        return _CPaths(self._stacked_dict)


class _CPaths(_Paths):
    """
    A lazy view providing compact representation of hierarchical paths.

    Extends _Paths to provide a factorized/compact representation where
    the hierarchical structure is represented as nested lists:
    - Leaf nodes: just the key
    - Internal nodes: [key, child1, child2, ...]

    The compact structure uses a bijective mapping:
    - Paths → Compact structure (factorization)
    - Compact structure → Paths (expansion)

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code.

    Parameters
    ----------
    stacked_dict : _StackedDict
        The nested dictionary to create a compact view for

    Attributes
    ----------
    _structure : Optional[List[Any]]
        Compact representation of paths (lazy-built)

    Examples
    --------
    >>> data = _StackedDict({'a': 1, 'b': {'c': 2, 'd': 3}})
    >>> c_paths = _CPaths(data)
    >>> c_paths.structure
    [['a'], ['b', 'c', 'd']]
    >>> list(c_paths)  # Inherited from _Paths
    [['a'], ['b'], ['b', 'c'], ['b', 'd']]

    See Also
    --------
    _Paths : Base class for path views
    """

    def __init__(self, stacked_dict: _StackedDict = None):
        super().__init__(stacked_dict)
        self._structure: Optional[List[Any]] = None

    def _ensure_structure(self) -> List[Any]:
        """
        Ensure compact structure is built (lazy initialization).

        Returns
        -------
        List[Any]
            The compact structure
        """
        if self._stacked_dict is not None and self._structure is None:
            self._structure = self._build_compact_structure()
        return self._structure

    @staticmethod
    def _validate_structure(structure: List[Any]) -> None:
        """
        Validate the compact structure format.

        Parameters
        ----------
        structure : List[Any]
            Structure to validate

        Raises
        ------
        ValueError
            If structure format is invalid

        Notes
        -----
        A valid structure is a list where each element is either:

          - A leaf value (any hashable type)
          - A list [key, child1, child2, ...] where key is the node and
            children are recursively valid structures
        """
        if not isinstance(structure, list):
            raise ValueError(
                f"Structure must be a list, got {type(structure).__name__}"
            )

        def validate_node(node, depth=0):
            if depth > MAX_DEPTH:  # Prevent infinite recursion
                raise ValueError(
                    f"Structure too deeply nested (max depth: {MAX_DEPTH})"
                )

            if isinstance(node, list):
                if len(node) == 0:
                    raise ValueError("Empty list not allowed in structure")
                # First element is the key, rest are children
                # Recursively validate children
                for child in node[1:]:
                    validate_node(child, depth + 1)
            # Leaf nodes can be any value (will be used as keys)

        for branch in structure:
            validate_node(branch)

    @property
    def structure(self) -> List[Any]:
        """
        Get the compact structure representation.

        Returns
        -------
        List[Any]
            Compact representation as nested lists

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict({'a': {'b': 1, 'c': 2}}))
        >>> c_paths.structure
        [['a', 'b', 'c']]
        """
        return self._ensure_structure()

    @structure.setter
    def structure(
        self, value: Union[_StackedDict, _HKey, List[Any], Dict[str, Any]]
    ) -> None:
        """
        Set or build the compact structure representation.

        Accepts the following input types:
        - _StackedDict (or dict): the source nested mapping to analyze
        - _HKey: an already-built hierarchical key tree
        - List[Any]: a compact structure as nested lists

        Parameters
        ----------
        value : Union[_StackedDict, _HKey, List[Any]]
            Input used to define the structure.

        Raises
        ------
        TypeError
            If value type is unsupported or structure type is invalid
        ValueError
            If provided compact structure format is invalid

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict())
        >>> # From compact structure (manual)
        >>> c_paths.structure = [['a'], ['d']]
        >>> c_paths.expand()
        [['a'], ['d']]

        >>> # From a stacked dict
        >>> c_paths.structure = _StackedDict({'a': {'b': 1}, 'd': 2})
        >>> c_paths.expand()
        [['a'], ['a', 'b'], ['d']]

        >>> # From an _HKey
        >>> hk = _HKey.build_forest({'x': {'y': {'z': 1}}})
        >>> c_paths.structure = hk
        >>> c_paths.expand()
        [['x'], ['x', 'y'], ['x', 'y', 'z']]
        """
        # Case 1: _StackedDict or dict
        if isinstance(value, _StackedDict) or isinstance(value, dict):
            # Normalize to _StackedDict
            self._stacked_dict = (
                value if isinstance(value, _StackedDict) else _StackedDict(value)
            )
            # Invalidate and rebuild from stacked dict
            self._hkey = None
            self._structure = self._build_compact_structure()
            return

        # Case 2: _HKey
        if isinstance(value, _HKey):
            # Replace internal tree and build compact structure from it
            self._hkey = value
            # Reuse existing builder which will read from self._hkey
            self._structure = self._build_compact_structure()
            return

        # Case 3: compact structure provided as list
        if isinstance(value, list):
            # Validate structure format
            self._validate_structure(value)
            self._structure = value
            # Do not alter existing _hkey/_stacked_dict here; they will be rebuilt lazily if used
            return

        raise TypeError(
            f"Unsupported type for structure: {type(value).__name__}. Expected _StackedDict, _HKey or list."
        )

    def _build_compact_structure(self) -> List[Any]:
        """
        Build the compact structure recursively from _hkey.

        The algorithm traverses the hierarchical key tree (_hkey):
        - If node is a leaf: return key
        - If node has children: return [key, compact_child1, compact_child2, ...]

        Returns
        -------
        List[Any]
            Compact representation as nested lists

        Notes
        -----
        Uses the _hkey attribute which maintains the hierarchical structure
        of all keys. This is the core factorization algorithm.
        """

        def compact_node(node) -> Any:
            """
            Recursively compact a node from _hkey tree.

            Parameters
            ----------
            node : _HKey node
                Node from the _hkey hierarchical structure

            Returns
            -------
            Any
                - key alone if leaf (no children)
                - [key, child1, child2, ...] if internal node (has children)
            """
            if node.is_leaf():
                # Leaf: return just the key
                return node.key

            # Internal node: [key, compact(child1), compact(child2), ...]
            children_compact = [compact_node(child) for child in node.iter_children()]
            return [node.key] + children_compact

        # Access _hkey from the _stacked_dict via inherited _ensure_hkey()
        hkey = self._ensure_hkey()

        if not hkey.has_children():
            return []

        # Compact each root-level child in _hkey
        return [compact_node(child) for child in hkey.iter_children()]

    @staticmethod
    def expand_structure(structure: List[Any]) -> List[List[Any]]:
        """
        Expand compact structure back to full paths.

        This is the inverse operation of compactification, establishing
        the bijection between compact and expanded representations.

        Parameters
        ----------
        structure : List[Any]
            Compact structure to expand

        Returns
        -------
        List[List[Any]]
            All expanded paths

        Examples
        --------
        >>> structure = [['a'], ['b', 'c', 'd']]
        >>> _CPaths.expand_structure(structure)
        [['a'], ['b'], ['b', 'c'], ['b', 'd']]

        >>> structure = [['x', ['y', 'z1', 'z2'], 'a']]
        >>> _CPaths.expand_structure(structure)
        [['x'], ['x', 'y'], ['x', 'y', 'z1'], ['x', 'y', 'z2'], ['x', 'a']]
        """
        all_paths = []

        def expand_node(node: Any, prefix: List[Any] = None) -> None:
            """
            Recursively expand a node.

            Parameters
            ----------
            node : Any
                Either a key (leaf) or [key, children...] (internal node)
            prefix : List[Any], optional
                Current path prefix
            """
            if prefix is None:
                prefix = []

            if not isinstance(node, list):
                # Leaf node: just a key
                current_path = prefix + [node]
                all_paths.append(current_path)
            else:
                # Internal node: [key, child1, child2, ...]
                key = node[0]
                current_path = prefix + [key]
                all_paths.append(current_path)

                # Recursively expand each child
                for child in node[1:]:
                    expand_node(child, current_path)

        # Expand each root branch
        for branch in structure:
            expand_node(branch)

        return all_paths

    def expand(self) -> List[List[Any]]:
        """
        Expand this instance's compact structure to full paths.

        Returns
        -------
        List[List[Any]]
            All expanded paths

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict({'a': {'b': 1}}))
        >>> c_paths.expand()
        [['a'], ['a', 'b']]

        Notes
        -----
        This is equivalent to calling expand_structure(self.structure),
        and should give the same result as list(self) from the parent class.
        """
        return self.expand_structure(self.structure)

    def __repr__(self) -> str:
        """
        Return technical representation with compact structure.

        Returns
        -------
        str
            String showing class name and compact structure

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict({'a': 1}))
        >>> repr(c_paths)
        "_CPaths([['a']])"
        """
        return f"{self.__class__.__name__}({self.structure})"

    def __str__(self) -> str:
        """
        Return readable string representation.

        Returns
        -------
        str
            Human-readable description

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict({'a': {'b': 1}, 'c': 2}))
        >>> str(c_paths)
        "_CPaths(3 paths): [['a', 'b'], ['c']]"
        """
        return f"_CPaths({len(self)} paths): {self.structure}"

    # ========================================================================
    # COVERAGE ANALYSIS METHODS
    # ========================================================================

    @staticmethod
    def _compare_path_sets(paths1: List[List[Any]], paths2: List[List[Any]]) -> tuple:
        """
        Compare two sets of paths and return statistics (private helper).

        Parameters
        ----------
        paths1 : List[List[Any]]
            First set of paths
        paths2 : List[List[Any]]
            Second set of paths

        Returns
        -------
        tuple
            (set1, set2, intersection, only_in_1, only_in_2)
        """
        set1 = {tuple(p) for p in paths1}
        set2 = {tuple(p) for p in paths2}
        intersection = set1 & set2
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        return set1, set2, intersection, only_in_1, only_in_2

    def is_covering(self, stacked_dict) -> bool:
        """
        Check if this _CPaths covers all paths in the given _StackedDict.

        A _CPaths is "covering" if its expanded paths exactly match all paths
        in the target _StackedDict.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        bool
            True if all paths in stacked_dict are present in this _CPaths

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1}, 'c': 2})
        >>> cpaths = _CPaths(sdict)
        >>> cpaths.is_covering(sdict)
        True

        >>> # Partial coverage
        >>> cpaths.structure = [['a']]  # Only covers 'a', not 'a.b' or 'c'
        >>> cpaths.is_covering(sdict)
        False

        Notes
        -----
        For a _CPaths created directly from a _StackedDict:
        _CPaths(sdict).is_covering(sdict) will ALWAYS return True
        because the compact structure is built from all paths in sdict.
        """
        target_paths = list(_Paths(stacked_dict))
        expanded_paths = self.expand()
        set1, set2, _, _, _ = self._compare_path_sets(expanded_paths, target_paths)
        return set1 == set2

    def coverage(self, stacked_dict) -> float:
        """
        Calculate the coverage percentage of this _CPaths over a _StackedDict.

        Coverage is defined as the ratio of paths in this _CPaths that exist
        in the target _StackedDict, divided by the total number of paths in
        the _StackedDict.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        float
            Coverage percentage between 0.0 and 1.0 (or > 1.0 if _CPaths
            contains paths not in stacked_dict)

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> cpaths = _CPaths(sdict)
        >>> cpaths.coverage(sdict)
        1.0

        >>> # Partial coverage: only 'a' and 'a.b' out of 4 paths
        >>> cpaths.structure = [['a', 'b']]
        >>> cpaths.coverage(sdict)
        0.5

        >>> # Over-coverage: includes paths not in sdict
        >>> cpaths.structure = [['a', 'b', 'c'], ['d'], ['e']]
        >>> cpaths.coverage(sdict)
        1.25

        Notes
        -----
        For a _CPaths created directly from a _StackedDict:
        _CPaths(sdict).coverage(sdict) will ALWAYS return 1.0
        because all paths from sdict are included.

        The coverage can be > 1.0 if the _CPaths contains more paths than
        the _StackedDict (e.g., manually set structure).
        """
        target_paths = list(_Paths(stacked_dict))
        expanded_paths = self.expand()

        _, set2, intersection, _, _ = self._compare_path_sets(
            expanded_paths, target_paths
        )

        if len(set2) == 0:
            return 0.0 if len(expanded_paths) > 0 else 1.0

        return len(intersection) / len(set2)

    def missing_paths(self, stacked_dict) -> List[List[Any]]:
        """
        Get paths from this _CPaths that are NOT in the _StackedDict.

        Returns the list of paths that exist in this _CPaths's expanded form
        but do not exist in the target _StackedDict. Useful for identifying
        extra or invalid paths.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        List[List[Any]]
            List of paths in _CPaths but not in stacked_dict

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1}})
        >>> cpaths = _CPaths(sdict)
        >>> cpaths.missing_paths(sdict)
        []

        >>> # Add extra paths
        >>> cpaths.structure = [['a', 'b', 'c'], ['d']]
        >>> cpaths.missing_paths(sdict)
        [['a', 'c'], ['d']]

        Notes
        -----
        For a _CPaths created directly from a _StackedDict:
        _CPaths(sdict).missing_paths(sdict) will ALWAYS return []
        because all paths are derived from sdict.

        See Also
        --------
        uncovered_paths : Get paths in _StackedDict not covered by _CPaths
        """
        target_paths = list(_Paths(stacked_dict))
        expanded_paths = self.expand()

        _, _, _, only_in_1, _ = self._compare_path_sets(expanded_paths, target_paths)

        # Preserve the original order from expanded_paths to avoid type comparison issues
        # when sorting heterogeneous path elements and to reflect user-provided order.
        extra_set = set(only_in_1)
        return [list(p) for p in expanded_paths if tuple(p) in extra_set]

    def uncovered_paths(self, stacked_dict) -> List[List[Any]]:
        """
        Get paths from _StackedDict that are NOT covered by this _CPaths.

        Returns the list of paths that exist in the target _StackedDict but
        are not present in this _CPaths's expanded form. Useful for identifying
        gaps in coverage.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        List[List[Any]]
            List of paths in stacked_dict but not in _CPaths

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> cpaths = _CPaths(sdict)
        >>> cpaths.uncovered_paths(sdict)
        []

        >>> # Partial structure
        >>> cpaths.structure = [['a', 'b']]
        >>> cpaths.uncovered_paths(sdict)
        [['a', 'c'], ['d']]

        Notes
        -----
        For a _CPaths created directly from a _StackedDict:
        _CPaths(sdict).uncovered_paths(sdict) will ALWAYS return []
        because all paths from sdict are included.

        See Also
        --------
        missing_paths : Get paths in _CPaths not in _StackedDict
        coverage : Get coverage ratio
        """
        target_paths = list(_Paths(stacked_dict))
        expanded_paths = self.expand()

        _, _, _, _, only_in_2 = self._compare_path_sets(expanded_paths, target_paths)

        # Preserve the original order from target_paths to avoid type comparison issues
        # when sorting heterogeneous path elements (e.g., str, int, tuple, frozenset).
        # Returning in traversal order is also more meaningful for users.
        missing_set = set(only_in_2)
        return [list(p) for p in target_paths if tuple(p) in missing_set]
