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

import copy
import json
import warnings
from collections import defaultdict, deque
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
)
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any, Self, TypeAlias, TypeVar, cast

from ._compat import override
from .exception import (
    StackedAttributeError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

MAX_DEPTH = 100

T = TypeVar("T", bound="_StackedDict")

_SetupSource: TypeAlias = Mapping[str, Any] | Iterable[tuple[str, Any]]
"Accepted by the default_setup setter: a mapping or (key, value) pairs."

# Marks a missing ``default`` argument, so that ``None`` stays a valid default.
_MISSING: Any = object()


def _reconstruct(
    cls: type, dictionary: dict[Any, Any], default_setup: dict[str, Any]
) -> "_StackedDict":
    """
    Module-level reconstruction helper for pickle.

    ``__reduce__`` must reference a module-level callable so that pickle can
    locate it by name during unpickling. A method reference (``cls.from_dict``)
    would not survive the pickle round-trip reliably across interpreter sessions.

    Parameters
    ----------
    cls : type
        The ``_StackedDict`` subclass to reconstruct.
    dictionary : dict
        Plain ``dict`` produced by ``to_dict()``.
    default_setup : dict
        Configuration dict (``indent``, ``default_factory``, …).

    Returns
    -------
    _StackedDict
        Reconstructed instance identical to the original.
    """
    return cls.from_dict(dictionary, default_setup=default_setup)


"""Internal functions"""


def compare_dict(d1: Any, d2: Any) -> bool:
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
    if type(d1) is not type(d2):
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


def unpack_items(
    dictionary: dict[Any, Any],
) -> Generator[tuple[tuple[Any, ...], Any], None, None]:
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


def from_dict(
    dictionary: dict[Any, Any], class_name: type[T], **class_options: Any
) -> T:
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
    class_name : type[T]
        The _StackedDict class (or subclass) to instantiate.
        Must be a subclass of _StackedDict.
    **class_options : dict
        Initialization options for the class instances. ``default_setup`` is
        optional and resolved by ``class_name._normalize_setup``.

    Returns
    -------
    T
        New instance of the specified class_name containing the dictionary structure.
        The return type matches the class type passed as class_name.

    Raises
    ------
    StackedKeyError
        If 'default_setup' is missing and class_name defines no default
        configuration
    StackedTypeError
        If class_name is not a valid _StackedDict class or subclass

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
    - Type variable T preserves the exact subclass type through the transformation

    See Also
    --------
    _StackedDict.__init__ : Constructor that uses this function
    _StackedDict.to_dict : Inverse operation (convert back to dict)
    """

    warnings.warn(
        "from_dict() free function is deprecated since 1.1.0 and will be removed in 1.5.0. Use ClassName.from_dict(dictionary, **class_options) instead. Example: NestedDictionary.from_dict(dictionary, default_setup={...})",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(class_name, type) or not issubclass(class_name, _StackedDict):
        raise StackedTypeError(
            f"class_name must be a _StackedDict class, got {type(class_name)}"
        )
    # The configuration is resolved by class_name._normalize_setup in __init__.
    dict_object: T = class_name(**class_options)

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
    children : tuple[_HKey, ...]
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
        self, key: Any, parent: "_HKey | None" = None, is_root: bool = False
    ) -> None:
        self.key: Any = key
        self.children: tuple[_HKey, ...] = ()
        self.parent: "_HKey | None" = parent
        self.is_root: bool = is_root

    @classmethod
    def build_forest(cls, stacked_dict: dict[Any, Any]) -> Self:
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
        Self
            Root node (with is_root=True, key=None) containing the forest.
            The root is an instance of the calling class; the nodes below it
            are plain ``_HKey`` instances, because ``_build_from_dict`` and
            ``add_child`` build ``_HKey`` explicitly.

        Examples
        --------
        >>> data = {'a': 1, 'b': {'c': 2}}
        >>> forest = _HKey.build_forest(data)
        >>> forest.get_child_keys()
        ['a', 'b']
        >>> forest.is_root
        True
        """
        root = cls(None, is_root=True)
        root._build_from_dict(stacked_dict)
        return root

    def _build_from_dict(self, current_dict: dict[Any, Any]) -> None:
        """
        Recursively build tree structure from a dictionary.

        Parameters
        ----------
        current_dict : dict
            Dictionary to process at current level
        """
        children_list: list[_HKey] = []

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

    def add_children(self, keys: list[Any]) -> tuple["_HKey", ...]:
        """
        Add multiple children at once (more efficient than repeated add_child).

        Parameters
        ----------
        keys : list[Any]
            list of keys to add as children

        Returns
        -------
        tuple[_HKey, ...]
            tuple of newly created child nodes

        Examples
        --------
        >>> node = _HKey('parent')
        >>> children = node.add_children(['a', 'b', 'c'])
        >>> len(node.children)
        3
        """
        new_children: list[_HKey] = []

        for key in keys:
            exists = any(child.key == key for child in self.children)
            if not exists:
                new_child = _HKey(key, parent=self)
                new_children.append(new_child)

        self.children = self.children + tuple(new_children)
        return tuple(new_children)

    def get_child(self, key: Any) -> "_HKey | None":
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

    def get_child_keys(self) -> list[Any]:
        """
        Get list of all child keys.

        Returns
        -------
        list[Any]
            list of keys for all children

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

    def get_path(self) -> list[Any]:
        """
        Get the path from root to this node.

        Traverses parent references to reconstruct the full path.

        Returns
        -------
        list[Any]
            list of keys from root to this node

        Examples
        --------
        >>> root = _HKey('a')
        >>> child = root.add_child('b')
        >>> grandchild = child.add_child('c')
        >>> grandchild.get_path()
        ['a', 'b', 'c']
        """
        path: list[Any] = []
        current: "_HKey | None" = self

        while current is not None and not current.is_root:
            path.append(current.key)
            current = current.parent

        return list(reversed(path))

    def find_by_path(self, path: list[Any]) -> "_HKey | None":
        """
        Find a node by following a path from this node.

        Parameters
        ----------
        path : list[Any]
            list of keys to follow

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
            child: "_HKey | None" = current.get_child(key)
            if child is None:
                return None
            current = child

        return current

    def get_all_paths(self) -> list[list[Any]]:
        """
        Get the full paths of every node in the subtree rooted at this node.

        Paths start at the root of the tree, in depth-first pre-order. On a
        non-root node, the first path is the path of the node itself; on the
        root, which has no key, only its descendants are listed.

        Returns
        -------
        list[list[Any]]
            list of all paths in the subtree

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> root.get_all_paths()
        [['a'], ['a', 'b'], ['a', 'c']]
        >>> root.find_by_path(['a']).get_all_paths()
        [['a'], ['a', 'b'], ['a', 'c']]
        """
        paths: list[list[Any]] = []
        # Path of the parent: collect_paths appends the key of each node,
        # this one included.
        base_path: list[Any] = self.get_path()[:-1] if not self.is_root else []

        def collect_paths(node: _HKey, current_path: list[Any]) -> None:
            if not node.is_root:
                node_path: list[Any] = current_path + [node.key]
                paths.append(node_path)

                for child in node.children:
                    collect_paths(child, node_path)
            else:
                for child in node.children:
                    collect_paths(child, current_path)

        collect_paths(self, base_path)
        return paths

    def get_descendants(self) -> list["_HKey"]:
        """
        Get all descendant nodes (children, grandchildren, etc.).

        Returns
        -------
        list[_HKey]
            list of all descendant nodes in DFS order
        """
        descendants: list[_HKey] = []

        def collect(node: _HKey) -> None:
            for child in node.children:
                descendants.append(child)
                collect(child)

        collect(self)
        return descendants

    def get_depth(self) -> int:
        """
        Get the depth of this node in the forest of keys.

        A top-level key is the root of its tree and has depth 0; each level
        below adds 1. The node returned by :meth:`build_forest`, which holds
        no key, also returns 0.

        Returns
        -------
        int
            Number of edges between this node and the root of its tree

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}})
        >>> root.find_by_path(['a']).get_depth()
        0
        >>> root.find_by_path(['a', 'b', 'c']).get_depth()
        2
        """
        depth: int = 0
        current: "_HKey | None" = self.parent

        while current is not None and not current.is_root:
            depth += 1
            current = current.parent

        return depth

    def get_max_depth(self) -> int:
        """
        Get the height of the subtree rooted at this node.

        On a key node, this is the number of edges on the longest path from
        the node down to a leaf: 0 for a leaf. On the node returned by
        :meth:`build_forest`, which holds no key and sits above the top-level
        keys, the extra edge makes the result the number of levels of the
        forest, that is the greatest depth of a key plus 1 (0 for an empty
        dictionary).

        Returns
        -------
        int
            Height of the subtree, or number of levels on the forest root

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}, 'd': 2})
        >>> root.find_by_path(['a']).get_max_depth()
        2
        >>> root.get_max_depth()
        3
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

    # def to_dict(self) -> dict[Any, Any]:
    #    """
    #    Convert the tree structure back to a nested dict.

    #    Returns
    #    -------
    #    dict[Any, Any]
    #        Nested dictionary representation of the tree
    #   """
    #    result: dict[Any, Any] = {}

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
        visit: "Callable[[_HKey], None] | None" = None,
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
        seen: set[int] = set()

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
        self, visit: "Callable[[_HKey], None] | None" = None
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
        visit : Optional[Callable[[_HKey], None]],
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
        self, visit: "Callable[[_HKey], None] | None" = None
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
        visit : Optional[Callable[[_HKey], None]],
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

    def bfs(self, visit: "Callable[[_HKey], None] | None" = None) -> Iterator["_HKey"]:
        """
        Breadth-First Search (level-order) traversal.

        BFS explores all nodes at depth N before moving to depth N+1.
        Uses a queue (deque) for optimal O(1) operations.

        This traversal is cycle-safe: nodes already seen are not enqueued
        again, preventing infinite loops in the presence of cycles.

        Parameters
        ----------
        visit : Optional[Callable[[_HKey], None]]
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
        queue: deque[_HKey] = deque([self])
        seen: set[int] = {id(self)}

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

    def dfs_find(self, predicate: Callable[["_HKey"], bool]) -> "_HKey | None":
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

    def bfs_find(self, predicate: Callable[["_HKey"], bool]) -> "_HKey | None":
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

    def find_all(self, predicate: Callable[["_HKey"], bool]) -> list["_HKey"]:
        """
        Find all nodes matching predicate.

        Parameters
        ----------
        predicate : Callable[[_HKey], bool]
            Function that returns True for target nodes

        Returns
        -------
        list[_HKey]
            list of all matching nodes

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}, 'c': {'b': 2}})
        >>> nodes = root.find_all(lambda n: n.key == 'b')
        >>> len(nodes)
        2
        >>> [n.get_path() for n in nodes]
        [['a', 'b'], ['c', 'b']]
        """
        results: list[_HKey] = []

        for node in self.dfs_preorder():
            if predicate(node):
                results.append(node)

        return results

    def find_by_key(
        self, key: Any, find_all: bool = False
    ) -> "_HKey | None | list[_HKey]":
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
        Optional[_HKey] or list[_HKey]
            Single node if find_all=False, list of nodes if value find_all=True

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

    def iter_by_level(self) -> Iterator[tuple[int, list["_HKey"]]]:
        """
        Iterate over nodes grouped by depth level.

        Yields tuples of (depth, nodes_at_depth) for each level of the tree.

        Yields
        ------
        tuple[int, list[_HKey]]
            (depth_level, list_of_nodes_at_that_level)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1, 'c': 2}})
        >>> for depth, nodes in root.iter_by_level():
        ...     keys = [n.key for n in nodes if not n.is_root]
        ...     if keys:
        ...         print(f'Level {depth}: {keys}')
        Level 0: ['a']
        Level 1: ['b', 'c']

        See Also
        --------
        bfs : Breadth-first traversal
        get_nodes_at_depth : Get nodes at specific depth
        """
        levels: dict[int, list[_HKey]] = defaultdict(list)

        for node in self.bfs():
            depth = node.get_depth() if not node.is_root else -1
            if depth >= 0:
                levels[depth].append(node)

        for depth in sorted(levels.keys()):
            yield depth, levels[depth]

    def get_nodes_at_depth(self, target_depth: int) -> list["_HKey"]:
        """
        Get all nodes at a specific depth.

        Parameters
        ----------
        target_depth : int
            Depth level to query (0 = root's children)

        Returns
        -------
        list[_HKey]
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

    def filter_paths(self, predicate: Callable[[list[Any]], bool]) -> list[list[Any]]:
        """
        Filter paths based on a predicate function.

        Parameters
        ----------
        predicate : Callable[[list[Any]], bool]
            Function that takes a path and returns True to include it

        Returns
        -------
        list[list[Any]]
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

    def map_nodes(self, func: Callable[["_HKey"], Any]) -> list[Any]:
        """
        Apply a function to all nodes and collect results.

        Parameters
        ----------
        func : Callable[[_HKey], Any]
            Function to apply to each node

        Returns
        -------
        list[Any]
            list of results from applying func to each node

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
        [['a'], ['a', 'x'], ['a', 'x', 'b']]

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

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the subtree rooted at this node.

        Returns
        -------
        dict[str, Any]
            Dictionary with the following entries:

            - ``total_nodes``: nodes of the subtree, this node included. On
              the node returned by :meth:`build_forest`, the count includes
              that node, which holds no key, so it is the number of keys
              plus 1.
            - ``leaf_count``: nodes without children.
            - ``max_depth``: result of :meth:`get_max_depth`, so the number
              of levels of the forest when called on the forest root.
            - ``avg_branching_factor``: mean number of children of the nodes
              that have children, rounded to 2 decimals.
            - ``total_paths``: number of paths returned by
              :meth:`get_all_paths`, one per key of the subtree.
            - ``levels``: number of levels of keys in the subtree.

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': {'c': 1}}, 'd': 2})
        >>> stats = root.get_statistics()
        >>> stats['total_nodes']
        5
        >>> stats['max_depth']
        3
        >>> stats['leaf_count']
        2
        >>> stats['levels']
        3
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

    def has_cycles(self) -> "tuple[bool, list[_HKey] | None]":
        """
        Check if the tree contains cycles (should not in a proper tree).

        Detects cycles by tracking visited nodes during DFS. A cycle exists
        if we encounter a node that's already in the current path (back edge).

        Returns
        -------
        tuple[bool, Optional[list[_HKey]]]
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
        visited: set[int] = set()
        rec_stack: set[int] = set()
        cycle_path: list[_HKey] = []

        def dfs_cycle_detect(node: _HKey, path: list[_HKey]) -> bool:
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

    def is_valid_tree(self) -> tuple[bool, list[str]]:
        """
        Validate that this is a proper tree structure.

        Checks multiple tree properties:

        * No cycles (acyclic)
        * Each non-root node has exactly one parent
        * Single root (or forest with explicit root marker)
        * All nodes reachable from root

        Returns
        -------
        tuple[bool, list[str]]
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
        issues: list[str] = []

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
            if node.parent is not None and not node.is_root:
                if node not in node.parent.children:
                    issues.append(f"Node {node.key} not in parent's children list")

        return len(issues) == 0, issues

    def check_parent_consistency(self) -> list[str]:
        """
        Check that parent-child relationships are consistent.

        Verifies that:

        * Each child's parent reference points to the correct parent
        * Each parent's children list contains the child

        Returns
        -------
        list[str]
            list of inconsistency messages (empty if consistent)

        Examples
        --------
        >>> root = _HKey.build_forest({'a': {'b': 1}})
        >>> issues = root.check_parent_consistency()
        >>> len(issues)
        0
        """
        issues: list[str] = []

        for node in self.dfs_preorder():
            for child in node.children:
                if child.parent != node:
                    issues.append(
                        f"Inconsistent parent: child {child.key} has parent {child.parent.key if child.parent is not None else 'None'} but is child of {node.key}"
                    )

        return issues

    @staticmethod
    def _check_arity(n: int, minimum: int, method: str) -> None:
        """
        Validate the arity argument of a tree predicate.

        Parameters
        ----------
        n : int
            Requested arity.
        minimum : int
            Smallest meaningful arity for the predicate.
        method : str
            Name of the calling predicate, used in the error message.

        Raises
        ------
        StackedValueError
            If ``n`` is below ``minimum``.
        """
        if n < minimum:
            raise StackedValueError(
                f"{method}() requires an arity n >= {minimum}", value=n
            )

    def is_complete_tree(self, n: int = 2) -> bool:
        """
        Check if this is a complete n-ary tree (binary by default).

        A complete tree has every level filled except possibly the last, whose
        nodes are as far left as possible. In BFS order, once a node has fewer
        than ``n`` children, no later node may have children. A node with more
        than ``n`` children makes the tree not n-ary, hence not complete.

        Nodes created with ``is_root=True`` are not checked: the root is the
        virtual container whose children are the top-level keys, so the number
        of top-level keys is free.

        Parameters
        ----------
        n : int, optional
            Arity of the tree, at least 2 (default: 2). Below 2 every tree would
            be trivially complete.

        Returns
        -------
        bool
            True if the tree is complete for arity ``n``.

        Raises
        ------
        StackedValueError
            If ``n`` is below 2.

        Examples
        --------
        >>> # Complete: the last level is filled from the left
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> d = b.add_child('d')
        >>> root.is_complete_tree()
        True

        >>> # Not complete: 'b' has no children while 'c', to its right, has one
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> d = c.add_child('d')
        >>> root.is_complete_tree()
        False

        Notes
        -----
        Terminology follows English usage. French *arbre complet* corresponds to
        English *perfect tree* (see ``is_perfect_tree``).

        See Also
        --------
        is_perfect_tree : Every level filled
        is_full_tree : Every internal node has exactly n children
        is_balanced : Height-balanced check
        """
        self._check_arity(n, 2, "is_complete_tree")

        queue: deque[_HKey] = deque([self])
        found_incomplete = False

        while queue:
            node = queue.popleft()
            if not node.is_root:
                count = len(node.children)
                if count > n:
                    return False
                if found_incomplete and count:
                    # A node after the first incomplete one must be a leaf
                    return False
                if count < n:
                    found_incomplete = True
            queue.extend(node.children)

        return True

    def is_perfect_tree(self, n: int = 2) -> bool:
        """
        Check if this is a perfect n-ary tree (binary by default).

        A perfect tree has every level filled: all leaves are at the same depth
        and every internal node has exactly ``n`` children.

        Nodes created with ``is_root=True`` are not checked for arity: the root
        is the virtual container whose children are the top-level keys.

        Parameters
        ----------
        n : int, optional
            Arity of the tree, at least 2 (default: 2). Below 2 every chain
            would be trivially perfect.

        Returns
        -------
        bool
            True if the tree is perfect for arity ``n``.

        Raises
        ------
        StackedValueError
            If ``n`` is below 2.

        Examples
        --------
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> d = b.add_child('d')
        >>> e = b.add_child('e')
        >>> f = c.add_child('f')
        >>> g = c.add_child('g')
        >>> root.is_perfect_tree()
        True
        >>> root.is_perfect_tree(n=3)
        False

        Notes
        -----
        Terminology follows English usage: a perfect tree is what French calls
        an *arbre complet*.

        See Also
        --------
        is_complete_tree : Last level may be partially filled
        is_full_tree : Every internal node has exactly n children
        is_balanced : Height-balanced check
        """
        self._check_arity(n, 2, "is_perfect_tree")

        leaves = list(self.iter_leaves())
        if leaves:
            first_leaf_depth = leaves[0].get_depth()
            if not all(leaf.get_depth() == first_leaf_depth for leaf in leaves):
                return False

        return all(
            len(node.children) == n
            for node in self.dfs_preorder()
            if node.has_children() and not node.is_root
        )

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

        def check_balance(node: _HKey) -> tuple[bool, int]:
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

    def count_nodes_by_degree(self) -> dict[int, int]:
        """
        Count nodes by their out-degree (number of children).

        Returns
        -------
        dict[int, int]
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
        degree_counts: dict[int, int] = defaultdict(int)

        for node in self.dfs_preorder():
            if not node.is_root:
                degree = len(node.children)
                degree_counts[degree] += 1

        return dict(degree_counts)

    def is_binary_tree(self) -> bool:
        """
        Check if this is a binary tree (all nodes have at most 2 children).

        Nodes created with ``is_root=True`` are not checked: the root is the
        virtual container whose children are the top-level keys, so the number
        of top-level keys is free.

        Returns
        -------
        bool
            True if every node other than the root has 0, 1, or 2 children

        Examples
        --------
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> root.is_binary_tree()
        True

        >>> d = root.add_child('d')  # Now has 3 children
        >>> root.is_binary_tree()
        False

        >>> # Three top-level keys, each with at most 2 children
        >>> forest = _HKey.build_forest({'a': {'b': 1, 'c': 2}, 'd': 3, 'e': 4})
        >>> forest.is_binary_tree()
        True
        """
        for node in self.dfs_preorder():
            if not node.is_root and len(node.children) > 2:
                return False
        return True

    def is_full_tree(self, n: int = 2) -> bool:
        """
        Check if this is a full n-ary tree (binary by default).

        A full tree (also called proper tree) has every internal node with
        exactly ``n`` children; leaves may be at different depths. With
        ``n=1`` the question is whether the tree is a chain.

        Nodes created with ``is_root=True`` are not checked: the root is the
        virtual container whose children are the top-level keys.

        Parameters
        ----------
        n : int, optional
            Expected number of children of every internal node, at least 1
            (default: 2).

        Returns
        -------
        bool
            True if every internal node has exactly ``n`` children.

        Raises
        ------
        StackedValueError
            If ``n`` is below 1.

        Examples
        --------
        >>> root = _HKey('a')
        >>> b = root.add_child('b')
        >>> c = root.add_child('c')
        >>> d = b.add_child('d')
        >>> e = b.add_child('e')
        >>> root.is_full_tree()
        True
        >>> root.is_full_tree(n=3)
        False

        See Also
        --------
        is_binary_tree : Check if binary
        is_perfect_tree : Check if perfect
        """
        self._check_arity(n, 1, "is_full_tree")

        return all(
            len(node.children) == n
            for node in self.dfs_preorder()
            if node.has_children() and not node.is_root
        )

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

    @override
    def __repr__(self) -> str:
        if self.is_root:
            return f"_HKey(ROOT, children={len(self.children)})"
        return f"_HKey(key={self.key!r}, children={len(self.children)})"


class _StackedDict(defaultdict[Any, Any]):
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
    *args : Mapping or iterable of (key, value) pairs, optional
        Dictionaries or iterables to initialize from
    default_setup : Mapping[str, Any]
        Configuration (keyword-only), see ``__init__``
    **kwargs : Any, optional
        Initialization data

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

    def __init__(
        self,
        *args: Mapping[Any, Any] | Iterable[tuple[Any, Any]],
        default_setup: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a new _StackedDict with configuration and optional data.

        The constructor requires configuration via the 'default_setup' parameter,
        which must contain at least 'indent' and 'default_factory' keys. Additional
        initialization data can be provided through args or kwargs.

        Parameters
        ----------
        *args : Mapping or iterable of (key, value) pairs
            Dictionaries, _StackedDict instances, or iterables of (key, value) pairs
        default_setup : Mapping[str, Any]
            Configuration with at least 'indent' and 'default_factory' keys.
            Keyword-only. The mapping is copied, never modified; subclasses
            complete it through ``_normalize_setup``.
        **kwargs : Any
            Direct key-value pairs to initialize

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

        # Initialize instance attributes

        self.indent: int = 0
        "indent is used to print the dictionary with json indentation"
        self._default_setup: set[tuple[str, Any]] = set()
        "default_setup is used to disseminate default parameters to stacked objects"

        super().__init__()
        self._apply_setup(type(self)._normalize_setup(default_setup))

        # Update dictionary

        if len(args):
            for item in args:
                if isinstance(item, self.__class__):
                    nested = item.deepcopy()
                elif isinstance(item, dict):
                    nested = self.__class__.from_dict(
                        item, default_setup=dict(self.default_setup)
                    )
                else:
                    nested = self.__class__.from_dict(
                        dict(item),
                        default_setup=dict(self.default_setup),
                    )
                self.update(nested)

        if kwargs:
            nested = self.__class__.from_dict(
                kwargs,
                default_setup=dict(self.default_setup),
            )
            self.update(nested)

    # ========================================================================
    # ALTERNATIVE CONSTRUCTORS
    # ========================================================================

    @classmethod
    def from_dict(cls, dictionary: dict[Any, Any], **class_options: Any) -> Self:
        """
        Recursively convert a standard dictionary to a ``_StackedDict`` or subclass.

        Alternative constructor that transforms a regular nested dictionary into a
        ``_StackedDict``-based structure. The target class is ``cls`` itself,
        eliminating the need to pass the class explicitly and preventing errors
        in recursive calls.

        Parameters
        ----------
        dictionary : dict
            The dictionary to transform (may be nested).
        **class_options : dict
            Initialization options, passed to ``cls`` at every level.
            ``default_setup`` is optional: as for the constructor, the
            configuration is resolved by ``cls._normalize_setup``, which
            supplies the default of the class when none is given.

        Returns
        -------
        Self
            New instance of the calling class containing the dictionary
            structure.

        Raises
        ------
        StackedKeyError
            If ``default_setup`` is missing and ``cls`` defines no default
            configuration (the base ``_StackedDict`` does not).

        Examples
        --------
        >>> nd = NestedDictionary.from_dict({'a': {'b': 1}})
        >>> nd['a']['b']
        1
        >>> strict = NestedDictionary.from_dict(
        ...     {'a': {'b': 1}},
        ...     default_setup={'indent': 0, 'default_factory': None}
        ... )

        Notes
        -----
        - Already-instantiated ``_StackedDict`` values are preserved as-is.
        - Regular ``dict`` values are recursively converted using ``cls``.
        - Non-dict values are assigned directly.

        See Also
        --------
        to_dict : Inverse operation.
        """
        # The configuration is resolved by cls._normalize_setup in __init__.
        dict_object = cls(**class_options)
        for key, value in dictionary.items():
            if isinstance(value, _StackedDict):
                dict_object[key] = value
            elif isinstance(value, dict):
                dict_object[key] = cls.from_dict(value, **class_options)
            else:
                dict_object[key] = value
        return dict_object

    # ========================================================================
    # PICKLE SUPPORT
    # ========================================================================

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        """
        Support pickle serialization.

        Returns a ``(callable, args)`` pair that pickle uses to reconstruct
        the instance. Uses the module-level ``_reconstruct`` function so that
        pickle can locate the callable by name across interpreter sessions.

        ``default_setup`` (including ``default_factory``) is preserved so that
        the reconstructed instance behaves identically to the original.

        Returns
        -------
        tuple
            ``(_reconstruct, (cls, dictionary, default_setup))``
        """
        return (
            _reconstruct,
            (self.__class__, self.to_dict(), dict(self.default_setup)),
        )

    # ========================================================================
    # SERIALIZATION METHODS (JSON + PICKLE)
    # ========================================================================

    def to_json(self, path: str | Path, indent: int | None = None) -> None:
        """
        Serialize this dictionary to a JSON file.

        Non-string keys are encoded as type-tagged strings of the form
        ``__type__:value`` (e.g., integer key ``42`` → ``"__int__:42"``,
        tuple key ``(1, 2)`` → ``"__tuple__:(1, 2)"``). Round-trips are
        lossless for supported types: ``str``, ``int``, ``float``, ``bool``,
        flat ``tuple``, flat ``frozenset``. File I/O is delegated to ``json.dump``.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        indent : int, optional
            JSON indentation level. Defaults to ``self.indent`` if not provided.

        Examples
        --------
        >>> nd = NestedDictionary({'a': {'b': 1}})
        >>> nd.to_json('/tmp/nd.json')

        See Also
        --------
        from_json : Reconstruct from a JSON file.
        """
        from .serialize import NestedDictionaryEncoder

        _indent = indent if indent is not None else self.indent
        with open(Path(path), "w", encoding="utf-8") as f:
            json.dump(self, f, cls=NestedDictionaryEncoder, indent=_indent or None)

    @classmethod
    def from_json(cls, path: str | Path, **class_options: Any) -> Self:
        """
        Reconstruct a ``_StackedDict`` (or subclass) from a JSON file.

        Non-string keys stored as ``__type__:value`` tagged strings are
        decoded back to their original Python types. Round-trips are lossless
        for supported types: ``str``, ``int``, ``float``, ``bool``, flat
        ``tuple``, flat ``frozenset``.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.
        **class_options : dict
            Passed to ``cls.from_dict``. The JSON file carries no
            configuration: ``default_setup`` gives it, and when it is absent
            ``cls._normalize_setup`` supplies the default of the class.

        Returns
        -------
        Self
            Reconstructed instance of the calling class.

        Raises
        ------
        StackedTypeError
            If the root of the JSON document is not an object (for example a
            list or a scalar), so no instance of the calling class is built.

        Examples
        --------
        >>> nd = NestedDictionary.from_json('/tmp/nd.json')
        >>> strict = NestedDictionary.from_json(
        ...     '/tmp/nd.json',
        ...     default_setup={'indent': 0, 'default_factory': None}
        ... )

        See Also
        --------
        to_json : Serialize to a JSON file.
        """
        from .serialize import _make_decoder_hook

        with open(Path(path), "r", encoding="utf-8") as f:
            loaded: object = json.load(
                f, object_pairs_hook=_make_decoder_hook(cls, class_options)
            )
        return cls._check_loaded(loaded, path)

    def to_pickle(
        self,
        path: str | Path,
        protocol: int | None = None,
    ) -> None:
        """
        Serialize this dictionary to a pickle file with SHA-256 verification.

        Writes two files: ``<path>`` (pickle) and ``<path>.sha256`` (hex digest).

        Parameters
        ----------
        path : str or Path
            Destination file path.
        protocol : int, optional
            Pickle protocol. Defaults to ``pickle.DEFAULT_PROTOCOL``.

        Warns
        -----
        UserWarning
            Pickle is unsafe with untrusted files.

        See Also
        --------
        from_pickle : Reconstruct from a pickle file.
        """
        from .serialize import _pickle_dump

        _pickle_dump(self, path, protocol=protocol)

    @classmethod
    def from_pickle(
        cls,
        path: str | Path,
        verify: bool = True,
        **class_options: Any,
    ) -> Self:
        """
        Reconstruct a ``_StackedDict`` (or subclass) from a pickle file.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file.
        verify : bool, optional
            If ``True`` (default), verify the SHA-256 sidecar before loading.
        **class_options : dict
            Not used directly (the pickled object carries its own state),
            but accepted for API symmetry with ``from_json``.

        Returns
        -------
        Self
            Reconstructed instance. The pickled object keeps its own class,
            which is the calling class or one of its subclasses.

        Raises
        ------
        StackedValueError
            If ``verify=True`` and the digest mismatches or sidecar is absent.
        StackedTypeError
            If the pickled object is not an instance of the calling class, for
            example a ``NestedDictionary`` file loaded with
            ``StrictNestedDictionary.from_pickle``.

        Warns
        -----
        UserWarning
            Pickle is unsafe with untrusted files.

        See Also
        --------
        to_pickle : Serialize to a pickle file.
        """
        from .serialize import _pickle_load

        return cls._check_loaded(_pickle_load(path, verify=verify), path)

    @classmethod
    def _check_loaded(cls, loaded: object, path: str | Path) -> Self:
        """
        Check that a deserialized object is an instance of the calling class.

        Shared by :meth:`from_json` and :meth:`from_pickle`, whose loaders
        return ``Any``. Instances of a subclass of ``cls`` are accepted.

        Parameters
        ----------
        loaded : object
            Object returned by the loader.
        path : str or Path
            Source file, used in the error message.

        Returns
        -------
        Self
            ``loaded``, unchanged.

        Raises
        ------
        StackedTypeError
            If ``loaded`` is not an instance of ``cls``.
        """
        if not isinstance(loaded, cls):
            raise StackedTypeError(
                f"'{path}' does not contain an instance of {cls.__name__}",
                expected_type=cls,
                actual_type=type(loaded),
            )
        return loaded

    # ========================================================================
    # CONFIGURATION (default_setup)
    # ========================================================================

    @classmethod
    def _normalize_setup(
        cls, setup: Mapping[str, Any] | Iterable[tuple[str, Any]] | None
    ) -> dict[str, Any]:
        """
        Build the configuration an instance of ``cls`` will use.

        Returns a new dict and never modifies ``setup``. The base class only
        checks that the required keys are present. Subclasses override this
        hook to supply defaults or to force the values that define them (for
        instance the ``default_factory`` of a strict or smooth variant), then
        delegate to ``super()``.

        Parameters
        ----------
        setup : Mapping[str, Any] or iterable of (str, Any) pairs, or None
            Requested configuration.

        Returns
        -------
        dict[str, Any]
            Configuration to apply.

        Raises
        ------
        StackedKeyError
            If ``setup`` is None, or if 'indent' or 'default_factory' is missing.
        """
        if setup is None:
            raise StackedKeyError(
                "Missing 'default_setup' argument. Pass default_setup={'indent': <int>, 'default_factory': <class|None>}.",
                key="default_setup",
            )
        normalized = dict(setup)
        if "indent" not in normalized:
            raise StackedKeyError(
                "Missing 'indent' argument in default settings", key="indent"
            )
        if "default_factory" not in normalized:
            raise StackedKeyError(
                "Missing 'default_factory' argument in default settings",
                key="default_factory",
            )
        return normalized

    def _apply_setup(self, setup: Mapping[str, Any]) -> None:
        """
        Apply a normalized configuration to this instance only.

        Every key is checked before any attribute is changed, so an invalid
        configuration leaves the instance untouched.

        Parameters
        ----------
        setup : Mapping[str, Any]
            Configuration returned by ``_normalize_setup``.

        Raises
        ------
        StackedAttributeError
            If a key is not an attribute of the instance.
        """
        for key in setup:
            if not hasattr(self, key):
                # You cannot initialize undefined attributes
                raise StackedAttributeError(
                    f"The key {key} is not an attribute of the {self.__class__} class.",
                    attribute=key,
                )
        for key, value in setup.items():
            setattr(self, key, value)
        self._default_setup = set(setup.items())

    def _propagate_setup(self, setup: Mapping[str, Any], visited: set[int]) -> None:
        """
        Apply a configuration to this instance and to every nested level.

        Each level normalizes the configuration through its own class, so a
        nested strict or smooth dictionary keeps its ``default_factory``.
        Levels already visited are skipped, which handles shared
        sub-structures and self-references.

        Parameters
        ----------
        setup : Mapping[str, Any]
            Configuration to propagate.
        visited : set[int]
            ``id()`` of the instances already configured.
        """
        if id(self) in visited:
            return
        visited.add(id(self))
        self._apply_setup(type(self)._normalize_setup(setup))
        for value in self.values():
            if isinstance(value, _StackedDict):
                value._propagate_setup(setup, visited)

    @property
    def default_setup(self) -> list[tuple[str, Any]]:
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
        default_setup.setter : set new configuration
        """
        priority = ["indent", "default_factory"]
        # Convert internal set of tuples to dict to deduplicate and access by key
        d = {k: v for (k, v) in self._default_setup}
        ordered: list[tuple[str, Any]] = []
        for p in priority:
            if p in d:
                ordered.append((p, d[p]))
        remaining = sorted(
            [(k, v) for (k, v) in self._default_setup if k not in priority],
            key=lambda kv: kv[0],
        )
        ordered.extend(remaining)
        return ordered

    # Asymmetric on purpose: the setter accepts any source of configuration,
    # the getter returns the normalized, ordered form (#106).
    @default_setup.setter
    def default_setup(
        self, value: _SetupSource  # pyright: ignore[reportPropertyTypeMismatch]
    ) -> None:
        """
        Replace the configuration and propagate it to every nested level.

        The new configuration is validated and normalized as in ``__init__``,
        then applied to this instance and to all nested ``_StackedDict``
        levels, which keeps the invariant that all levels share the same
        configuration. Each level normalizes it through its own class.

        Parameters
        ----------
        value : Mapping[str, Any] or iterable of (str, Any) pairs
            New configuration to apply. It is not modified.

        Raises
        ------
        StackedKeyError
            If 'indent' or 'default_factory' is missing.
        StackedAttributeError
            If a key is not an attribute of the instance.

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.default_setup = {'indent': 4, 'default_factory': None}
        >>> sd.indent, sd['a'].indent
        (4, 4)
        """
        self._propagate_setup(type(self)._normalize_setup(value), set())

    @override
    def __str__(self, padding: int = 0) -> str:
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

    @override
    def __copy__(self) -> Self:
        """
        Create a shallow copy of the _StackedDict.

        Creates a new _StackedDict with the same keys and values, but values
        are not recursively copied. Nested _StackedDict instances are referenced,
        not duplicated.

        Returns
        -------
        Self
            Shallow copy of the same class, with the same configuration

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

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """
        Create a deep copy of the _StackedDict.

        Implements the ``copy.deepcopy`` protocol. Every key and value is
        copied recursively, including mutable leaf values such as lists, so
        changes to the copy never affect the original.

        Parameters
        ----------
        memo : dict[int, object] or None, optional
            Mapping from ``id()`` of already copied objects to their copies,
            maintained by the ``copy`` module. Pass it through unchanged when
            calling this method from another ``__deepcopy__``. ``None`` starts
            a new copy operation.

        Returns
        -------
        Self
            Independent copy of the same class, with the same configuration

        Examples
        --------
        >>> import copy
        >>> sd = _StackedDict({'a': {'b': [1]}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd2 = copy.deepcopy(sd)
        >>> sd2['a']['b'].append(2)
        >>> sd['a']['b']
        [1]

        Notes
        -----
        - The new instance is registered in ``memo`` before its content is
          copied. An object reachable through several paths is therefore
          copied once and stays shared in the copy, and a dictionary that
          contains itself does not cause infinite recursion.
        - The class and ``default_setup`` of the original are preserved, so
          subclasses and the three public variants copy to their own type.

        See Also
        --------
        __copy__ : Shallow copy alternative
        deepcopy : Public method wrapper
        """

        if memo is None:
            memo = {}
        new = self.__class__(default_setup=dict(self.default_setup))
        memo[id(self)] = new
        for key, value in self.items():
            new[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return new

    @override
    def __setitem__(self, key: Any, value: Any) -> None:
        """
        set item with support for hierarchical keys.

        Supports both flat keys and hierarchical paths (as lists).
        For hierarchical keys, automatically creates intermediate
        _StackedDict levels as needed.

        Parameters
        ----------
        key : Any or list[Any]
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

    @override
    def __getitem__(self, key: Any) -> Any:
        """
        Get item with support for hierarchical keys.

        Supports both flat keys and hierarchical paths (as lists).
        For hierarchical keys, traverses the nested structure to
        retrieve the value at the specified path.

        Parameters
        ----------
        key : Any or list[Any]
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
        - list keys traverse the hierarchy
        - Raises KeyError if path doesn't exist

        See Also
        --------
        __setitem__ : set items with hierarchical keys
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

    @override
    def __delitem__(self, key: Any) -> None:
        """
        Delete item with support for hierarchical keys and cleanup.

        Deletes the item at the specified key or hierarchical path.
        For hierarchical keys, automatically removes empty parent
        dictionaries after deletion.

        Parameters
        ----------
        key : Any or list[Any]
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

    @override
    def __eq__(self, other: object) -> bool:
        """
        Check strict equality, like :meth:`equal`.

        ``a == b`` is ``a.equal(b)``: same class, same ``default_setup`` and
        same content. A plain ``dict`` is never equal to a nested dictionary,
        even with the same content; use :meth:`similar` to compare content.

        Parameters
        ----------
        other : object
            Object to compare with

        Returns
        -------
        bool
            True if ``other`` is equal to this dictionary

        Examples
        --------
        >>> setup = {'indent': 2, 'default_factory': None}
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup=setup)
        >>> sd == _StackedDict({'a': {'b': 1}}, default_setup=setup)
        True
        >>> sd == {'a': {'b': 1}}
        False
        >>> sd.similar({'a': {'b': 1}})
        True

        See Also
        --------
        equal : Strict equality
        isomorph : Same content, any class of the family
        similar : Same content only
        """

        return self.equal(other)

    @override
    def __ne__(self, other: object) -> bool:
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

    def equal(self, other: object) -> bool:
        """
        Check equality: same class, configuration, and content.

        Two _StackedDict instances are equal if they have:
        1. Identical class type (exact match, not subclasses)
        2. Identical default_setup configuration
        3. Identical dictionary structure and values

        This is the strictest comparison, and the one used by ``==``.

        Parameters
        ----------
        other : object
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
        >>> sd1.equal(sd2)
        True

        >>> # Different setup
        >>> sd3 = _StackedDict({'a': 1}, default_setup={'indent': 4, 'default_factory': None})
        >>> sd1.equal(sd3)
        False

        See Also
        --------
        __eq__ : Same as equal
        __ne__ : Inequality check
        isomorph : Same content, any class of the family
        similar : Same content only
        """

        if not isinstance(other, _StackedDict) or type(other) is not type(self):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def isomorph(self, other: object) -> bool:
        """
        Check if two nested dictionaries are isomorphic.

        Two structures are isomorphic if they:
        1. Are both _StackedDict instances (any class of the family)
        2. Have identical dictionary content (keys and values)

        They describe the same nested structure, up to the choice of class:
        a NestedDictionary and a StrictNestedDictionary with the same content
        differ only in what reading a missing key does. Configuration
        differences are ignored. A plain dict is never isomorphic.

        Parameters
        ----------
        other : object
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
        >>> sd1.equal(sd2)
        False
        >>> sd1.isomorph(sd2)
        True
        >>> sd1.isomorph({'a': 1})
        False

        See Also
        --------
        equal : Strict equality (includes class and setup)
        similar : Same content only, plain dicts accepted
        """

        if not isinstance(other, _StackedDict):
            return False

        return compare_dict(self.to_dict(), other.to_dict())

    def similar(self, other: object) -> bool:
        """
        Check if two structures have the same content, whatever holds it.

        Two structures are similar if they represent the same nested
        dictionary content, regardless of whether they are _StackedDict
        instances or plain dicts. Class and configuration are ignored.

        Parameters
        ----------
        other : object
            Dictionary to compare with

        Returns
        -------
        bool
            True if ``other`` is a dict with the same content

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': 1}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.similar({'a': {'b': 1}})
        True
        >>> sd.similar({'a': {'b': 2}})
        False

        Notes
        -----
        Checks if sd[k1]...[kn] == other[k1]...[kn] for all paths.
        This is the most permissive comparison method:
        ``equal`` implies ``isomorph``, which implies ``similar``.

        See Also
        --------
        equal : Strict equality
        isomorph : Same content, any class of the family
        """

        if not isinstance(other, dict):
            return False
        elif isinstance(other, _StackedDict):
            return compare_dict(self.to_dict(), other.to_dict())
        else:
            return compare_dict(self.to_dict(), dict(other))

    def unpacked_items(self) -> Generator[tuple[tuple[Any, ...], Any], None, None]:
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

    def unpacked_keys(self) -> Generator[tuple[Any, ...], None, None]:
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

    def unpacked_values(self) -> Generator[Any, None, None]:
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

    def to_dict(self) -> dict[Any, Any]:
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

    @override
    def copy(self) -> Self:
        """
        Create a shallow copy of the _StackedDict.

        Returns
        -------
        Self
            Shallow copy of the same class, with the same configuration

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

    def deepcopy(self) -> Self:
        """
        Create a deep copy of the _StackedDict.

        Creates a completely independent copy where all nested structures
        and mutable leaf values are recursively duplicated. Equivalent to
        ``copy.deepcopy(self)``.

        Returns
        -------
        Self
            Complete independent copy of the same class

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

        return copy.deepcopy(self)

    @override
    def pop(self, key: Any | list[Any], default: Any = _MISSING) -> Any:
        """
        Remove and return value at key or hierarchical path.

        Removes the specified key (flat or hierarchical) and returns its value.
        Automatically cleans up empty parent dictionaries after removal.
        If the key doesn't exist, returns the default value, ``None`` included,
        or raises an error when no default is given, as ``dict.pop`` does.

        Parameters
        ----------
        key : Any or list[Any]
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
        >>> sd.pop('nonexistent', None) is None  # None is a valid default
        True

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
                    if default is not _MISSING:
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
                if default is not _MISSING:
                    return default
                raise StackedKeyError(
                    f"Key path {key} does not exist.", key=key[-1], path=key[:-1]
                )
        else:
            # Handle flat keys
            if key in self:
                return super().pop(key)
            if default is not _MISSING:
                return default
            raise StackedKeyError(f"Key {key!r} does not exist.", key=key)

    @override
    def popitem(self) -> tuple[list[Any], Any]:
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
        path: list[Any] = []
        # Each entry is (current_dict, current_path)
        stack: list[tuple[Any, list[Any]]] = [(self, [])]

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

    @override
    def update(
        self,
        m: "SupportsKeysAndGetItem[Any, Any] | Iterable[tuple[Any, Any]] | None" = None,
        /,
        **kwargs: Any,
    ) -> None:
        """
        Update _StackedDict with key/value pairs from mapping, iterable, or kwargs.

        Merges the provided mapping, iterable of key-value pairs, or keyword
        arguments into this _StackedDict, converting regular dicts to _StackedDict
        instances recursively while preserving existing _StackedDict values.
        Inserted _StackedDict values keep their identity and receive this
        instance's configuration through the ``default_setup`` setter.

        Parameters
        ----------
        m : SupportsKeysAndGetItem or Iterable[tuple[Any, Any]], optional
            Positional-only. A mapping (dict, _StackedDict), any object with
            ``keys()`` and ``__getitem__``, or an iterable of (key, value)
            tuples to merge, as accepted by ``dict.update``. If None, only
            kwargs are used.
        **kwargs : Any
            Additional key/value pairs to merge

        Examples
        --------
        >>> sd = _StackedDict({'a': 1}, default_setup={'indent': 2, 'default_factory': None})

        >>> # From dict
        >>> sd.update({'b': 2, 'c': {'d': 3}})
        >>> sd['c']['d']
        3

        >>> # From iterable
        >>> sd.update([('e', 4), ('f', {'g': 5})])
        >>> sd['f']['g']
        5

        >>> # Using kwargs
        >>> sd.update(h=6, i={'j': 7})
        >>> sd['i']['j']
        7

        >>> # Combined
        >>> sd.update({'k': 8}, l=9)
        >>> sd['k'], sd['l']
        (8, 9)

        Notes
        -----
        - Accepts mappings (dict, _StackedDict, etc.) and any object with
          ``keys()`` and ``__getitem__``
        - Accepts iterables of (key, value) tuples
        - Accepts keyword arguments
        - Regular dicts are converted to _StackedDict recursively
        - _StackedDict values are accepted directly with synchronized config
        - Configuration is synchronized across all nested instances
        - Later values override earlier ones for duplicate keys

        See Also
        --------
        __init__ : Initialization with data
        __setitem__ : set individual items
        from_dict : Convert dict to _StackedDict
        """

        # Handle mapping or iterable argument
        if m is not None:
            # Convert to dict if it's an iterable of tuples
            if not isinstance(m, Mapping):
                try:
                    m = dict(m)
                except (TypeError, ValueError) as e:
                    raise StackedTypeError(
                        f"update() argument must be a mapping or iterable of pairs, got {type(m).__name__}",
                        expected_type=Mapping,
                        actual_type=type(m),
                    ) from e

            # Process the mapping
            for key, value in m.items():
                if isinstance(value, _StackedDict):
                    value.default_setup = self.default_setup
                    self[key] = value
                elif isinstance(value, dict):
                    nested_dict = self.__class__.from_dict(
                        value, default_setup=dict(self.default_setup)
                    )
                    self[key] = nested_dict
                else:
                    self[key] = value

        # Process kwargs

        for key, value in kwargs.items():
            if isinstance(value, _StackedDict):
                value.default_setup = self.default_setup
                self[key] = value
            elif isinstance(value, dict):
                nested_dict = self.__class__.from_dict(
                    value, default_setup=dict(self.default_setup)
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

    def key_list(self, key: Any) -> list[tuple[Any, ...]]:
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
        list[tuple[Any, ...]]
            list of paths (as tuples) containing the key

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

        __key_list: list[tuple[Any, ...]] = []

        if self.is_key(key):
            for keys in self.unpacked_keys():
                if key in keys:
                    __key_list.append(keys)
        else:
            raise StackedKeyError(
                f"Cannot find the key: {key} in the stacked dictionary", key=key
            )

        return __key_list

    def items_list(self, key: Any) -> list[Any]:
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
        list :
            list of values from paths containing the key

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

    def paths(self) -> "_Paths":
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

    def compact_paths(self) -> "_CPaths":
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
        >>> c_paths = sd.compact_paths()
        >>> c_paths.structure
        [['a', 'b', 'c']]

        >>> # Expand back to full paths
        >>> c_paths.expand()
        [['a'], ['a', 'b'], ['a', 'c']]

        Notes
        -----
        - The structure built from a dictionary is the canonical compact form
          of its paths; expanding it gives those paths back
        - Useful for path coverage analysis
        - More compact representation for deeply nested structures

        See Also
        --------
        paths : Get full paths view
        _CPaths : Compact view class documentation
        """

        return _CPaths(self)

    def dfs(
        self, node: Mapping[Any, Any] | None = None, path: list[Any] | None = None
    ) -> Generator[tuple[list[Any], Any], None, None]:
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
        ...     print(f'{path} -> {value}')
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

    def bfs(self) -> Generator[tuple[tuple[Any, ...], Any], None, None]:
        """
        Breadth-first traversal of the terminal values.

        Walks the dictionary level by level with a queue, and yields the
        keys whose value is not a nested dictionary, with their path: all
        those of depth N before those of depth N+1. Keys whose value is a
        nested dictionary are traversed but not yielded, so an empty nested
        dictionary yields nothing.

        Yields
        ------
        tuple
            (path_tuple, value) for each terminal value, in breadth-first
            order

        Examples
        --------
        >>> sd = _StackedDict({'a': {'b': {'c': 1}}}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.bfs())
        [(('a', 'b', 'c'), 1)]

        >>> sd = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3}, default_setup={'indent': 2, 'default_factory': None})
        >>> list(sd.bfs())
        [(('d',), 3), (('a', 'b'), 1), (('a', 'c'), 2)]

        Notes
        -----
        - Yields terminal values only, unlike :meth:`dfs`, which also yields
          the keys whose value is a nested dictionary
        - Returns paths as immutable tuples

        See Also
        --------
        dfs : Depth-first traversal
        _HKey.bfs : Tree-based BFS traversal
        """

        queue: deque[tuple[tuple[Any, ...], _StackedDict]] = deque(
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
        Compute the number of levels of the nested structure.

        This is the number of keys on the longest path. Top-level keys have
        depth 0, so the result is the greatest depth of a key plus 1. An
        empty dictionary has height 0.

        Returns
        -------
        int
            Number of keys on the longest path

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
        size : Count every key
        leaves : Get all leaf values
        """

        return max((len(path) for path in self.paths()), default=0)

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

        >>> # A key whose value is an empty dictionary counts
        >>> sd = _StackedDict({'a': {}}, default_setup={'indent': 2, 'default_factory': None})
        >>> sd.size()
        1

        Notes
        -----
        - Counts all keys at all levels
        - Equivalent to number of nodes in tree representation
        - Different from len() which only counts top-level keys

        See Also
        --------
        height : Get the number of levels
        __len__ : Get top-level key count
        """

        return sum(1 for _ in self.dfs())

    def leaves(self) -> list[Any]:
        """
        Extract the values of all leaf keys from the nested structure.

        A leaf is a key without children: its value is either not a nested
        dictionary, or an empty one. Returns the values of these keys.

        Returns
        -------
        list :
            list of all leaf values

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
        [_StackedDict(None, {})]

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

        return [
            value
            for _, value in self.dfs()
            if not isinstance(value, _StackedDict) or not value
        ]

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

        def check_balance(node: Any) -> tuple[int, bool]:
            if not isinstance(node, _StackedDict) or not node:
                return 0, True  # Height, is_balanced
            heights: list[int] = []
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

    def ancestors(self, value: Any) -> list[Any]:
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
        list :
            list of keys forming the path to the value (excluding final key)

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

    def __init__(self, stacked_dict: _StackedDict | None = None):
        self._stacked_dict = stacked_dict
        self._hkey: _HKey | None = None  # Lazy initialization

    def _ensure_hkey(self) -> "_HKey":
        """
        Ensure _HKey tree is built (lazy initialization).

        Returns
        -------
        _HKey
            The built tree structure

        Raises
        ------
        StackedKeyError
            If no dictionary is attached to this view
        """
        if self._stacked_dict is not None and self._hkey is None:
            self._hkey = _HKey.build_forest(self._stacked_dict)
        if self._hkey is None:
            raise StackedKeyError(
                "Cannot build path tree: no dictionary attached.",
                key="_hkey",
            )
        return self._hkey

    def __iter__(self) -> Iterator[list[Any]]:
        """
        Iterate over all hierarchical paths.

        Yields
        ------
        list[Any]
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

    def __contains__(self, path: list[Any]) -> bool:
        """
        Check if a path exists.

        Uses the optimized tree structure for O(n) lookup where n is path length,
        much faster than iterating all paths.

        Parameters
        ----------
        path : list[Any]
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

    @override
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

    @override
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

    @override
    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String showing class name and paths
        """
        return f"{self.__class__.__name__}({list(self)})"

    def get_children(self, path: list[Any]) -> list[Any]:
        """
        Get child keys at the next level after the given path.

        Parameters
        ----------
        path : list[Any]
            Path to query

        Returns
        -------
        list[Any]
            list of child keys, empty if path not found

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

    def has_children(self, path: list[Any]) -> bool:
        """
        Check if a path has any children.

        Parameters
        ----------
        path : list[Any]
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

    def get_subtree_paths(self, prefix: list[Any]) -> list[list[Any]]:
        """
        Get all paths that start with the given prefix.

        Parameters
        ----------
        prefix : list[Any]
            Prefix to filter by

        Returns
        -------
        list[list[Any]]
            list of paths with this prefix

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

        return node.get_all_paths()

    def filter_paths(self, predicate: Callable[[list[Any]], bool]) -> list[list[Any]]:
        """
        Filter paths based on a predicate function.

        Parameters
        ----------
        predicate : Callable[[list[Any]], bool]
            Function that returns True for paths to include

        Returns
        -------
        list[list[Any]]
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
        Get the number of levels of the paths.

        This is the number of keys on the longest path, that is the greatest
        depth of a key plus 1, since top-level keys have depth 0.

        Returns
        -------
        int
            Number of keys on the longest path

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': {'c': 1}}}))
        >>> paths.get_depth()
        3
        """
        hkey = self._ensure_hkey()
        return hkey.get_max_depth()

    def get_leaf_paths(self) -> list[list[Any]]:
        """
        Get all leaf paths (paths with no children).

        Returns
        -------
        list[list[Any]]
            list of leaf paths

        Examples
        --------
        >>> paths = _Paths(_StackedDict({'a': {'b': 1}, 'c': 2}))
        >>> paths.get_leaf_paths()
        [['a', 'b'], ['c']]
        """
        hkey = self._ensure_hkey()
        return [node.get_path() for node in hkey.iter_leaves()]

    def to_compact(self) -> "_CPaths":
        """
        Convert this _Paths to a _CPaths.

        Returns
        -------
        _CPaths
            Compact representation with the same paths
        """
        return _CPaths(self._stacked_dict)


_StructureSource: TypeAlias = _StackedDict | _HKey | list[Any] | dict[str, Any]
"Accepted by the _CPaths.structure setter: a nested mapping, a key tree or a compact structure."


class _CPaths(_Paths):
    """
    A lazy view providing compact representation of hierarchical paths.

    Extends _Paths to provide a factorized/compact representation where
    the hierarchical structure is represented as nested lists:
    - Leaf nodes: just the key
    - Internal nodes: [key, child1, child2, ...]

    Inside a node list, every element after the first is a child of the
    first: ``['b', 'c', 'd']`` is ``b`` with two children, while
    ``['b', ['c', 'd']]`` is the chain ``b``, ``c``, ``d``.

    Paths and compact structure are converted both ways:
    - Paths → Compact structure (factorization), which gives the canonical form
    - Compact structure → Paths (expansion)

    Each set of paths has exactly one canonical form. Other structures are
    accepted and can expand to the same paths, such as ``['a']`` for the leaf
    ``'a'``.

    .. warning::
       This is a private class (underscore prefix) and should not be instantiated
       directly by external code.

    Parameters
    ----------
    stacked_dict : _StackedDict
        The nested dictionary to create a compact view for

    Attributes
    ----------
    _structure : Optional[list[Any]]
        Compact representation of paths (lazy-built)

    Examples
    --------
    >>> data = _StackedDict({'a': 1, 'b': {'c': 2, 'd': 3}})
    >>> c_paths = _CPaths(data)
    >>> c_paths.structure
    ['a', ['b', 'c', 'd']]
    >>> list(c_paths)  # Inherited from _Paths
    [['a'], ['b'], ['b', 'c'], ['b', 'd']]

    See Also
    --------
    _Paths : Base class for path views
    """

    def __init__(self, stacked_dict: _StackedDict | None = None):
        super().__init__(stacked_dict)
        self._structure: list[Any] | None = None

    def _ensure_structure(self) -> "list[Any]":
        """
        Ensure compact structure is built (lazy initialization).

        Returns
        -------
        list[Any]
            The compact structure

        Raises
        ------
        StackedKeyError
            If no dictionary is attached to this view
        """
        if self._stacked_dict is not None and self._structure is None:
            self._structure = self._build_compact_structure()
        if self._structure is None:
            raise StackedKeyError(
                "Cannot build compact structure: no dictionary attached.",
                key="_structure",
            )
        return self._structure

    @staticmethod
    def _validate_structure(structure: list[Any]) -> None:
        """
        Validate the compact structure format.

        Parameters
        ----------
        structure : list[Any]
            Structure to validate

        Raises
        ------
        ValueError
            If structure format is invalid
        StackedTypeError
            If a key (a leaf, or the first element of a node list) is not
            hashable, so that it could not be a key of a nested dictionary.
            The error carries the path of the parent node.

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

        def check_key(key: object, path: list[object]) -> None:
            # hash() rather than isinstance(key, Hashable): a tuple holding
            # a list is a Hashable instance but cannot be hashed.
            try:
                _ = hash(key)
            except TypeError:
                raise StackedTypeError(
                    f"Structure key {key!r} is not hashable and cannot be a "
                    + "key of a nested dictionary",
                    expected_type=Hashable,
                    actual_type=type(key),
                    path=path,
                ) from None

        def validate_node(
            node: object, depth: int = 0, path: list[object] | None = None
        ) -> None:
            if depth > MAX_DEPTH:  # Prevent infinite recursion
                raise ValueError(
                    f"Structure too deeply nested (max depth: {MAX_DEPTH})"
                )
            parent_path: list[object] = path if path is not None else []

            if isinstance(node, list):
                items = cast(list[object], node)
                if len(items) == 0:
                    raise ValueError("Empty list not allowed in structure")
                # First element is the key, rest are children
                check_key(items[0], parent_path)
                for child in items[1:]:
                    validate_node(child, depth + 1, parent_path + [items[0]])
            else:
                # A leaf is a key with no children
                check_key(node, parent_path)

        for branch in structure:
            validate_node(branch)

    @property
    def structure(self) -> list[Any]:
        """
        Get the compact structure representation.

        Returns
        -------
        list[Any]
            Compact representation as nested lists

        Examples
        --------
        >>> c_paths = _CPaths(_StackedDict({'a': {'b': 1, 'c': 2}}))
        >>> c_paths.structure
        [['a', 'b', 'c']]
        """
        return self._ensure_structure()

    # Asymmetric on purpose: the setter accepts a nested mapping, a key tree or
    # a compact structure; the getter always returns the compact structure (#106).
    @structure.setter
    def structure(
        self, value: _StructureSource  # pyright: ignore[reportPropertyTypeMismatch]
    ) -> None:
        """
        set or build the compact structure representation.

        Accepts the following input types:
        - _StackedDict (or dict): the source nested mapping to analyze
        - _HKey: an already-built hierarchical key tree
        - list[Any]: a compact structure as nested lists

        Parameters
        ----------
        value : _StackedDict, dict, _HKey or list[Any]
            Input used to define the structure. A plain dict is wrapped in a
            _StackedDict with ``{'indent': 0, 'default_factory': None}``.

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
                value
                if isinstance(value, _StackedDict)
                else _StackedDict(
                    value, default_setup={"indent": 0, "default_factory": None}
                )
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

    def _build_compact_structure(self) -> list[Any]:
        """
        Build the compact structure recursively from _hkey.

        The algorithm traverses the hierarchical key tree (_hkey):
        - If node is a leaf: return key
        - If node has children: return [key, compact_child1, compact_child2, ...]

        Returns
        -------
        list[Any]
            Compact representation as nested lists

        Notes
        -----
        Uses the _hkey attribute which maintains the hierarchical structure
        of all keys. This is the core factorization algorithm.
        """

        def compact_node(node: _HKey) -> Any:
            """
            Recursively compact a node from _hkey tree.

            Parameters
            ----------
            node : _HKey
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
    def expand_structure(structure: list[Any]) -> list[list[Any]]:
        """
        Expand compact structure back to full paths.

        This is the inverse operation of compactification, establishing
        the bijection between compact and expanded representations.

        Parameters
        ----------
        structure : list[Any]
            Compact structure to expand

        Returns
        -------
        list[list[Any]]
            All expanded paths

        Examples
        --------
        >>> structure = ['a', ['b', 'c', 'd']]
        >>> _CPaths.expand_structure(structure)
        [['a'], ['b'], ['b', 'c'], ['b', 'd']]

        >>> # Inside a node list, a nested list is a chain, not siblings
        >>> _CPaths.expand_structure([['b', ['c', 'd']]])
        [['b'], ['b', 'c'], ['b', 'c', 'd']]

        >>> structure = [['x', ['y', 'z1', 'z2'], 'a']]
        >>> _CPaths.expand_structure(structure)
        [['x'], ['x', 'y'], ['x', 'y', 'z1'], ['x', 'y', 'z2'], ['x', 'a']]
        """
        all_paths = []

        def expand_node(node: Any, prefix: list[Any] | None = None) -> None:
            """
            Recursively expand a node.

            Parameters
            ----------
            node : Any
                Either a key (leaf) or [key, children...] (internal node)
            prefix : list[Any], optional
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

    def expand(self) -> list[list[Any]]:
        """
        Expand this instance's compact structure to full paths.

        Returns
        -------
        list[list[Any]]
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

    @override
    def __repr__(self) -> str:
        """
        Return technical representation with compact structure.

        Returns
        -------
        str
            String showing class name and compact structure

        Examples
        --------
        >>> setup = {'indent': 2, 'default_factory': None}
        >>> c_paths = _CPaths(_StackedDict({'a': 1}, default_setup=setup))
        >>> repr(c_paths)
        "_CPaths(['a'])"
        """
        return f"{self.__class__.__name__}({self.structure})"

    @override
    def __str__(self) -> str:
        """
        Return readable string representation.

        The prefix is the name of the actual class, so a public subclass such
        as ``CompactPathsView`` does not show the private ``_CPaths`` name.

        Returns
        -------
        str
            Class name, number of paths and compact structure

        Examples
        --------
        >>> setup = {'indent': 2, 'default_factory': None}
        >>> sd = _StackedDict({'a': {'b': 1}, 'c': 2}, default_setup=setup)
        >>> c_paths = _CPaths(sd)
        >>> str(c_paths)
        "_CPaths(3 paths): [['a', 'b'], 'c']"
        """
        return f"{self.__class__.__name__}({len(self)} paths): {self.structure}"

    # ========================================================================
    # COVERAGE ANALYSIS METHODS
    # ========================================================================

    @staticmethod
    def _compare_path_sets(
        paths1: list[list[Any]], paths2: list[list[Any]]
    ) -> tuple[Any, ...]:
        """
        Compare two sets of paths and return statistics (private helper).

        Parameters
        ----------
        paths1 : list[list[Any]]
            First set of paths
        paths2 : list[list[Any]]
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

    def is_covering(self, stacked_dict: "_StackedDict") -> bool:
        """
        Check if this _CPaths describes exactly the paths of a _StackedDict.

        With S the set of paths expanded from this structure and T the set of
        paths of ``stacked_dict``, the result is ``S == T``. It is stricter
        than full coverage: ``coverage()`` returns ``1.0`` as soon as T is
        included in S, while ``is_covering()`` also requires S to hold no
        other path.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        bool
            True if both sets of paths are equal

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1}, 'c': 2})
        >>> c_paths = _CPaths(sdict)
        >>> c_paths.is_covering(sdict)
        True

        >>> # Partial coverage
        >>> c_paths.structure = [['a']]  # Only covers 'a', not 'a.b' or 'c'
        >>> c_paths.is_covering(sdict)
        False

        >>> # Every path of sdict plus an extra one: full coverage, not equal
        >>> c_paths.structure = [['a', 'b'], 'c', 'e']
        >>> c_paths.coverage(sdict), c_paths.is_covering(sdict)
        (1.0, False)

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

    def coverage(self, stacked_dict: "_StackedDict") -> float:
        """
        Calculate the share of the paths of a _StackedDict found in this _CPaths.

        With S the set of paths expanded from this structure and T the set of
        paths of ``stacked_dict``, coverage is ``len(S & T) / len(T)``. Paths
        of S that are not in T do not change it, so the value stays between
        0.0 and 1.0. When T is empty, the result is 1.0 if S is empty too and
        0.0 otherwise.

        Parameters
        ----------
        stacked_dict : _StackedDict
            The _StackedDict to compare against

        Returns
        -------
        float
            Coverage between 0.0 and 1.0

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> c_paths = _CPaths(sdict)
        >>> c_paths.coverage(sdict)
        1.0

        >>> # Partial coverage: only 'a' and 'a.b' out of 4 paths
        >>> c_paths.structure = [['a', 'b']]
        >>> c_paths.coverage(sdict)
        0.5

        >>> # Extra paths do not raise the value above 1.0
        >>> c_paths.structure = [['a', 'b', 'c'], ['d'], ['e']]
        >>> c_paths.coverage(sdict)
        1.0

        Notes
        -----
        For a _CPaths created directly from a _StackedDict:
        _CPaths(sdict).coverage(sdict) will ALWAYS return 1.0
        because all paths from sdict are included.

        Use ``missing_paths()`` to list the extra paths and ``is_covering()``
        to check that the two sets are equal.
        """
        target_paths = list(_Paths(stacked_dict))
        expanded_paths = self.expand()

        _, set2, intersection, _, _ = self._compare_path_sets(
            expanded_paths, target_paths
        )

        if len(set2) == 0:
            return 0.0 if len(expanded_paths) > 0 else 1.0

        return len(intersection) / len(set2)

    def missing_paths(self, stacked_dict: "_StackedDict") -> list[list[Any]]:
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
        list[list[Any]]
            list of paths in _CPaths but not in stacked_dict

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1}})
        >>> c_paths = _CPaths(sdict)
        >>> c_paths.missing_paths(sdict)
        []

        >>> # Add extra paths
        >>> c_paths.structure = [['a', 'b', 'c'], ['d']]
        >>> c_paths.missing_paths(sdict)
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

    def uncovered_paths(self, stacked_dict: "_StackedDict") -> list[list[Any]]:
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
        list[list[Any]]
            list of paths in stacked_dict but not in _CPaths

        Examples
        --------
        >>> sdict = _StackedDict({'a': {'b': 1, 'c': 2}, 'd': 3})
        >>> c_paths = _CPaths(sdict)
        >>> c_paths.uncovered_paths(sdict)
        []

        >>> # Partial structure
        >>> c_paths.structure = [['a', 'b']]
        >>> c_paths.uncovered_paths(sdict)
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
