Paths
=====

A **path** is the ordered list of keys that leads from the root of a nested
dictionary to any node — intermediate or leaf. For example, in:

.. code-block:: python

    {"a": {"b": 1, "c": 2}, "d": 3}

the paths are ``['a']``, ``['a', 'b']``, ``['a', 'c']``, and ``['d']``.

Standard Python gives you no direct way to enumerate them. **ndict-tools**
exposes all paths through a lazy view object: :class:`~ndict_tools.PathsView`.


Obtaining a paths view
-----------------------

Call :meth:`~ndict_tools.NestedDictionary.paths` on any
:class:`~ndict_tools.NestedDictionary`:

.. code-block:: python

    from ndict_tools import NestedDictionary

    nd = NestedDictionary({"a": {"b": 1, "c": 2}, "d": 3})
    paths = nd.paths()


The view is *lazy*: no paths are materialised in memory until you actually
iterate or query the view. The underlying tree is built once on first access
and reused for all subsequent operations.


Iterating over paths
---------------------

.. code-block:: python

    for path in paths:
        print(path)

    # ['a']
    # ['a', 'b']
    # ['a', 'c']
    # ['d']

Paths are yielded in depth-first, pre-order — each node before its children.


Counting paths
--------------

.. code-block:: python

    len(paths)   # 4


Membership testing
------------------

Membership testing uses the internal tree structure and does not iterate all
paths:

.. code-block:: python

    ['a', 'b'] in paths    # True
    ['a', 'z'] in paths    # False
    ['d'] in paths         # True


Navigating the tree
--------------------

A :class:`~ndict_tools.PathsView` understands the parent–child relationships
between paths:

.. code-block:: python

    # Direct children of a path
    paths.get_children(['a'])       # ['b', 'c']
    paths.get_children(['a', 'b'])  # []  — leaf node

    # Check whether a path has children
    paths.has_children(['a'])       # True
    paths.has_children(['d'])       # False

    # All paths rooted at a given prefix
    paths.get_subtree_paths(['a'])  # [['a'], ['a', 'b'], ['a', 'c']]

    # Leaf paths only (no children)
    paths.get_leaf_paths()          # [['a', 'b'], ['a', 'c'], ['d']]

    # Maximum depth across all paths
    paths.get_depth()               # 2


Filtering paths
---------------

Pass a predicate to keep only the paths that match a condition:

.. code-block:: python

    # Paths deeper than one level
    paths.filter_paths(lambda p: len(p) > 1)
    # [['a', 'b'], ['a', 'c']]

    # Paths that contain the key 'b'
    paths.filter_paths(lambda p: 'b' in p)
    # [['a', 'b']]


Converting to compact form
---------------------------

A :class:`~ndict_tools.PathsView` can be converted to a
:class:`~ndict_tools.CompactPathsView` in one step:

.. code-block:: python

    compact = paths.to_compact()
    compact.structure   # [['a', 'b', 'c'], ['d']]

See :doc:`compact_paths` for the full picture.


Equality
--------

Two :class:`~ndict_tools.PathsView` objects are equal when they contain the
same set of paths, regardless of order:

.. code-block:: python

    nd2 = NestedDictionary({"d": 3, "a": {"c": 2, "b": 1}})
    nd.paths() == nd2.paths()   # True
