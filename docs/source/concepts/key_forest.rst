The Forest of Keys
==================

The keys of a nested dictionary form a forest: a set of rooted trees, one per
top-level key. **ndict-tools** builds this forest explicitly, with the private
class :class:`~ndict_tools.tools._HKey`, and computes on it. This page defines
the forest, matches each part of a nested dictionary with a part of the
forest, and explains why the operations and predicates of trees apply to
nested dictionaries.

The examples use this dictionary:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": {"b": {"c": 1}, "d": 2}, "e": {}, "f": 3})

Its forest has three trees, rooted at ``a``, ``e`` and ``f``. The leaves are
drawn with rounded corners:

.. mermaid::

    flowchart TD
        subgraph ta [tree a]
            a[a] --> ab[b]
            ab --> abc([c])
            a --> ad([d])
        end
        subgraph te [tree e]
            e([e])
        end
        subgraph tf [tree f]
            f([f])
        end


Definition
----------

Let :math:`D` be a nested dictionary.

**Vertices.** Every key of :math:`D`, at every level, is a vertex. The same key
can appear at several places (``"name"`` under two different parents), so a
vertex is identified by its path :math:`p = (k_0, k_1, \ldots, k_d)`, where
:math:`k_0` is a top-level key and each :math:`k_{i+1}` is a key of the value
of :math:`k_i`. :math:`V` is the set of these paths.

**Edges.** When the value of :math:`p` is a dictionary, there is an edge from
:math:`p` to :math:`p \cdot k` for each key :math:`k` of that dictionary.
:math:`E` is the set of these edges. A value that is not a dictionary is
carried by its key and is not a vertex.

**Forest.** :math:`F = (V, E)` is a forest. Every vertex of length greater
than 1 has exactly one parent, the path without its last key, and following
parents always ends at a top-level key. The roots are the top-level keys, the
paths of length 1. There is one tree per root, so :math:`F` has ``len(nd)``
trees.

The children of a vertex are ordered: dictionaries keep the insertion order of
their keys, and the forest keeps it too. The trees are *ordered trees*; "left"
and "right" below refer to that order.

.. doctest::

    >>> len(nd)
    3
    >>> list(nd.paths())
    [['a'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'd'], ['e'], ['f']]


Vocabulary
----------

**Depth.** The depth of a vertex is the number of edges between it and the
root of its tree: :math:`d(p) = |p| - 1`. A top-level key has depth 0, its
children depth 1, and so on.

**Leaf.** A vertex without children is a leaf. A key is a leaf when its value
is not a dictionary, or when its value is an empty dictionary. In the example,
``c``, ``d``, ``e`` and ``f`` are leaves; ``e`` holds an empty dictionary.
:math:`L` is the set of leaves.

**Internal node.** A vertex with at least one child: a key whose value is a
non-empty dictionary (``a`` and ``b``).

**Path.** The path of a vertex lists the keys from the root of its tree down to
it. It is the hierarchical key that reaches the vertex: ``nd[["a", "b", "c"]]``.

**Height and levels.** The height of a tree is the greatest depth of its
vertices, :math:`h(T) = \max_{p \in T} d(p)`; a tree reduced to its root has
height 0. A tree of height :math:`h` has :math:`h + 1` levels. The number of
levels of the forest is the largest number of levels of its trees.

.. doctest::

    >>> from ndict_tools.tools import _HKey
    >>> forest = _HKey.build_forest(nd)
    >>> forest.find_by_path(["a"]).get_depth()
    0
    >>> forest.find_by_path(["a", "b", "c"]).get_depth()
    2
    >>> [n.key for n in forest.iter_leaves()]
    ['c', 'd', 'e', 'f']


Measures of a nested dictionary
-------------------------------

The measures of :class:`~ndict_tools.NestedDictionary` are measures of its
forest:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - In the forest
     - On the example
   * - ``len(nd)``
     - number of trees, :math:`|R|`
     - 3
   * - ``nd.size()``
     - number of vertices, :math:`|V|`
     - 6
   * - ``len(nd.paths())``
     - number of vertices, one path each
     - 6
   * - ``nd.leaves()``
     - values of the leaves, in depth-first order
     - 4 values
   * - ``nd.paths().get_leaf_paths()``
     - paths of the leaves
     - 4 paths
   * - ``nd.height()``
     - number of levels, :math:`\max_{p \in V} d(p) + 1`
     - 3
   * - ``nd.occurrences(k)``
     - :math:`\sum_{\ell \in L} |\{ i : \ell_i = k \}|`
     - 2 for ``"a"``

``occurrences(k)`` counts the appearances of :math:`k` in the paths of the
leaves: ``a`` appears in ``['a', 'b', 'c']`` and in ``['a', 'd']``.

.. doctest::

    >>> nd.size()
    6
    >>> len(nd.leaves())
    4
    >>> nd.paths().get_leaf_paths()
    [['a', 'b', 'c'], ['a', 'd'], ['e'], ['f']]
    >>> nd.height()
    3
    >>> nd.occurrences("a")
    2


Why tree operations apply
-------------------------

:class:`~ndict_tools.tools._HKey` is a node of the forest: it holds a key
(``key``), a reference to its parent (``parent``) and the tuple of its
children (``children``), in insertion order. This is the usual representation
of an ordered rooted tree. The forest of a dictionary is built by
``_HKey.build_forest()``, which creates one node per vertex and one child per
edge, following the definition above. A :class:`~ndict_tools.PathsView` builds
it on its first access (see :doc:`paths`).

Since the structure is a forest of ordered trees, the algorithms written for
trees run on it unchanged, and their results translate back to the dictionary
through the paths of the nodes. The methods of ``_HKey`` fall into five
families:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Family
     - Methods
     - What they give on a dictionary
   * - Traversal
     - ``dfs_preorder``, ``dfs_postorder``, ``bfs``, ``iter_by_level``,
       ``iter_leaves``
     - the keys in depth-first, breadth-first or level order
   * - Search
     - ``find_by_path``, ``find_by_key``, ``find_all``, ``dfs_find``,
       ``bfs_find``
     - the node of a path, of a key, or of a predicate
   * - Measures
     - ``get_depth``, ``get_max_depth``, ``get_nodes_at_depth``,
       ``count_nodes_by_degree``, ``get_statistics``, ``get_balance_factor``
     - depths, heights, counts of children
   * - Shape
     - ``is_valid_tree``, ``has_cycles``, ``is_dag``, ``is_balanced``,
       ``is_binary_tree``, ``is_complete_tree``, ``is_perfect_tree``,
       ``is_full_tree``
     - whether the nesting has a given tree shape
   * - Transformation
     - ``map_nodes``, ``prune``, ``filter_paths``
     - a function applied to every key, a sub-forest, a subset of paths

On the tree rooted at ``a``:

.. doctest::

    >>> a = forest.find_by_path(["a"])
    >>> [n.key for n in a.dfs_preorder()]
    ['a', 'b', 'c', 'd']
    >>> [n.key for n in a.dfs_postorder()]
    ['c', 'b', 'd', 'a']
    >>> [n.key for n in a.bfs()]
    ['a', 'b', 'd', 'c']
    >>> forest.find_by_key("d").get_path()
    ['a', 'd']
    >>> [n.get_path() for n in forest.get_nodes_at_depth(1)]
    [['a', 'b'], ['a', 'd']]


Tree shapes
-----------

The shape predicates take an arity :math:`n`, 2 by default. On a forest, the
conditions apply to every key; the number of top-level keys is free, and the
levels are those of the whole forest.

**Full tree.** Every internal node has exactly :math:`n` children. Leaves can
be at different depths. ``is_full_tree(n)``, with :math:`n \geq 1`.

**Perfect tree.** Every internal node has exactly :math:`n` children and every
leaf has the same depth: all levels are filled. ``is_perfect_tree(n)``, with
:math:`n \geq 2`.

**Complete tree.** No node has more than :math:`n` children, and in
breadth-first order every node has :math:`n` children up to the first node that
has fewer; no node after it has children. All levels are filled except
possibly the last, whose leaves are as far left as possible.
``is_complete_tree(n)``, with :math:`n \geq 2`.

A perfect tree is full and complete. A full tree need not be complete, nor a
complete tree full.

.. doctest::

    >>> scores = NestedDictionary({
    ...     "north": {"home": 3, "away": 1},
    ...     "south": {"home": 2, "away": 2},
    ... })
    >>> shapes = _HKey.build_forest(scores)

.. mermaid::

    flowchart TD
        north[north] --> nh([home])
        north --> na([away])
        south[south] --> sh([home])
        south --> sa([away])

Every internal node has two children and every leaf has depth 1: the forest
is full, perfect and complete for :math:`n = 2`.

.. doctest::

    >>> shapes.is_full_tree(), shapes.is_perfect_tree(), shapes.is_complete_tree()
    (True, True, True)
    >>> shapes.is_perfect_tree(n=3)
    False
    >>> forest.is_full_tree(), forest.is_complete_tree()
    (False, False)


Limits of the model
-------------------

**Shared references.** A dictionary can hold the same sub-dictionary under two
keys. The forest has one subtree per occurrence, each with its own paths, so it
stays a forest:

.. doctest::

    >>> shared = NestedDictionary({"k": 1})
    >>> twice = NestedDictionary()
    >>> twice["p"] = shared
    >>> twice["q"] = shared
    >>> twice["p"] is twice["q"]
    True
    >>> list(twice.paths())
    [['p'], ['p', 'k'], ['q'], ['q', 'k']]

.. mermaid::

    flowchart TD
        p[p] --> pk([k])
        q[q] --> qk([k])

**Cycles.** A nested dictionary can contain itself, directly or through one of
its values. The construction allows it on purpose. Such a dictionary has no
forest of keys: its paths are infinite. Building the forest, and the methods
that walk every key (``dfs()``, ``paths()``, ``size()``, ``str()``,
``to_dict()``), raise :exc:`RecursionError`. ``copy.deepcopy()`` is the
exception: it copies the cycle.

.. doctest::

    >>> loop = NestedDictionary({"a": {}})
    >>> loop["a"]["back"] = loop
    >>> loop.size()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RecursionError: maximum recursion depth exceeded

For the same reason, a forest built from a dictionary never contains a cycle:
``has_cycles()`` reports none and ``is_dag()`` returns ``True``. These two
predicates check ``_HKey`` trees assembled by hand.
