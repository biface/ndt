Compact Paths
=============

A :class:`~ndict_tools.CompactPathsView` expresses the same set of paths as
a :class:`~ndict_tools.PathsView`, but in a **factorised** form that groups
siblings under their shared parent. This makes the structure easier to read
and inspect at a glance.


The compact structure format
-----------------------------

The compact form is a Python list where each element represents a branch of
the dictionary tree:

- A **leaf** (a node with no children) is represented by its key alone.
- An **internal node** (a node with children) is represented as a list
  ``[key, child1, child2, ...]``, where each child follows the same rule
  recursively.

For the dictionary ``{"a": {"b": 1, "c": 2}, "d": 3}`` the compact structure
is:

.. code-block:: python

    [['a', 'b', 'c'], ['d']]

Reading it: ``'a'`` has two children ``'b'`` and ``'c'``; ``'d'`` is a leaf.

The full expanded paths ``['a']``, ``['a', 'b']``, ``['a', 'c']``, ``['d']``
can be recovered exactly from this compact form — the mapping is bijective.


Obtaining a compact paths view
-------------------------------

.. code-block:: python

    from ndict_tools import NestedDictionary

    nd = NestedDictionary({"a": {"b": 1, "c": 2}, "d": 3})
    cpaths = nd.compact_paths()


Reading the structure
----------------------

.. code-block:: python

    cpaths.structure
    # [['a', 'b', 'c'], ['d']]


Expanding to full paths
------------------------

:meth:`~ndict_tools.CompactPathsView.expand` reconstructs all paths from the
compact structure:

.. code-block:: python

    cpaths.expand()
    # [['a'], ['a', 'b'], ['a', 'c'], ['d']]

Iterating a :class:`~ndict_tools.CompactPathsView` directly yields the same
result (inherited from :class:`~ndict_tools.PathsView`):

.. code-block:: python

    list(cpaths)
    # [['a'], ['a', 'b'], ['a', 'c'], ['d']]


Setting a custom structure
---------------------------

The :attr:`~ndict_tools.CompactPathsView.structure` attribute is writable.
You can assign a hand-crafted compact structure and then expand or analyse it:

.. code-block:: python

    cpaths.structure = [['a', 'b']]   # only two of the original four paths

    cpaths.expand()
    # [['a'], ['a', 'b']]

You can also assign a :class:`~ndict_tools.NestedDictionary` directly to
rebuild the structure from a different source:

.. code-block:: python

    nd2 = NestedDictionary({"x": {"y": 1, "z": 2}})
    cpaths.structure = nd2
    cpaths.structure
    # [['x', 'y', 'z']]


Converting between views
-------------------------

Both view classes can be converted to the other in one step:

.. code-block:: python

    # PathsView → CompactPathsView
    paths  = nd.paths()
    cpaths = paths.to_compact()

    # CompactPathsView → PathsView
    paths2 = cpaths.to_paths()


Deeper nesting example
-----------------------

The compact format handles arbitrary depth. Given:

.. code-block:: python

    nd = NestedDictionary({
        "europe": {
            "france": {"capital": "Paris"},
            "germany": {"capital": "Berlin"},
        },
        "asia": {"japan": {"capital": "Tokyo"}},
    })

    nd.compact_paths().structure
    # [['europe', ['france', 'capital'], ['germany', 'capital']],
    #  ['asia', ['japan', 'capital']]]

Each branch is factorised: ``'europe'`` groups its two country sub-branches,
each of which groups its single ``'capital'`` leaf.
