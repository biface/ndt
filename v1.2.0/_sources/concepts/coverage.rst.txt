Coverage
========

**Coverage** answers the question: *how much of a nested dictionary does a
given set of paths describe?*

This is useful when you work with a partial or independently defined
:class:`~ndict_tools.CompactPathsView` and want to know how much of a target
dictionary it accounts for.


Full coverage
--------------

A :class:`~ndict_tools.CompactPathsView` built directly from a dictionary
always covers it completely:

.. code-block:: python

    from ndict_tools import NestedDictionary

    nd = NestedDictionary({"a": {"b": 1, "c": 2}, "d": 3})
    cpaths = nd.compact_paths()

    cpaths.is_covering(nd)   # True
    cpaths.coverage(nd)      # 1.0


Partial coverage
-----------------

Assign a reduced structure to simulate partial coverage:

.. code-block:: python

    cpaths.structure = [['a', 'b']]   # only paths ['a'] and ['a', 'b']

    cpaths.is_covering(nd)   # False
    cpaths.coverage(nd)      # 0.5  — 2 out of 4 paths covered


Identifying uncovered paths
----------------------------

:meth:`~ndict_tools.CompactPathsView.uncovered_paths` returns the paths that
exist in the dictionary but are absent from the compact structure:

.. code-block:: python

    cpaths.uncovered_paths(nd)
    # [['a', 'c'], ['d']]


Identifying missing paths
--------------------------

:meth:`~ndict_tools.CompactPathsView.missing_paths` returns the paths present
in the compact structure but absent from the dictionary — useful when the
structure was set manually and may contain invalid entries:

.. code-block:: python

    cpaths.structure = [['a', 'b', 'c'], ['d'], ['e']]   # 'e' does not exist

    cpaths.missing_paths(nd)
    # [['e']]


Over-coverage
--------------

Coverage can exceed 1.0 when the compact structure contains more paths than
the dictionary has:

.. code-block:: python

    cpaths.structure = [['a', 'b', 'c'], ['d'], ['e']]
    cpaths.coverage(nd)   # 1.25  — 5 paths in structure, 4 in nd, 4 match


Practical example
------------------

Suppose you receive a specification describing the expected structure of a
nested dictionary, and you want to verify that an incoming payload satisfies
it:

.. code-block:: python

    from ndict_tools import NestedDictionary

    # Expected structure
    expected = NestedDictionary({
        "user": {"name": None, "email": None},
        "settings": {"theme": None},
    })
    spec = expected.compact_paths()

    # Incoming payload
    payload = NestedDictionary({
        "user": {"name": "Alice", "email": "alice@example.com"},
        "settings": {"theme": "dark", "lang": "fr"},
    })

    spec.is_covering(payload)        # False — 'settings.lang' is extra
    spec.uncovered_paths(payload)    # []    — no path from spec is missing
    spec.missing_paths(payload)      # []    — no path from spec is absent
    payload.compact_paths().coverage(expected)   # 1.0 — payload covers all of spec
