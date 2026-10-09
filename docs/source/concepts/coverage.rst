Coverage
========

**Coverage** answers the question: *how much of a nested dictionary does a
given set of paths describe?*

This is useful when you work with a partial or independently defined
:class:`~ndict_tools.CompactPathsView` and want to know how much of a target
dictionary it accounts for.

All coverage methods compare two sets of paths: the paths expanded from the
compact structure, written *S* below, and the paths of the dictionary, written
*T*.

- :meth:`~ndict_tools.CompactPathsView.coverage` returns the share of *T*
  found in *S*: the number of paths in both sets divided by the number of
  paths in *T*. Paths of *S* that are not in *T* do not change it, so the
  value is always between 0.0 and 1.0.
- :meth:`~ndict_tools.CompactPathsView.is_covering` returns ``True`` when the
  two sets are equal.
- :meth:`~ndict_tools.CompactPathsView.uncovered_paths` lists the paths of
  *T* that are not in *S*.
- :meth:`~ndict_tools.CompactPathsView.missing_paths` lists the paths of *S*
  that are not in *T*.

``coverage() == 1.0`` therefore means that every path of the dictionary is
described, while ``is_covering()`` also requires that the structure describes
no other path.


Full coverage
--------------

A :class:`~ndict_tools.CompactPathsView` built directly from a dictionary
always covers it completely:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": {"b": 1, "c": 2}, "d": 3})
    >>> cpaths = nd.compact_paths()
    >>> cpaths.is_covering(nd)
    True
    >>> cpaths.coverage(nd)
    1.0


Partial coverage
-----------------

Assign a reduced structure to simulate partial coverage:

.. doctest::

    >>> cpaths.structure = [['a', 'b']]  # only paths ['a'] and ['a', 'b']
    >>> cpaths.is_covering(nd)
    False
    >>> cpaths.coverage(nd)  # 2 out of 4 paths covered
    0.5


Identifying uncovered paths
----------------------------

:meth:`~ndict_tools.CompactPathsView.uncovered_paths` returns the paths that
exist in the dictionary but are absent from the compact structure:

.. doctest::

    >>> cpaths.uncovered_paths(nd)
    [['a', 'c'], ['d']]


Identifying missing paths
--------------------------

:meth:`~ndict_tools.CompactPathsView.missing_paths` returns the paths present
in the compact structure but absent from the dictionary — useful when the
structure was set manually and may contain invalid entries:

.. doctest::

    >>> cpaths.structure = [['a', 'b', 'c'], ['d'], ['e']]  # 'e' does not exist
    >>> cpaths.missing_paths(nd)
    [['e']]


Extra paths
------------

A structure that describes every path of the dictionary and more has full
coverage, but it is not covering, since the two sets of paths differ:

.. doctest::

    >>> cpaths.structure = [['a', 'b', 'c'], ['d'], ['e']]
    >>> cpaths.coverage(nd)  # the 4 paths of nd are described
    1.0
    >>> cpaths.is_covering(nd)  # but ['e'] is not a path of nd
    False


Practical example
------------------

Suppose you receive a specification describing the expected structure of a
nested dictionary, and you want to verify that an incoming payload satisfies
it:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> # Expected structure
    >>> expected = NestedDictionary({
    ...     "user": {"name": None, "email": None},
    ...     "settings": {"theme": None},
    ... })
    >>> spec = expected.compact_paths()
    >>> # Incoming payload
    >>> payload = NestedDictionary({
    ...     "user": {"name": "Alice", "email": "alice@example.com"},
    ...     "settings": {"theme": "dark", "lang": "fr"},
    ... })
    >>> spec.is_covering(payload)  # 'settings.lang' is extra
    False
    >>> spec.uncovered_paths(payload)  # paths of the payload not in the spec
    [['settings', 'lang']]
    >>> spec.missing_paths(payload)  # no path of the spec is absent
    []
    >>> payload.compact_paths().coverage(expected)  # payload covers all of spec
    1.0
