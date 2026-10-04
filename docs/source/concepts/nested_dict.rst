Nested Dictionaries
===================

Python's built-in :class:`dict` can hold any value — including another
:class:`dict`. This natural nesting has no dedicated type in the standard
library, which makes common tasks (checking whether a key exists anywhere in
the hierarchy, counting its occurrences, or converting the whole structure to
JSON) unnecessarily verbose.

**ndict-tools** fills this gap by providing three ready-to-use classes built
on top of :class:`collections.defaultdict`.


What is a nested dictionary?
-----------------------------

A nested dictionary is a dictionary whose values may themselves be
dictionaries, to any depth:

.. doctest::

    >>> data = {
    ...     "project": {
    ...         "name": "ndict-tools",
    ...         "version": "1.1.0",
    ...         "authors": {
    ...             "lead": "biface",
    ...         },
    ...     },
    ...     "license": "MIT",
    ... }

Accessing ``data["project"]["authors"]["lead"]`` requires knowing the exact
path in advance. With **ndict-tools**, you can also write:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary(data)
    >>> nd[["project", "authors", "lead"]]
    'biface'
    >>> nd[["license"]]
    'MIT'

The list notation is a *hierarchical key*: each element is a level in the
nesting.


The three public classes
------------------------

All three classes accept the same construction patterns — a plain
:class:`dict`, a :class:`zip`, or a list of ``(key, value)`` pairs:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> # From a plain dict
    >>> nd = NestedDictionary({"a": {"b": 1}, "c": 2})
    >>> nd.to_dict()
    {'a': {'b': 1}, 'c': 2}
    >>> # From a zip
    >>> nd = NestedDictionary(zip(["a", "b"], [{"x": 1}, 2]))
    >>> nd.to_dict()
    {'a': {'x': 1}, 'b': 2}
    >>> # From a list of pairs
    >>> nd = NestedDictionary([("a", {"b": 1}), ("c", 2)])
    >>> nd.to_dict()
    {'a': {'b': 1}, 'c': 2}

The difference between the three classes lies in how they respond to an
**unknown key**.


NestedDictionary — lenient by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessing a missing key silently creates a new, empty
:class:`~ndict_tools.NestedDictionary` at that location. This mirrors the
behaviour of :class:`collections.defaultdict`.

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": 1})
    >>> missing = nd["missing"]  # does not raise
    >>> type(missing).__name__, len(missing)  # an empty NestedDictionary
    ('NestedDictionary', 0)
    >>> nd["missing"]["deeper"] = 42  # works: intermediate levels are created
    >>> nd[["missing", "deeper"]]
    42


StrictNestedDictionary — raises on missing keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessing a missing key raises a :class:`KeyError`, just like a plain
:class:`dict`. Use this class when unknown keys should never go unnoticed.

.. doctest::

    >>> from ndict_tools import StrictNestedDictionary
    >>> nd = StrictNestedDictionary({"a": 1})
    >>> nd["missing"]
    Traceback (most recent call last):
        ...
    KeyError: 'missing'


SmoothNestedDictionary — always returns a nested dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessing a missing key returns a new, empty
:class:`~ndict_tools.SmoothNestedDictionary`. Unlike :class:`NestedDictionary`,
every returned object is guaranteed to be a :class:`SmoothNestedDictionary`,
making deep chaining safe regardless of depth.

.. doctest::

    >>> from ndict_tools import SmoothNestedDictionary
    >>> nd = SmoothNestedDictionary({"a": 1})
    >>> result = nd["missing"]["even"]["deeper"]
    >>> type(result).__name__, len(result)  # an empty SmoothNestedDictionary
    ('SmoothNestedDictionary', 0)


Hierarchical key access
-----------------------

Any :class:`~ndict_tools.NestedDictionary` (and its subclasses) accepts a
Python :class:`list` as a key. Reading and writing both work:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": {"b": {"c": 42}}})
    >>> # Reading
    >>> nd[["a", "b", "c"]]
    42
    >>> nd[["a", "b"]].to_dict()
    {'c': 42}
    >>> # Writing: intermediate levels are created automatically
    >>> nd[["x", "y", "z"]] = 99
    >>> nd["x"]["y"]["z"]
    99
    >>> # Deleting: empty parents are cleaned up automatically
    >>> del nd[["x", "y", "z"]]
    >>> "x" in nd
    False


Constructing from an existing dict
-----------------------------------

The :meth:`~ndict_tools.NestedDictionary.from_dict` class method provides
an alternative constructor that recursively converts a plain dictionary.

Because a plain :class:`dict` carries no information about how missing keys
should behave or how the structure should be printed, you must supply that
configuration explicitly via ``default_setup``:

- ``indent`` — number of spaces used when printing the dictionary (``0``
  disables indentation).
- ``default_factory`` — the class instantiated for missing keys (set to
  ``NestedDictionary`` for lenient behaviour, ``None`` for strict behaviour).

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> plain = {"region": {"country": {"city": "Paris"}}}
    >>> nd = NestedDictionary.from_dict(
    ...     plain,
    ...     default_setup={"indent": 2, "default_factory": NestedDictionary},
    ... )
    >>> nd[["region", "country", "city"]]
    'Paris'


Searching across all levels
-----------------------------

Because all keys are tracked in the hierarchy, you can search without
knowing the depth:

.. doctest::

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({
    ...     "europe": {"france": {"capital": "Paris"}},
    ...     "asia": {"japan": {"capital": "Tokyo"}},
    ... })
    >>> nd.is_key("capital")  # exists somewhere
    True
    >>> nd.occurrences("capital")
    2
    >>> nd.key_list("capital")
    [('europe', 'france', 'capital'), ('asia', 'japan', 'capital')]
    >>> nd.items_list("capital")
    ['Paris', 'Tokyo']


Serialisation
--------------

A :class:`~ndict_tools.NestedDictionary` can be saved to and restored from
JSON or pickle without any extra setup:

.. testsetup:: serialisation

    import os
    import tempfile
    import warnings

    _previous_dir = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    warnings.simplefilter("ignore", UserWarning)

.. testcleanup:: serialisation

    os.chdir(_previous_dir)
    warnings.resetwarnings()

.. doctest:: serialisation

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": {"b": 1}})
    >>> # JSON round-trip
    >>> nd.to_json("nd.json")
    >>> nd2 = NestedDictionary.from_json(
    ...     "nd.json",
    ...     default_setup={"indent": 0, "default_factory": NestedDictionary},
    ... )
    >>> nd2.to_dict()
    {'a': {'b': 1}}
    >>> # Pickle round-trip (SHA-256 sidecar written automatically)
    >>> nd.to_pickle("nd.pkl")
    >>> nd3 = NestedDictionary.from_pickle("nd.pkl")
    >>> nd3.to_dict()
    {'a': {'b': 1}}
