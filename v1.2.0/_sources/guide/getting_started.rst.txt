Part 1 — Getting Started
========================

This page covers everything you need to go from zero to a working
:class:`~ndict_tools.NestedDictionary` in a few minutes.


Installation
------------

Install **ndict-tools** from PyPI using ``pip``:

.. code-block:: bash

   pip install ndict-tools

Or with ``uv``:

.. code-block:: bash

   uv add ndict-tools

The package requires **Python 3.10 or later** and has no runtime
dependencies beyond the standard library.


Creating a nested dictionary
-----------------------------

All three public classes share the same construction interface. The
simplest way is to pass a plain :class:`dict`:

.. code-block:: python

   from ndict_tools import NestedDictionary

   nd = NestedDictionary({"project": {"name": "ndict-tools", "version": "1.1.0"}})

You can also pass any iterable of ``(key, value)`` pairs or a ``zip``:

.. code-block:: python

   nd = NestedDictionary(zip(["a", "b"], [{"x": 1}, 2]))
   nd = NestedDictionary([("a", {"x": 1}), ("b", 2)])

To convert a pre-existing plain dictionary — including deeply nested ones
— use :meth:`~ndict_tools.NestedDictionary.from_dict`. Because a plain
:class:`dict` carries no information about how missing keys or printing
should behave, you must supply that configuration explicitly:

.. code-block:: python

   plain = {"europe": {"france": "Paris", "germany": "Berlin"}}
   nd = NestedDictionary.from_dict(
       plain,
       default_setup={"indent": 2, "default_factory": NestedDictionary},
   )

The ``default_setup`` dictionary accepts two keys:

- ``indent`` — number of spaces used when printing (``0`` disables indentation).
- ``default_factory`` — the class instantiated for missing keys; set to
  ``NestedDictionary`` for lenient behaviour or ``None`` for strict.


Reading, writing, and deleting
--------------------------------

Standard single-key access works exactly like a plain :class:`dict`:

.. code-block:: python

   nd = NestedDictionary({"a": {"b": 1}, "c": 2})

   nd["a"]          # NestedDictionary({'b': 1})
   nd["c"]          # 2

For multi-level access, pass a :class:`list` of keys — a *hierarchical key*:

.. code-block:: python

   nd[["a", "b"]]         # 1

   nd[["a", "b"]] = 99    # write — intermediate levels created automatically
   nd[["a", "b"]]         # 99

   del nd[["a", "b"]]     # delete — empty parent 'a' is cleaned up automatically
   "a" in nd              # False

You can also chain standard attribute access:

.. code-block:: python

   nd["a"]["b"]    # equivalent to nd[["a", "b"]]


Searching across all levels
----------------------------

Unlike a plain :class:`dict`, a :class:`~ndict_tools.NestedDictionary`
tracks every key at every depth. You can search without knowing the exact
path:

.. code-block:: python

   nd = NestedDictionary({
       "europe": {"france": {"capital": "Paris"}},
       "asia":   {"japan":  {"capital": "Tokyo"}},
   })

   nd.is_key("capital")       # True  — exists somewhere in the tree
   nd.occurrences("capital")  # 2     — appears at two different paths
   nd.key_list("capital")     # [('europe', 'france', 'capital'),
                              #  ('asia', 'japan', 'capital')]
   nd.items_list("capital")   # ['Paris', 'Tokyo']


Choosing the right variant
---------------------------

All three classes share the same interface. The only difference is what
happens when you access a key that does not exist.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Behaviour on unknown key
   * - :class:`~ndict_tools.NestedDictionary`
     - Returns a new empty :class:`~ndict_tools.NestedDictionary` and
       records the access. Mirrors :class:`collections.defaultdict`.
   * - :class:`~ndict_tools.StrictNestedDictionary`
     - Raises :class:`KeyError`. Use when unknown keys should never go
       unnoticed — configuration parsing, validated payloads.
   * - :class:`~ndict_tools.SmoothNestedDictionary`
     - Returns a new empty :class:`~ndict_tools.SmoothNestedDictionary`
       at any depth. Safe for deep chaining without prior key checks.

A quick rule of thumb:

- Building a structure from scratch → :class:`~ndict_tools.NestedDictionary`
- Validating or reading a known structure → :class:`~ndict_tools.StrictNestedDictionary`
- Navigating an unknown structure without guards → :class:`~ndict_tools.SmoothNestedDictionary`
