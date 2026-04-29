Part 4 — Extending the Package
================================

This part is aimed at developers who want to subclass, modify, or
contribute to **ndict-tools**. It explains the internal architecture and
the contracts that hold the package together.

.. note::

   All private classes described here are documented in the
   :doc:`/api/internal/tools` and :doc:`/api/internal/serialize` reference
   pages. Their interfaces are subject to change between releases.


The private/public split
-------------------------

The package is organised in three layers, each with a clearly defined
responsibility:

.. code-block:: text

   ndict_tools/
   ├── tools.py      ← private engine  (_StackedDict, _HKey, _Paths, _CPaths)
   ├── core.py       ← public wrappers (NestedDictionary, PathsView, …)
   ├── serialize.py  ← private I/O     (_encode_key, NestedDictionaryEncoder, …)
   ├── exception.py  ← exception hierarchy
   └── __init__.py   ← re-exports from core.py only

The rule is simple: **nothing from** ``tools.py`` **or** ``serialize.py``
**is exported from** ``__init__.py``. Users import exclusively from the
top-level package. Internal classes are underscored and considered
implementation details.

When you extend the package, subclass the public classes in ``core.py``,
not the private classes in ``tools.py`` directly. The public classes are
the stable surface.


The ``default_setup`` contract
--------------------------------

Every ``_StackedDict`` instance carries a ``default_setup`` dictionary
that bundles its configuration:

.. code-block:: python

   default_setup = {
       "indent": 2,              # spaces used when printing
       "default_factory": NestedDictionary,  # class for missing keys
   }

This contract must be respected by any method that creates a new instance
— constructors, ``from_dict``, ``from_json``, ``from_pickle``, and any
custom factory you write. The ``default_setup`` must round-trip cleanly
through serialisation so that the reconstructed object behaves identically
to the original.

When writing a custom subclass, forward ``default_setup`` explicitly:

.. code-block:: python

   from ndict_tools import NestedDictionary

   class TaggedDictionary(NestedDictionary):
       """A NestedDictionary that carries an optional tag."""

       def __init__(self, *args, tag: str = "", **kwargs):
           super().__init__(*args, **kwargs)
           self.tag = tag

       @classmethod
       def from_dict(cls, dictionary, *, default_setup, tag="", **kwargs):
           instance = super().from_dict(
               dictionary, default_setup=default_setup, **kwargs
           )
           instance.tag = tag
           return instance


The ``from_dict`` recursion pattern
--------------------------------------

:meth:`~ndict_tools.NestedDictionary.from_dict` converts a plain
:class:`dict` recursively. The recursion is driven by ``default_factory``
inside ``default_setup``: whenever a value is itself a :class:`dict`, the
method calls ``default_factory.from_dict(value, ...)`` to convert it to
the same class.

This means the ``default_factory`` must itself implement ``from_dict``.
All three public classes satisfy this contract. If you write a custom
subclass and use it as ``default_factory``, ensure it does too.

.. code-block:: python

   from ndict_tools import NestedDictionary

   plain = {"a": {"b": {"c": 1}}}

   # default_factory=NestedDictionary — all levels become NestedDictionary
   nd = NestedDictionary.from_dict(
       plain,
       default_setup={"indent": 0, "default_factory": NestedDictionary},
   )
   type(nd["a"])        # <class 'NestedDictionary'>
   type(nd["a"]["b"])   # <class 'NestedDictionary'>


The ``_HKey`` tree model
--------------------------

Internally, all paths are stored in a tree of :class:`~ndict_tools.tools._HKey`
nodes. Each node is an immutable named tuple that holds:

- its **key** (the single dictionary key this node represents),
- its **children** (a tuple of ``_HKey`` nodes, one per child key),
- a **leaf flag** (``True`` when this node has no children).

.. code-block:: text

   nd = {"a": {"b": 1, "c": 2}, "d": 3}

   _HKey tree:
   root
   ├── _HKey(key='a', leaf=False)
   │   ├── _HKey(key='b', leaf=True)
   │   └── _HKey(key='c', leaf=True)
   └── _HKey(key='d', leaf=True)

The tree is built once by :class:`~ndict_tools.tools._Paths` on first
access and reused for all subsequent operations (membership, children
lookup, subtree traversal, filtering). It is never mutated — a new tree
is built whenever the underlying dictionary changes.

``_HKey`` uses ``__slots__`` and is hashable. It is never exposed in the
public API.


Writing a custom path filter
-----------------------------

The simplest extension point is :meth:`~ndict_tools.PathsView.filter_paths`.
It takes any callable that accepts a path (a :class:`list` of keys) and
returns a :class:`bool`:

.. code-block:: python

   from ndict_tools import NestedDictionary

   nd = NestedDictionary({
       "prod":    {"db": {"host": "h1", "port": 5432}, "cache": {"ttl": 60}},
       "staging": {"db": {"host": "h2", "port": 5433}},
   })

   pv = nd.paths()

   # All leaf paths under 'prod'
   prod_leaves = pv.filter_paths(
       lambda p: p[0] == "prod" and not pv.has_children(p)
   )
   # [['prod', 'db', 'host'], ['prod', 'db', 'port'], ['prod', 'cache', 'ttl']]

   # All paths that contain the key 'port'
   port_paths = pv.filter_paths(lambda p: "port" in p)
   # [['prod', 'db', 'port'], ['staging', 'db', 'port']]

For more structured filtering, subclass :class:`~ndict_tools.tools._Paths`
and override ``filter_paths`` — then expose the subclass via a new method
on your custom dictionary class.


Writing a custom subclass
--------------------------

Here is a minimal but complete example: a ``FrozenNestedDictionary`` that
raises on any write attempt after construction:

.. code-block:: python

   from ndict_tools import NestedDictionary
   from ndict_tools.exception import StackedKeyError

   class FrozenNestedDictionary(NestedDictionary):
       """A NestedDictionary that cannot be modified after construction."""

       _frozen = False

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self._frozen = True

       def __setitem__(self, key, value):
           if self._frozen:
               raise StackedKeyError(
                   "FrozenNestedDictionary is read-only.",
                   key=key,
               )
           super().__setitem__(key, value)

       def __delitem__(self, key):
           if self._frozen:
               raise StackedKeyError(
                   "FrozenNestedDictionary is read-only.",
                   key=key,
               )
           super().__delitem__(key)

   nd = FrozenNestedDictionary({"a": 1})
   nd["a"]          # 1
   nd["b"] = 2      # raises StackedKeyError
