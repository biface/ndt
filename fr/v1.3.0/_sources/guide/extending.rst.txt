Part 5 — Extending the Package
==============================

This part is for developers who write their own class on top of
**ndict-tools**. It explains what to subclass, how a subclass sets its
configuration, and what it inherits without writing anything.

.. testsetup:: extending

   import os
   import tempfile

   _previous_dir = os.getcwd()
   os.chdir(tempfile.mkdtemp())

.. testcleanup:: extending

   os.chdir(_previous_dir)


Public and private classes
--------------------------

The package has a private engine and a public interface:

.. code-block:: text

   ndict_tools/
   ├── tools.py      ← private engine   (_StackedDict, _HKey, _Paths, _CPaths)
   ├── core.py       ← public classes   (NestedDictionary, PathsView, …)
   ├── serialize.py  ← private helpers  (JSON key encoding, pickle checks)
   ├── exception.py  ← exceptions
   └── __init__.py   ← exports the public classes and the exceptions

``__init__.py`` exports the classes of ``core.py`` and the exceptions, and
nothing from ``tools.py`` or ``serialize.py``. Subclass the public classes,
usually :class:`~ndict_tools.NestedDictionary`, never the private ones: the
private classes can change between releases. Their reference is in
:doc:`/api/internal/tools` and :doc:`/api/internal/serialize`.


The configuration of a subclass
-------------------------------

The configuration of a nested dictionary, its ``default_setup``, is what
every level of the dictionary shares: ``indent`` and ``default_factory``
for the public classes. A subclass extends the package by adding its own
setting to it, which then follows the dictionary everywhere the
configuration goes.

Every way of building an instance resolves the configuration through one
class method, ``_normalize_setup``: the constructor, the ``default_setup``
setter, :meth:`~ndict_tools.NestedDictionary.from_dict` and
:meth:`~ndict_tools.NestedDictionary.from_json`. It receives the
``default_setup`` given by the caller, or ``None``, and returns the
configuration to apply as a new :class:`dict`, without changing the one it
received. :class:`~ndict_tools.NestedDictionary` supplies its default
there; :class:`~ndict_tools.StrictNestedDictionary` and
:class:`~ndict_tools.SmoothNestedDictionary` force their
``default_factory`` there.

Here is an ``Inventory`` of a house whose configuration also holds the unit
in which powers are given:

.. doctest:: extending

   >>> from ndict_tools import NestedDictionary
   >>> class Inventory(NestedDictionary):
   ...     """A nested dictionary describing the equipment of a house."""
   ...
   ...     def __init__(self, *args, **kwargs):
   ...         self.unit: str = "W"
   ...         super().__init__(*args, **kwargs)
   ...
   ...     @classmethod
   ...     def _normalize_setup(cls, setup):
   ...         normalized = dict(setup) if setup else {"indent": 2, "default_factory": cls}
   ...         normalized.setdefault("unit", "W")
   ...         return super()._normalize_setup(normalized)
   ...
   ...     def total_power(self):
   ...         total = sum(value for path, value in self.unpacked_items() if path[-1] == "power")
   ...         return f"{total} {self.unit}"

The two methods have separate jobs:

- ``__init__`` declares ``unit`` as an attribute of the instance, before
  calling the parent constructor, as :class:`~ndict_tools.NestedDictionary`
  does for ``indent``. The parent constructor then applies the
  configuration, and it only accepts keys that are attributes of the
  instance: without this declaration, ``unit`` would raise
  :class:`~ndict_tools.StackedAttributeError`.
- ``_normalize_setup`` gives the default configuration, completes a
  configuration that has no ``unit``, and hands the result to the parent
  class, which checks that ``indent`` and ``default_factory`` are present.

Four rules follow.

**1. A new setting is an instance attribute and a configuration key.**
The constructor sets it on every level, like ``indent``. A house measured
in kilowatts:

.. doctest:: extending

   >>> house = Inventory(
   ...     {
   ...         "kitchen": {"lights": "ceiling", "heating": {"type": "radiator", "power": 1.5}},
   ...         "bedroom": {"heating": {"type": "radiator", "power": 1.0}},
   ...     },
   ...     default_setup={"indent": 2, "default_factory": Inventory, "unit": "kW"},
   ... )
   >>> house.unit, house["kitchen"].unit, house[["kitchen", "heating"]].unit
   ('kW', 'kW', 'kW')
   >>> house.total_power()
   '2.5 kW'
   >>> house["bedroom"].total_power()
   '1.0 kW'

Assigning ``default_setup`` later changes the setting at every level, as
Part 1 shows for ``indent``.

**2. Define the default, or always pass it.** Without ``default_setup``,
an ``Inventory`` gets the configuration of ``_normalize_setup``, in watts:

.. doctest:: extending

   >>> Inventory({"office": {"heating": {"power": 500}}}).total_power()
   '500 W'

A subclass that does not override ``_normalize_setup`` inherits the
default of its parent, which has no ``unit``.

**3. Use the subclass as its own** ``default_factory``. The nested
dictionaries given to the constructor or to ``from_dict`` are converted
with the class that builds them. Keys created by reading a missing key,
however, come from ``default_factory``. The default above uses ``cls``, so
both kinds of levels are ``Inventory`` objects, and so are those of a
subclass of ``Inventory``:

.. doctest:: extending

   >>> home = Inventory({"kitchen": {"lights": "ceiling"}})
   >>> type(home["kitchen"]).__name__
   'Inventory'
   >>> type(home["attic"]).__name__
   'Inventory'

Without this, the two kinds of levels differ. A bare subclass keeps the
``default_factory`` of :class:`~ndict_tools.NestedDictionary`:

.. doctest:: extending

   >>> class BareInventory(NestedDictionary):
   ...     pass
   >>> bare = BareInventory({"kitchen": {"lights": "ceiling"}})
   >>> type(bare["kitchen"]).__name__
   'BareInventory'
   >>> type(bare["attic"]).__name__
   'NestedDictionary'

:class:`~ndict_tools.SmoothNestedDictionary` follows the same rule: its
``_normalize_setup`` sets ``default_factory`` to itself.

**4. The other constructors need no override.**
:meth:`~ndict_tools.NestedDictionary.from_dict`,
:meth:`~ndict_tools.NestedDictionary.copy` and
:meth:`~ndict_tools.NestedDictionary.deepcopy` build through the class of
the instance and keep its configuration, ``unit`` included:

.. doctest:: extending

   >>> kilowatts = {"indent": 2, "default_factory": Inventory, "unit": "kW"}
   >>> Inventory.from_dict({"garage": {"charger": {"power": 7.4}}}, default_setup=kilowatts).total_power()
   '7.4 kW'
   >>> copied = house.deepcopy()
   >>> type(copied).__name__, copied.unit, copied == house
   ('Inventory', 'kW', True)

:meth:`~ndict_tools.NestedDictionary.from_pickle` returns the object with
the configuration it had when written. Pickle stores the class by its name,
so a subclass that is pickled must be defined at the top level of an
importable module. A JSON file holds the content only (see
:doc:`serialization`): :meth:`~ndict_tools.NestedDictionary.from_json`
returns an ``Inventory`` with the default configuration, in watts, unless
``default_setup`` is given.

.. doctest:: extending

   >>> house.to_json("house.json")
   >>> Inventory.from_json("house.json").unit
   'W'
   >>> Inventory.from_json("house.json", default_setup=kilowatts).total_power()
   '2.5 kW'

Since ``unit`` is part of the configuration, it also counts in comparisons
(see :doc:`exploring`): the same content with another unit is isomorphic,
not equal.

.. doctest:: extending

   >>> watts = {"indent": 2, "default_factory": Inventory, "unit": "W"}
   >>> in_watts = Inventory(house.to_dict(), default_setup=watts)
   >>> in_watts == house, in_watts.isomorph(house)
   (False, True)


Where the model is described
----------------------------

The guide only covers the public interface. How the keys of a nested
dictionary form a forest, what its paths and compact paths are, and how
coverage is computed are explained in the :doc:`/concepts/index` section.
The private classes that implement them are documented in the API
reference.
