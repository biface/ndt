Part 1 — Getting Started
========================

This part takes you from installation to a working
:class:`~ndict_tools.NestedDictionary`: building one, reading and changing it
with hierarchical keys, configuring it, and choosing between the three
public classes.


Installation
------------

Install **ndict-tools** from PyPI with ``pip``:

.. code-block:: bash

   pip install ndict-tools

or add it to a project managed by ``uv``:

.. code-block:: bash

   uv add ndict-tools

The package requires **Python 3.11 or later** and has no runtime dependencies
beyond the standard library.


Building a nested dictionary
----------------------------

Pass a plain :class:`dict` to the constructor. Here is a small house of one
floor: each room holds its equipment, and some equipment has settings of its
own.

.. doctest::

   >>> from ndict_tools import NestedDictionary
   >>> house = NestedDictionary({
   ...     "kitchen": {
   ...         "lights": "ceiling",
   ...         "heating": {"type": "radiator", "power": 1500},
   ...     },
   ...     "living room": {"lights": "floor lamp", "heating": {"type": "fireplace"}},
   ...     "garage": {},
   ...     "garden": {"shed": "tools"},
   ... })

The constructor converts every nested :class:`dict` into a
:class:`~ndict_tools.NestedDictionary`, at every level. The garage, which has
nothing in it yet, is an empty nested dictionary:

.. doctest::

   >>> type(house["kitchen"]).__name__
   'NestedDictionary'
   >>> type(house["kitchen"]["heating"]).__name__
   'NestedDictionary'
   >>> len(house["garage"])
   0

:meth:`~ndict_tools.NestedDictionary.to_dict` gives the content back as plain
dictionaries, which is the easiest way to look at it:

.. doctest::

   >>> house.to_dict()["living room"]
   {'lights': 'floor lamp', 'heating': {'type': 'fireplace'}}

The constructor accepts the same arguments as :class:`dict`: an iterable of
``(key, value)`` pairs, a ``zip``, or keyword arguments. Keyword arguments
are data, never settings:

.. doctest::

   >>> rooms = NestedDictionary([("bedroom", {"lights": "ceiling"}), ("bathroom", {"lights": "mirror"})])
   >>> rooms.to_dict()
   {'bedroom': {'lights': 'ceiling'}, 'bathroom': {'lights': 'mirror'}}
   >>> rooms = NestedDictionary(zip(["bedroom", "bathroom"], [{"lights": "ceiling"}, {"lights": "mirror"}]))
   >>> rooms.to_dict()
   {'bedroom': {'lights': 'ceiling'}, 'bathroom': {'lights': 'mirror'}}
   >>> NestedDictionary(shed="tools", pond={"fish": 3}).to_dict()
   {'shed': 'tools', 'pond': {'fish': 3}}

:meth:`~ndict_tools.NestedDictionary.from_dict` builds the same structure from
an existing plain dictionary. It is useful when the dictionary comes from
elsewhere, for example a parsed configuration file:

.. doctest::

   >>> parsed = {"cellar": {"lights": "bulb", "boiler": {"fuel": "gas"}}}
   >>> cellar = NestedDictionary.from_dict(parsed)
   >>> type(cellar["cellar"]["boiler"]).__name__
   'NestedDictionary'


Hierarchical keys
-----------------

A single key works as with a plain :class:`dict`, and keys can be chained:

.. doctest::

   >>> house["garden"]["shed"]
   'tools'

A :class:`list` of keys is a *hierarchical key*: it names a path from a
top-level key down to a nested one, and reaches it in one step.

.. doctest::

   >>> house[["kitchen", "heating", "power"]]
   1500

Writing through a hierarchical key creates the missing levels. The house gets
an attic, and the kitchen a new heating power:

.. doctest::

   >>> house[["attic", "window"]] = "skylight"
   >>> house["attic"].to_dict()
   {'window': 'skylight'}
   >>> house[["kitchen", "heating", "power"]] = 2000
   >>> house[["kitchen", "heating"]].to_dict()
   {'type': 'radiator', 'power': 2000}

``del`` and :meth:`~ndict_tools.NestedDictionary.pop` accept hierarchical keys
too. When the deleted key was the last one of its level, the emptied levels
above it are removed as well: taking the tools out of the shed removes the
garden.

.. doctest::

   >>> del house[["attic", "window"]]
   >>> "attic" in house
   False
   >>> house.pop(["garden", "shed"])
   'tools'
   >>> "garden" in house
   False

Only ``del`` and :meth:`~ndict_tools.NestedDictionary.pop` remove levels,
and only the levels they empty. A level that was empty from the start, like
the garage, stays:

.. doctest::

   >>> "garage" in house
   True

Only a :class:`list` is read as a path. A :class:`tuple` is an ordinary
hashable key, as in a plain :class:`dict`:

.. doctest::

   >>> grid = NestedDictionary({(0, 1): "door", 0: {1: "window"}})
   >>> grid[(0, 1)]
   'door'
   >>> grid[[0, 1]]
   'window'

The ``in`` operator, :meth:`~ndict_tools.NestedDictionary.get` and
:meth:`~ndict_tools.NestedDictionary.setdefault` take a single key, as for a
plain :class:`dict`. To test whether a path exists, ask the view of the paths
(see :doc:`paths`):

.. doctest::

   >>> house.get("garden", "no garden")
   'no garden'
   >>> ["kitchen", "heating", "type"] in house.paths()
   True


Configuration
-------------

Every nested dictionary carries a configuration, ``default_setup``, with two
settings:

- ``indent``: the number of spaces added at each level when the dictionary is
  printed;
- ``default_factory``: what reading a missing key does. A class creates an
  empty nested dictionary of that class under the key; ``None`` raises
  :class:`KeyError`.

Without ``default_setup``, a :class:`~ndict_tools.NestedDictionary` uses
``indent`` 0 and creates :class:`~ndict_tools.NestedDictionary` levels. Pass
``default_setup`` to choose; it needs both settings. Reading it returns the
settings as an ordered list of pairs:

.. doctest::

   >>> kitchen = NestedDictionary(
   ...     {"lights": "ceiling", "heating": {"type": "radiator", "power": 1500}},
   ...     default_setup={"indent": 2, "default_factory": NestedDictionary},
   ... )
   >>> kitchen.default_setup
   [('indent', 2), ('default_factory', <class 'ndict_tools.core.NestedDictionary'>)]

The list is the normalized form of the configuration: always the same order,
whatever the form it was given in. Printing uses ``indent``:

.. doctest::

   >>> print(kitchen)
   {
     lights : ceiling,
     heating : {
         type : radiator,
         power : 1500,
     },
   }

All levels of a nested dictionary share its configuration. Assigning
``default_setup`` changes it at every level:

.. doctest::

   >>> kitchen.default_setup = {"indent": 4, "default_factory": None}
   >>> kitchen["heating"].default_setup
   [('indent', 4), ('default_factory', None)]


Choosing a class
----------------

The three public classes have the same interface. They differ in what reading
a missing key does, that is, in their ``default_factory``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Reading a missing key
   * - :class:`~ndict_tools.NestedDictionary`
     - Creates an empty nested dictionary under the key, like
       :class:`collections.defaultdict`. The ``default_factory`` can be
       changed in ``default_setup``.
   * - :class:`~ndict_tools.StrictNestedDictionary`
     - Raises :class:`KeyError`. The ``default_factory`` is always ``None``.
   * - :class:`~ndict_tools.SmoothNestedDictionary`
     - Creates an empty :class:`~ndict_tools.SmoothNestedDictionary` under the
       key. The ``default_factory`` is always
       :class:`~ndict_tools.SmoothNestedDictionary`.

With a :class:`~ndict_tools.NestedDictionary`, reading a room that does not
exist adds it:

.. doctest::

   >>> plan = NestedDictionary({"kitchen": {"lights": "ceiling"}})
   >>> plan["office"].to_dict()
   {}
   >>> "office" in plan
   True

This suits a house you are still furnishing. To check an inventory that is
already complete, a :class:`~ndict_tools.StrictNestedDictionary` reports the
unknown room instead:

.. doctest::

   >>> from ndict_tools import StrictNestedDictionary
   >>> inventory = StrictNestedDictionary({"kitchen": {"lights": "ceiling"}})
   >>> inventory["office"]
   Traceback (most recent call last):
       ...
   KeyError: 'office'

:class:`~ndict_tools.StrictNestedDictionary` and
:class:`~ndict_tools.SmoothNestedDictionary` keep their ``default_factory``
whatever ``default_setup`` says. Only ``indent`` can be chosen:

.. doctest::

   >>> from ndict_tools import SmoothNestedDictionary
   >>> sketch = SmoothNestedDictionary(default_setup={"indent": 2, "default_factory": None})
   >>> sketch.default_setup
   [('indent', 2), ('default_factory', <class 'ndict_tools.core.SmoothNestedDictionary'>)]
   >>> type(sketch["attic"]["window"]).__name__
   'SmoothNestedDictionary'

.. warning::

   Because reading a missing key adds it, a tool that reads keys it does not
   know changes the data. The variable viewers of debuggers built on
   ``pydevd``, such as those of PyCharm and VS Code, read the attributes of a
   dictionary subclass as keys first: inspecting a
   :class:`~ndict_tools.NestedDictionary` can add keys such as
   ``default_setup`` or ``_default_setup``. To inspect without side effects,
   use a :class:`~ndict_tools.StrictNestedDictionary` or set
   ``default_factory`` to ``None``.

   Keep in mind the difference between the attribute ``house.default_setup``,
   the configuration, and the item ``house["default_setup"]``, a key of the
   data.


Where to go next
----------------

- :doc:`paths` lists, filters and compares the paths of a nested dictionary.
- :doc:`serialization` saves a nested dictionary to JSON or pickle.
- :doc:`extending` explains how to write your own subclass.
