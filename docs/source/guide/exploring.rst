Part 2 — Exploring a Nested Dictionary
======================================

Part 1 built a nested dictionary and changed it key by key. This part looks
at a whole house at once: finding where a key or a value is, walking through
every key, measuring the structure, comparing two houses, and copying one.
Apart from the last section, nothing here changes the dictionary.

The house has two floors and a garden. Every room has lights and heating,
the kitchen has appliances, and the garage is still empty:

.. doctest::

   >>> from ndict_tools import NestedDictionary
   >>> house = NestedDictionary({
   ...     "ground floor": {
   ...         "kitchen": {
   ...             "lights": "ceiling",
   ...             "heating": {"type": "radiator", "power": 1500},
   ...             "appliances": {"oven": "electric", "fridge": "A++"},
   ...         },
   ...         "living room": {"lights": "floor lamp", "heating": {"type": "fireplace"}},
   ...         "garage": {},
   ...     },
   ...     "first floor": {
   ...         "bedroom": {"lights": "ceiling", "heating": {"type": "radiator", "power": 1000}},
   ...         "bathroom": {"lights": "mirror", "heating": {"type": "towel rail"}},
   ...     },
   ...     "garden": {"shed": "tools"},
   ... })

The methods below describe the keys of the house as the
:doc:`/concepts/key_forest` does: every key is a node, identified by its
path, and a key without children is a leaf. A leaf holds a value, or an empty
dictionary like the garage.


Searching across levels
-----------------------

:meth:`~ndict_tools.NestedDictionary.is_key` tells whether a key appears at
any level. It looks at keys only, never at values:

.. doctest::

   >>> house.is_key("heating")
   True
   >>> house.is_key("radiator")
   False

:meth:`~ndict_tools.NestedDictionary.key_list` returns the full paths that go
through a key, as tuples, and
:meth:`~ndict_tools.NestedDictionary.items_list` the values at the end of
these paths. Which rooms have a heating power, and how much?

.. doctest::

   >>> house.key_list("power")
   [('ground floor', 'kitchen', 'heating', 'power'), ('first floor', 'bedroom', 'heating', 'power')]
   >>> house.items_list("power")
   [1500, 1000]

These paths always go down to a leaf. For a key that holds a dictionary,
such as ``heating``, there is one path per leaf below it, not one per
``heating`` key:

.. doctest::

   >>> house.key_list("heating")[:2]
   [('ground floor', 'kitchen', 'heating', 'type'), ('ground floor', 'kitchen', 'heating', 'power')]
   >>> len(house.key_list("heating"))
   6

:meth:`~ndict_tools.NestedDictionary.occurrences` counts in the same way: it
counts the paths of leaves in which the key appears. Each of the four rooms
has its lights as a leaf, so ``lights`` appears 4 times. There are also four
``heating`` keys, but the kitchen and the bedroom have two settings each
under theirs, so ``heating`` appears in 6 paths:

.. doctest::

   >>> house.occurrences("lights")
   4
   >>> house.occurrences("heating")
   6

To count rooms that have heating, count the ``heating`` keys rather than
their occurrences, for instance with the paths of the house (see
:doc:`paths`):

.. doctest::

   >>> sum(1 for path in house.paths() if path[-1] == "heating")
   4

:meth:`~ndict_tools.NestedDictionary.ancestors` works the other way round:
given a value, it returns the path of the dictionary that holds it, without
the key of the value itself. Where is the fireplace?

.. doctest::

   >>> house.ancestors("fireplace")
   ['ground floor', 'living room', 'heating']

When a value appears more than once, ``ancestors`` returns the first one,
in the order of the keys. There are two radiators; the kitchen comes first:

.. doctest::

   >>> house.ancestors("radiator")
   ['ground floor', 'kitchen', 'heating']

A value that is not in the house raises
:class:`~ndict_tools.StackedValueError`:

.. doctest::

   >>> house.ancestors("swimming pool")
   Traceback (most recent call last):
       ...
   ndict_tools.exception.StackedValueError: Value swimming pool not found in the dictionary. (value: swimming pool)


Walking through every key
-------------------------

Several methods go through the whole dictionary. The examples use the living
room, which is small enough to show every step:

.. doctest::

   >>> living_room = house[["ground floor", "living room"]]

:meth:`~ndict_tools.NestedDictionary.unpacked_items` yields every leaf with
its path, as a tuple, in the order of the keys.
:meth:`~ndict_tools.NestedDictionary.unpacked_keys` and
:meth:`~ndict_tools.NestedDictionary.unpacked_values` yield the paths and
the values alone:

.. doctest::

   >>> list(living_room.unpacked_items())
   [(('lights',), 'floor lamp'), (('heating', 'type'), 'fireplace')]
   >>> list(living_room.unpacked_keys())
   [('lights',), ('heating', 'type')]
   >>> list(living_room.unpacked_values())
   ['floor lamp', 'fireplace']

:meth:`~ndict_tools.NestedDictionary.dfs`, depth first, also yields the keys
that hold a dictionary. Their value is that nested dictionary; here it
is converted for display:

.. doctest::

   >>> for path, value in living_room.dfs():
   ...     print(path, value.to_dict() if isinstance(value, dict) else value)
   ['lights'] floor lamp
   ['heating'] {'type': 'fireplace'}
   ['heating', 'type'] fireplace

:meth:`~ndict_tools.NestedDictionary.bfs`, breadth first, yields the values
that are not dictionaries, level by level: all those near the top before the
deeper ones. On the ground floor, the lights of both rooms come first:

.. doctest::

   >>> ground_floor = house["ground floor"]
   >>> [value for path, value in ground_floor.bfs()]
   ['ceiling', 'floor lamp', 'radiator', 1500, 'electric', 'A++', 'fireplace']

``bfs`` skips the garage, which holds no value. The garage is still a leaf:
:meth:`~ndict_tools.NestedDictionary.unpacked_keys` lists its path, and
:meth:`~ndict_tools.NestedDictionary.leaves`, which returns the values of all
leaves, ends with its empty dictionary:

.. doctest::

   >>> list(ground_floor.unpacked_keys())[-1]
   ('garage',)
   >>> len(ground_floor.leaves())
   8
   >>> ground_floor.leaves()[-1].to_dict()
   {}


Measuring the structure
-----------------------

``len`` counts the top-level keys only, as for a :class:`dict`: two floors
and a garden.
:meth:`~ndict_tools.NestedDictionary.size` counts every key at every level,
and :meth:`~ndict_tools.NestedDictionary.height` the number of levels:

.. doctest::

   >>> len(house)
   3
   >>> house.size()
   26
   >>> house.height()
   4

The four levels are the floor, the room, the equipment and its settings, as
in ``ground floor → kitchen → heating → power``.

:meth:`~ndict_tools.NestedDictionary.is_balanced` tells whether the branches
of the dictionary have about the same depth. The house is not balanced: the
garden is two levels deep, the ground floor four. The first floor is: both
rooms go down to their heating settings.

.. doctest::

   >>> house.is_balanced()
   False
   >>> house["first floor"].is_balanced()
   True

The :doc:`/concepts/key_forest` page gives the exact definitions of these
measures.


Comparing two dictionaries
--------------------------

The package offers three comparisons, from the strictest to the most
lenient:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Comparison
     - True when both have the same content and...
   * - ``==``, :meth:`~ndict_tools.NestedDictionary.equal`
     - the same class and the same ``default_setup``.
   * - :meth:`~ndict_tools.NestedDictionary.isomorph`
     - both are nested dictionaries of the package, whatever their class
       and configuration.
   * - :meth:`~ndict_tools.NestedDictionary.similar`
     - nothing more: the other one can even be a plain :class:`dict`.

Take the living room again, and the same room read from a validated
inventory, as a :class:`~ndict_tools.StrictNestedDictionary`:

.. doctest::

   >>> from ndict_tools import StrictNestedDictionary
   >>> room = {"lights": "floor lamp", "heating": {"type": "fireplace"}}
   >>> checked = StrictNestedDictionary(room)
   >>> living_room == checked
   False
   >>> living_room.isomorph(checked)
   True
   >>> living_room.similar(checked)
   True

The two rooms have the same content, but their classes differ in what
reading a missing key does, so they are not equal. Compared with the plain
dictionary they were built from, only ``similar`` holds:

.. doctest::

   >>> living_room == room
   False
   >>> living_room.isomorph(room)
   False
   >>> living_room.similar(room)
   True

Unlike a :class:`dict`, a nested dictionary is never ``==`` to a plain
dictionary. To compare its content with one, use
:meth:`~ndict_tools.NestedDictionary.similar`, or convert it first:

.. doctest::

   >>> living_room.to_dict() == room
   True


Copying and merging
-------------------

:meth:`~ndict_tools.NestedDictionary.to_dict` returns plain dictionaries at
every level, ready for any code that expects a :class:`dict`:

.. doctest::

   >>> plain = house.to_dict()
   >>> type(plain["first floor"]["bedroom"])
   <class 'dict'>

:meth:`~ndict_tools.NestedDictionary.copy` makes a shallow copy, as for a
:class:`dict`: a new top level, with the same nested dictionaries inside.
Changing a room through the copy changes the original:

.. doctest::

   >>> sketch = house.copy()
   >>> sketch == house
   True
   >>> sketch[["garden", "shed"]] = "bicycles"
   >>> house[["garden", "shed"]]
   'bicycles'

:meth:`~ndict_tools.NestedDictionary.deepcopy`, like
:func:`copy.deepcopy`, copies every level. The copy can change without
touching the original:

.. doctest::

   >>> project = house.deepcopy()
   >>> project[["garden", "shed"]] = "workshop"
   >>> house[["garden", "shed"]]
   'bicycles'
   >>> project[["garden", "shed"]]
   'workshop'

:meth:`~ndict_tools.NestedDictionary.update` works on the top-level keys,
like :meth:`dict.update`: a key it receives replaces the whole value. Here
the new garden replaces the old one, shed included:

.. doctest::

   >>> project.update({"garden": {"pond": "fish"}})
   >>> project["garden"].to_dict()
   {'pond': 'fish'}

To change one key deep inside, assign through its path, as in Part 1:

.. doctest::

   >>> project[["garden", "shed"]] = "workshop"
   >>> project["garden"].to_dict()
   {'pond': 'fish', 'shed': 'workshop'}
