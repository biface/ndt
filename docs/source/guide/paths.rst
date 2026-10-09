Part 3 — Working with Paths
===========================

Every key of a nested dictionary is reached by a path, the list of keys from
the top level down to it. This part lists these paths, moves along them,
filters them, writes them in a compact form, and uses that form to check a
dictionary against a list of what it should contain. How paths are defined
is explained in :doc:`/concepts/paths`, :doc:`/concepts/compact_paths` and
:doc:`/concepts/coverage`.

The examples use the first floor of the house:

.. doctest::

   >>> from ndict_tools import NestedDictionary
   >>> first_floor = NestedDictionary({
   ...     "bedroom": {"lights": "ceiling", "heating": {"type": "radiator", "power": 1000}},
   ...     "bathroom": {"lights": "mirror", "heating": {"type": "towel rail"}},
   ... })


Listing the paths
-----------------

:meth:`~ndict_tools.NestedDictionary.paths` returns a
:class:`~ndict_tools.PathsView`. It plays for paths the role that
:meth:`dict.keys` plays for keys: it holds one path per key, at every level,
in the order of the keys.

.. doctest::

   >>> paths = first_floor.paths()
   >>> for path in paths:
   ...     print(path)
   ['bedroom']
   ['bedroom', 'lights']
   ['bedroom', 'heating']
   ['bedroom', 'heating', 'type']
   ['bedroom', 'heating', 'power']
   ['bathroom']
   ['bathroom', 'lights']
   ['bathroom', 'heating']
   ['bathroom', 'heating', 'type']

There are as many paths as keys, and ``in`` tests whether a path exists:

.. doctest::

   >>> len(paths)
   9
   >>> ["bathroom", "heating"] in paths
   True
   >>> ["bathroom", "heating", "power"] in paths
   False

Unlike :meth:`dict.keys`, the view does not follow later changes. It reads
the dictionary the first time it is used and keeps that state. To see the
paths of a dictionary after a change, call
:meth:`~ndict_tools.NestedDictionary.paths` again:

.. doctest::

   >>> first_floor["office"] = {"lights": "desk lamp"}
   >>> len(paths)
   9
   >>> len(first_floor.paths())
   11
   >>> del first_floor["office"]


Moving along the paths
----------------------

:meth:`~ndict_tools.PathsView.get_children` returns the keys just below a
path, and :meth:`~ndict_tools.PathsView.has_children` whether there are any:

.. doctest::

   >>> paths = first_floor.paths()
   >>> paths.get_children(["bedroom"])
   ['lights', 'heating']
   >>> paths.get_children(["bedroom", "lights"])
   []
   >>> paths.has_children(["bedroom", "heating"])
   True

:meth:`~ndict_tools.PathsView.get_subtree_paths` returns a path and every
path below it, :meth:`~ndict_tools.PathsView.get_leaf_paths` the paths of the
keys without children:

.. doctest::

   >>> paths.get_subtree_paths(["bedroom", "heating"])
   [['bedroom', 'heating'], ['bedroom', 'heating', 'type'], ['bedroom', 'heating', 'power']]
   >>> paths.get_leaf_paths()
   [['bedroom', 'lights'], ['bedroom', 'heating', 'type'], ['bedroom', 'heating', 'power'], ['bathroom', 'lights'], ['bathroom', 'heating', 'type']]

:meth:`~ndict_tools.PathsView.get_depth` returns the number of levels, like
:meth:`~ndict_tools.NestedDictionary.height`: the room, the equipment and its
settings.

.. doctest::

   >>> paths.get_depth()
   3


Filtering paths
---------------

:meth:`~ndict_tools.PathsView.filter_paths` keeps the paths for which a
function returns ``True``. The function receives each path as a list. Where
is the type of heating recorded, and which keys sit three levels down?

.. doctest::

   >>> paths.filter_paths(lambda path: path[-1] == "type")
   [['bedroom', 'heating', 'type'], ['bathroom', 'heating', 'type']]
   >>> paths.filter_paths(lambda path: len(path) == 3)
   [['bedroom', 'heating', 'type'], ['bedroom', 'heating', 'power'], ['bathroom', 'heating', 'type']]


Compact paths
-------------

The list of paths repeats every prefix: ``bedroom`` appears in five paths.
:meth:`~ndict_tools.NestedDictionary.compact_paths` returns a
:class:`~ndict_tools.CompactPathsView`, which writes each key once. Its
:attr:`~ndict_tools.CompactPathsView.structure` is a list with one entry per
top-level key: the key, then its children; a child that has children of its
own is a list in turn.

.. doctest::

   >>> compact = first_floor.compact_paths()
   >>> compact.structure
   [['bedroom', 'lights', ['heating', 'type', 'power']], ['bathroom', 'lights', ['heating', 'type']]]

The compact view holds the same paths as the other one, and offers the same
methods. :meth:`~ndict_tools.CompactPathsView.expand` writes them in full,
and the two views convert into each other:

.. doctest::

   >>> compact.expand() == list(paths)
   True
   >>> type(compact.to_paths()).__name__
   'PathsView'
   >>> type(paths.to_compact()).__name__
   'CompactPathsView'


Checking a dictionary against a list
------------------------------------

A compact view can serve as a checklist. When you take over a house, you
check, room by room, the lights and the type of heating. The checklist is
built from a dictionary whose keys are the items to check; its values do not
matter:

.. doctest::

   >>> checklist = NestedDictionary({
   ...     "bedroom": {"lights": None, "heating": {"type": None}},
   ...     "bathroom": {"lights": None, "heating": {"type": None}},
   ... }).compact_paths()

Four methods compare the paths of the checklist with those of a dictionary.
Two of them list the differences:

- :meth:`~ndict_tools.CompactPathsView.missing_paths` lists the paths of the
  checklist that the dictionary does not have: what is missing from the
  house;
- :meth:`~ndict_tools.CompactPathsView.uncovered_paths` lists the paths of
  the dictionary that the checklist does not have: what the house has beyond
  the list.

The first floor has everything on the list, and also the power of the
bedroom radiator, which the list does not mention:

.. doctest::

   >>> checklist.missing_paths(first_floor)
   []
   >>> checklist.uncovered_paths(first_floor)
   [['bedroom', 'heating', 'power']]

:meth:`~ndict_tools.CompactPathsView.coverage` is the share of the
dictionary's paths that the checklist describes: here 8 of 9.
:meth:`~ndict_tools.CompactPathsView.is_covering` is ``True`` only when the
two sets of paths are the same:

.. doctest::

   >>> round(checklist.coverage(first_floor), 2)
   0.89
   >>> checklist.is_covering(first_floor)
   False

Coverage looks at the dictionary only. In a bathroom without lights, every
path of the house is on the list, so coverage is complete, although an item
of the list is missing. Only ``missing_paths`` reports it:

.. doctest::

   >>> unlit = NestedDictionary({
   ...     "bedroom": {"lights": "ceiling", "heating": {"type": "radiator"}},
   ...     "bathroom": {"heating": {"type": "towel rail"}},
   ... })
   >>> checklist.coverage(unlit)
   1.0
   >>> checklist.missing_paths(unlit)
   [['bathroom', 'lights']]
   >>> checklist.is_covering(unlit)
   False

To check that nothing on the list is missing, test ``missing_paths``. To
check that the house matches the list exactly, use ``is_covering``:

.. doctest::

   >>> inspected = NestedDictionary({
   ...     "bedroom": {"lights": "ceiling", "heating": {"type": "radiator"}},
   ...     "bathroom": {"lights": "mirror", "heating": {"type": "towel rail"}},
   ... })
   >>> checklist.is_covering(inspected)
   True
