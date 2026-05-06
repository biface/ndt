Dictionary and Path Classes
============================

.. automodule:: ndict_tools.core
   :no-members:
   :no-undoc-members:

Dictionary classes
------------------

.. autoclass:: ndict_tools.NestedDictionary
   :members:
   :inherited-members: defaultdict, dict
   :show-inheritance:

.. autoclass:: ndict_tools.StrictNestedDictionary
   :members:
   :show-inheritance:

   All methods are inherited from :class:`~ndict_tools.NestedDictionary`.
   The only difference is the behaviour on unknown keys: a
   :class:`KeyError` is raised instead of silently creating a new nested
   dictionary.

.. autoclass:: ndict_tools.SmoothNestedDictionary
   :members:
   :show-inheritance:

   All methods are inherited from :class:`~ndict_tools.NestedDictionary`.
   The only difference is that every access to an unknown key returns a
   new empty :class:`~ndict_tools.SmoothNestedDictionary` — even at
   arbitrary depth.

Path views
----------

.. autoclass:: ndict_tools.PathsView
   :members:
   :inherited-members: _Paths
   :show-inheritance:

.. autoclass:: ndict_tools.CompactPathsView
   :members:
   :show-inheritance:

   Methods inherited from :class:`~ndict_tools.PathsView` (iteration,
   membership, filtering…) are documented on that class. Only the members
   specific to compact representation are listed here.
