``tools`` — Core Infrastructure
================================

.. warning::

   **Internal module — subject to change without notice.**
   Do not import or subclass these objects in production code.

.. automodule:: ndict_tools.tools
   :no-members:
   :no-undoc-members:

Module-level helpers
--------------------

.. autofunction:: ndict_tools.tools.compare_dict

.. autofunction:: ndict_tools.tools.unpack_items

``_HKey`` — Hierarchical key node
-----------------------------------

.. autoclass:: ndict_tools.tools._HKey
   :members:
   :show-inheritance:

``_StackedDict`` — Base engine
--------------------------------

.. autoclass:: ndict_tools.tools._StackedDict
   :members:
   :exclude-members: indent
   :show-inheritance:

``_Paths`` — Path view
-----------------------

.. autoclass:: ndict_tools.tools._Paths
   :members:
   :show-inheritance:

``_CPaths`` — Compact path view
---------------------------------

.. autoclass:: ndict_tools.tools._CPaths
   :members:
   :show-inheritance:
