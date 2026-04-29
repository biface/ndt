Public API Reference
====================

This section documents all classes and exceptions that are part of the stable
public API of **ndict-tools**. Everything listed here is exported from the
top-level ``ndict_tools`` package and covered by the project's stability
guarantees.

.. code-block:: python

    from ndict_tools import NestedDictionary, PathsView, CompactPathsView
    from ndict_tools import StackedDictionaryError  # and subclasses

.. toctree::
   :maxdepth: 1
   :hidden:

   core
   exceptions

Dictionary classes
------------------

.. autosummary::

   ndict_tools.NestedDictionary
   ndict_tools.StrictNestedDictionary
   ndict_tools.SmoothNestedDictionary

Path views
----------

.. autosummary::

   ndict_tools.PathsView
   ndict_tools.CompactPathsView

Exceptions
----------

.. autosummary::

   ndict_tools.StackedDictionaryError
   ndict_tools.NestedDictionaryException
   ndict_tools.StackedKeyError
   ndict_tools.StackedAttributeError
   ndict_tools.StackedTypeError
   ndict_tools.StackedValueError
   ndict_tools.StackedIndexError
