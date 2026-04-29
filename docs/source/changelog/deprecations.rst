Deprecation Notices
===================

This page tracks all active and resolved deprecations in **ndict-tools**.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Symbol
     - Deprecated in
     - Removed in
     - Migration
   * - ``DictPaths``
     - 1.0.0
     - 1.2.0
     - Use :class:`~ndict_tools.CompactPathsView` or ``nd.compact_paths()``.
   * - ``from_dict`` *(free function)*
     - 1.1.0
     - 1.5.0 *(planned)*
     - Use ``NestedDictionary.from_dict(data, default_setup={...})``.

Active deprecations
-------------------

``from_dict`` free function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. deprecated:: 1.1.0

   The free function ``from_dict`` (importable from ``ndict_tools.tools``)
   is deprecated. It will be **removed in 1.5.0** (issue :issue:`81`).

   Calling it emits a :class:`DeprecationWarning` with an explicit removal
   notice.

   Migrate to the classmethod available on all three public classes:

   .. code-block:: python

      # Before — deprecated since 1.1.0, removed in 1.5.0
      from ndict_tools.tools import from_dict
      nd = from_dict(data, NestedDictionary, default_setup={...})

      # After
      nd = NestedDictionary.from_dict(data, default_setup={...})

Resolved deprecations
---------------------

``DictPaths``
~~~~~~~~~~~~~

.. deprecated:: 1.0.0
.. versionremoved:: 1.2.0

   ``DictPaths`` was deprecated in 1.0.0 and **removed in 1.2.0** (issue
   :issue:`49`).

   Use :class:`~ndict_tools.CompactPathsView` or call
   :meth:`~ndict_tools.NestedDictionary.compact_paths` on any
   :class:`~ndict_tools.NestedDictionary` instance.

   .. code-block:: python

      # Before — removed in 1.2.0
      from ndict_tools import DictPaths

      # After
      from ndict_tools import CompactPathsView
      cpaths = nd.compact_paths()
