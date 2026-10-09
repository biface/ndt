Changelog
=========

.. toctree::
   :maxdepth: 1
   :hidden:

   history
   deprecations

Version history
---------------

Each published version keeps its own documentation on the
`GitHub Pages archive <https://biface.github.io/ndt/>`_, under ``/vX.Y.Z/``;
from 1.3.0, the French translation is under ``/fr/vX.Y.Z/``. Read the Docs
serves `stable <https://ndict-tools.readthedocs.io/en/stable/>`_ (the last
published version) and `latest <https://ndict-tools.readthedocs.io/en/latest/>`_
(the development version). Earlier 0.x versions are not documented; the
documentation of 0.9.0 is kept until October 2027.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Version
     - Date
     - Summary
   * - `1.3.0 <https://biface.github.io/ndt/v1.3.0/>`__
     - *(planned)*
     - Python 3.11 minimum, typed package (``py.typed``), strict ``==``
       (breaking), Guide and Concepts rewritten, French translation, per-version
       documentation archive.
   * - `1.2.0 <https://biface.github.io/ndt/v1.2.0/>`__
     - 2026-05-06
     - Cleanup, documentation overhaul, removal of ``DictPaths``.
   * - `1.1.0 <https://biface.github.io/ndt/v1.1.0/>`__
     - 2026-04-05
     - Serialisation (JSON + pickle), ``from_dict`` classmethod, Python 3.13
       support, Python 3.9 dropped.
   * - `1.0.0 <https://biface.github.io/ndt/v1.0.0/>`__
     - 2026-01-31
     - First stable release. Public API, exception hierarchy, PyPI publication.
   * - `0.9.0 <https://biface.github.io/ndt/v0.9.0/>`__
     - 2025-11-04
     - ``default_setup`` generalised, Python versions earlier than 3.9 dropped.
       Gathers the earlier 0.x work. Documentation kept until October 2027.

Full history
------------

The complete change log is maintained in :doc:`history`.

Active deprecations are listed in :doc:`deprecations`.
