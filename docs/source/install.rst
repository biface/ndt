Installation
============

Command Line
------------

This package is available on the Python Package Index (PyPI_) and is easy to install. Simply use pip as shown below:

.. code-block:: console

    (.venv) $ pip3 install ndict-tools

Alternatively, use your IDE's interface to install this package from PyPI_.

.. _PyPI: https://pypi.org/project/ndict-tools/

From GitHub
-----------

This package is also available on `GitHub <https://github.com/biface/ndt>`_. You can download the desired version from the `release directory <https://github.com/biface/ndt/releases>`_ and unpack it into your project.

Versioning Policy
-----------------

This project follows a specific versioning strategy to distinguish between experimental and stable releases:

**Odd Minor Versions (Experimental)**
    Odd minor versions (e.g., 0.1.x, 0.3.x, 0.5.x) are experimental releases intended for testing new features
    and improvements. These versions may receive corrective patches in the format ``n.p.1`` or ``n.p.1-1``.
    Experimental versions are transitional and are typically deprecated once the following stable version is released.

**Even Minor Versions (Stable)**
    Even minor versions (e.g., 0.2.x, 0.4.x, 0.6.x) are stabilized releases that follow experimental versions.
    These versions are thoroughly tested and recommended for production use.

.. note::
    Experimental odd minor versions have a short lifecycle and will be phased out rapidly after the next stable
    even minor version is released. Users are encouraged to upgrade to stable versions for long-term support.

Version History
---------------

This project follows semantic versioning with named releases that reflect their key features:

.. versionadded:: 1.2.0 "Persistence (Serializer)" - *Stable*

    - Stabilized JSON serialization with comprehensive file save/load testing.
    - Added a JSON converter for nested dictionaries that preserves non-string key types (e.g., integers, tuples).
      This ensures compatibility with complex data structures during serialization and deserialization.

.. versionremoved:: 1.2.0

    - The ``NestedDictionary`` class no longer uses class-specific definition attributes for initialization.
      Instead, it now systematically uses class-specific attribute initialization.

.. versionadded:: 1.1.0 "JSON Bridge (Encoder)" - *Experimental*

    - Introduced JSON encoding and decoding capabilities for nested dictionaries.
    - This experimental version lays the groundwork for full persistence features in version 1.2.0.

.. versionadded:: 1.0.0 "Path Manager (Compass)" - *Stable*

    - Stabilized key tree structures and path traversal mechanisms introduced in version 0.9.0.
    - Production-ready path management with robust navigation features.

.. deprecated:: 1.0.0

    - The use of ``NestedDictionary`` class-specific parameters (``indent`` and ``strict`` keys in the ``__init__`` method)
      at instance initialization is now deprecated.

.. versionchanged:: 0.9.0 "Tree Builder (Seedling)" - *Experimental*

    - The ``DictPaths`` class has been renamed to the private class ``_Paths``.
      This technical class is now used internally to manage sets of paths.
    - The public ``DictPaths`` class now inherits from ``_CPaths`` and provides a way to manually build search paths.

.. versionadded:: 0.9.0

    - Introduced the ``_HKey`` class, which represents a hierarchical tree of keys for generating paths.
    - Introduced the ``_CPaths`` class, a compact and user-friendly way to manually create path research structures.
      This class will serve as the foundation for a future public class.
    - Added support for encoding and decoding when exporting to or importing from pickle and JSON file formats.
    - **Note**: Path traversal mechanisms are experimental and will be stabilized in version 1.0.0.

.. versionadded:: 0.8.0 "Core Refactor (Blueprint)"

    - Stabilized architecture with generalized parameter handling through ``default_setup``.
    - Added generalized handling of specific attributes for child classes of ``_StackedDict``.
      Further details will be provided in a dedicated section for developers.
    - Introduced ``DictPaths`` as a collection of paths within ``_StackedDict``.

.. versionchanged:: 0.7.0 "Pathfinder (Trailblazer)" - *Experimental*

    - The ``update`` method is now exclusive to the ``_StackedDict`` class to ensure standardized updates for future subclasses.
    - Introduced unified path-based navigation treating paths as lists of keys.

.. versionadded:: 0.6.1

    - Added path and tree-like management functions.
      These features are currently in early testing and will be fully integrated in the stable 1.0.0 release.

.. versionadded:: 0.6.0 "Hierarchy (Roots)"

    - Introduced support for nested keys using Python lists: ``sd[[1, 2, 3]] == sd[1][2][3]``.
    - Note: Double brackets ``[[...]]`` are used to denote a list of keys.
    - This version establishes the foundation for hierarchical key structures.

.. important::

    - Versions prior to 0.6.0 are no longer supported.
    - Version 1.0.0 is the first production-stable release with finalized public API names.