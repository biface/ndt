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

Version History
---------------

.. versionremoved:: 1.2.0

    - The ``NestedDictionary`` class no longer uses class-specific definition attributes for initialization.
      Instead, it now systematically uses class-specific attribute initialization.

.. deprecated:: 1.0.0

    - The use of ``NestedDictionary`` class-specific parameters (``indent`` and ``strict`` keys in the ``__init__`` method)
      at instance initialization is now deprecated.

.. versionadded:: 1.0.0

    - Added a JSON converter for nested dictionaries that preserves non-string key types (e.g., integers, tuples).
      This ensures compatibility with complex data structures during serialization and deserialization.

.. versionchanged:: 0.9.0

    - The ``DictPaths`` class has been renamed to the private class ``_Paths``.
      This technical class is now used internally to manage sets of paths.
    - The public ``DictPaths`` class now inherits from ``_CPaths`` and provides a way to manually build search paths.

.. versionadded:: 0.9.0

    - Introduced the ``_HKey`` class, which represents a hierarchical tree of keys for generating paths.
    - Introduced the ``_CPaths`` class, a compact and user-friendly way to manually create path research structures.
      This class will serve as the foundation for a future public class.
    - Added support for encoding and decoding when exporting to or importing from pickle and JSON file formats.

.. versionadded:: 0.8.0

    - Added generalized handling of specific attributes for child classes of ``_StackedDict``.
      Further details will be provided in a dedicated section for developers.
    - Introduced ``DictPaths`` as a collection of paths within ``_StackedDict``.

.. versionchanged:: 0.7.0

    - The ``update`` method is now exclusive to the ``_StackedDict`` class to ensure standardized updates for future subclasses.

.. versionadded:: 0.6.1

    - Added path and tree-like management functions.
      These features are currently in early testing and will be fully integrated in the stable 1.0.0 release.

.. versionadded:: 0.6.0

    - Introduced support for nested keys using Python lists: ``sd[[1, 2, 3]] == sd[1][2][3]``.
    - Note: Double brackets ``[[...]]`` are used to denote a list of keys.

.. important::

    - Versions prior to 0.6.0 are no longer supported.
    - Version 1.0.0 will be the first stable release, and public class names will not change after this version.

