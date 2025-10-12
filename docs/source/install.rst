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

Versions
--------

.. versionremoved:: 1.2.0

    - Removal of ``NestedDictionary`` class-specific definition attributes for systematic use of class-specific attribute initialization.

.. deprecated:: 1.0.0

    - Use of the ``NestedDictionary`` class's specific parameterization at instance initialization (``indent`` and ``strict`` keys of the ``__init__`` method)

.. versionchanged:: 1.0.0

    - The ``DictSearch`` will become the public view of paths and will be renamed as ``DictPaths`` as the public view of paths in nested dictionaries.

.. versionchanged:: 0.9.0

   - Renaming ``DictPaths`` as a private class ``_Paths`` which will be a technical class to manage set of paths.

.. versionadded:: 0.9.0

    - Introduction of ``_HKey`` class as the hierarchical tree of keys used to production paths.
    - Introduction of ``DictSearch`` class in order to give easy access to ``_Paths`` set of hierarchical keys.
    - Introduction of encoding for exports to or imports from pickle and JSON file formats.

.. versionadded:: 0.8.0

    - Introduction of generalized handling of specific attributes of ``_StackedDict`` child classes. This addition will be explained later in a section for developers.

.. versionchanged:: 0.7.0

    - Moved the update method exclusively to the ``_StackedDict`` class to standardize updates for future subclasses.

.. versionadded:: 0.6.1

    - Added path and tree-like management functions. These functions are still in the early testing stages and are not expected to be fully integrated until the stable version 1.0.0.

.. versionadded:: 0.6.0

    - Introduced nested keys with Python lists: ``sd[[1, 2, 3]] == sd[1][2][3]``.
    - Note the use of double brackets ``[[...]]`` to manage the key list.

.. important::

   - All versions prior to version 0.6.0 are no longer supported.
   - version 1.0.0 will be the first stable version.
