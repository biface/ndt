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

.. versionchanged:: 0.7.0
    Moved the update method exclusively to the ``_StackedDict`` class to standardize updates for future subclasses.

.. versionadded:: 0.6.1
    Added path and tree-like management functions. These functions are still in the early testing stages and are not expected to be fully integrated until the stable version 1.0.0.

.. versionadded:: 0.6.0
    Introduced nested keys with Python lists: ``sd[[1, 2, 3]] == sd[1][2][3]``.
    Note the use of double brackets ``[[...]]`` to manage the key list.
