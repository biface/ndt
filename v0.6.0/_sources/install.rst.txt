Installation
============

Command line
------------

This package is available on Python Package Index PyPI_ and easy to isntall. Simply use PIP as below

.. code-block:: console

    (.venv) $ pip3 install ndict-tools

or use your IDE usual interface to install this package from PyPI_

.. _PyPI: https://pypi.org/project/ndict-tools/

From github
-----------

This package is also released on `GitHub <https://github.com/biface/ndt>`_. You can collect from
the `release directory <https://github.com/biface/ndt/releases>`_ the desired version and unpack
it in your project.

Versions
--------

.. versionadded:: 0.6.0
    Introducing nested keys with python lists : ``sd[[1, 2, 3]] eq sd[1][2][3]``
    Pay particular attention to the use of double brackets ``[[...]]`` to manage the key list.