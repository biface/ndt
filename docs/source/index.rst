.. Nested Dictionary Tools documentation master file, created by
   sphinx-quickstart on Fri Jul 26 17:59:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Overview
========

**Nested Dictionary Tools (NDT)** is a simple toolbox to manage nested dictionaries.

Nested dictionaries are present in Python without necessarily being managed. For example,
to access the keys of nested dictionaries in Python, you have to browse the dictionaries.
In contrast, this toolbox provides access to all keys. So you can check that a key is
present in all the keys, or count its occurrences.


.. toctree::
   install
   usage
   :numbered:
   :maxdepth: 2
   :caption: Content
   :name: maintoc


Concept
-------

Nested dictionaries are allowed in Python since the value of a key can be a dict. Nevertheless,
keys in a nested dictionary are not manageable from the main dictionary. A NestedDictionary is one of
the implementation to do so.

Behavior
^^^^^^^^

A NestedDictionary inherits from defaultdict. It has the same properties, except that items, keys
and values can be de-nested.

Classical dictionary
""""""""""""""""""""

.. code-block:: console

   $ d = dict([('first', 1), ('third', 3)], second={'first': 1, 'second':2})
   $ d
   {'first': 1, 'third': 3, 'second': {'first': 1, 'second': 2}}
   $ d.keys()
   dict_keys(['first', 'third', 'second'])

Nested dictionary
"""""""""""""""""

.. code-block:: console

   $ nd = NestedDictionary([('first', 1), ('third', 3)], second={'first': 1, 'second':2})
   $ nd
   NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {'first': 1, 'third': 3, 'second': NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {'first': 1, 'second': 2})})
   $ nd.keys()
   dict_keys(['first', 'third', 'second'])
   $ nd.key_list('second')
   [('second', 'first'), ('second', 'second')]

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
