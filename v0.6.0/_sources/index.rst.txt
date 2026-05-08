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
   api
   :numbered:
   :maxdepth: 2
   :caption: Content
   :name: maintoc


Concept
-------

Nested dictionaries are allowed in Python since the value of a key can be a dict. Nevertheless,
keys in a nested dictionary are not manageable from the main dictionary. A NestedDictionary is one of
the implementation to do so.

Differences (from dict)
^^^^^^^^^^^^^^^^^^^^^^^

A NestedDictionary inherits from defaultdict. It has the same properties, except that items, keys
and values can be de-nested.

Classical dictionary
""""""""""""""""""""

.. code-block:: console

   $ d = dict([('first', 1), ('third', 3)], second={'first': 1, 'second':2})
   $ d
   {'first': 1, 'third': 3, 'second': {'first': 1, 'second': 2}}

   d= {dict:3}
      'first' = {int} 1
      'third' = {int} 3
      'second' = {dict:2}
         'first' = {int} 1
         'second' = {int} 2
         __len__ = {int} 2
      __len__ = {int} 3

   d's keys are :
   ['first', 'third', 'second']

   $ d.keys()
   dict_keys(['first', 'third', 'second'])

Nested dictionary
"""""""""""""""""

.. code-block:: console

   $ from ndict import NestedDictionary
   $ nd = NestedDictionary([('first', 1), ('third', 3)], second={'first': 1, 'second':2})
   $ nd
   NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {'first': 1, 'third': 3, 'second': NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {'first': 1, 'second': 2})})

   nd = {NestedDictionary:3}
      default_factory = {type} <class 'ndict_tools.core.NestedDictionary>
      indent = {int} 0
      'first' = {int} 1
      'third' = {int} 3
      'second' = {NestedDictionary:2}
         default_factory = {type} <class 'ndict_tools.core.NestedDictionary>
         indent = {int} 0
         'first' = {int} 1
         'second' = {int} 2
         __len__ = {int} 2
      __len__ = {int} 3

   nd's keys are
   [('first',), ('third',), ('second', 'first'), ('second', 'second')]

   $ nd.keys() -- keep the same behavior as classical dictionary --
   dict_keys(['first', 'third', 'second'])
   $ nd.key_list('second')
   [('second', 'first'), ('second', 'second')]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
