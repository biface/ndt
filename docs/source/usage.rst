Usage
=====

Principle
---------
The principle is quite simple, just as a dictionary can be the value of a dictionary key. If it is a dictionary, a
NestedDictionary is necessarily the value of the key of a NestedDictionary, and so on.

However, unlike a conventional dictionary, nested keys will be exposed as tuples. Even so, they can still be used
as conventional keys.

.. code-block:: console

    $ a = NestedDictionary({'first': 1,
                            'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                            'third': 3,
                            'fourth': 4})

    a's keys are :
    [('first',), ('second', '1'), ('second', '2'), ('second', '3'), ('third',), ('fourth',)]

    $ a['second']['1'] = "2:1"

Behavior
--------

Nested dictionaries inherit from defaultdict_. The default_factory attribute characterizes the behavior of this class:

If the nested dictionary is to behave strictly like a dictionary, then the default_factory attribute is set to None.
If you request the value of a key that doesn't exist, you'll get a KeyError. The configuration parameter is
``strict=True``

.. code-block:: python

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({'first': 1,
                               'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                               'third': 3,
                               'fourth': 4},
                               strict=True)
    nd.default_factory

    >>> nd['fifth']
    Traceback (most recent call last):
      File "/snap/pycharm-professional/401/plugins/python/helpers/pydev/pydevconsole.py", line 364, in runcode
        coro = func()
      File "<input>", line 1, in <module>
    KeyError: 'fifth'

If the nested dictionary is to have flexible behavior, then the default_factory attribute is set to NestedDictionary.
If you request a key that doesn't exist, a NestedDictionary instance will be created accordingly and returned. The
configuration parameter is ``strict=False`` or **no parameter**

.. code-block:: python

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({'first': 1,
                               'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                               'third': 3,
                               'fourth': 4},
                               strict=False)
    >>> nd.default_factory
    <class 'ndict_tools.core.NestedDictionary'>
    >>> nd['fifth']
    NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {})

And with **no parameter**

.. code-block:: python

    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({'first': 1,
                               'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                               'third': 3,
                               'fourth': 4})
    >>> nd.default_factory
    <class 'ndict_tools.core.NestedDictionary'>
    >>> nd['fifth']
    NestedDictionary(<class 'ndict_tools.core.NestedDictionary'>, {})


Examples
--------

.. code-block:: console

    $ a = NestedDictionary({'first': 1,
                            'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                            'third': 3,
                            'fourth': 4})
    $ b = NestedDictionary(zip(['first', 'second', 'third', 'fourth'],
                               [1, {'1': "2:1", '2': "2:2", '3': "3:2"}, 3, 4]))
    $ c = NestedDictionary([('first', 1),
                            ('second', {'1': "2:1", '2': "2:2", '3': "3:2"}),
                            ('third', 3),
                            ('fourth', 4)])
    $ d = NestedDictionary([('third', 3),
                            ('first', 1),
                            ('second', {'1': "2:1", '2': "2:2", '3': "3:2"}),
                            ('fourth', 4)])
    $ e = NestedDictionary([('first', 1), ('fourth', 4)],
                           third = 3,
                           second = {'1': "2:1", '2': "2:2", '3': "3:2"})

    a == b == c == d == e


Class attributes and methods
----------------------------

.. module:: ndict_tools
.. autoclass:: NestedDictionary

    .. autoattribute:: indent
        :no-index:
    .. autoattribute:: default_factory
        :no-index:
    .. automethod:: __str__()
    .. automethod:: update()
    .. automethod:: occurrences()
        :no-index:
    .. automethod:: is_key()
        :no-index:
    .. automethod:: key_list()
        :no-index:
    .. automethod:: items_list()
        :no-index:
    .. automethod:: to_dict()
        :no-index:


.. _defaultdict: https://docs.python.org/3/library/collections.html#collections.defaultdict