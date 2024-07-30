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
    :no-index:
.. autoclass:: NestedDictionary

    .. autoattribute:: indent
    .. autoattribute:: default_factory
    .. automethod:: __str__()
    .. automethod:: update()
    .. automethod:: occurrences()
    .. automethod:: is_key()
    .. automethod:: key_list()
    .. automethod:: items_list()
    .. automethod:: to_dict()

