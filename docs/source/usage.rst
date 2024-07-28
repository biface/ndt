Usage
=====

Instance
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

.. module:: ndict_tools.core
.. autoclass:: NestedDictionary

    .. autoattribute:: indent
    .. autoattribute:: default_factory
    .. automethod:: __str__()
    .. automethod:: update()
    .. automethod:: occurrences()
    .. automethod:: is_key()
    .. automethod:: key_list()
    .. autoexception:: ndict_tools.exception.StackedKeyError
    .. automethod:: items_list()
    .. automethod:: to_dict()

