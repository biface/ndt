Usage
=====

Principle
---------

The core concept is straightforward: just as a dictionary can contain another dictionary as a value, a
``NestedDictionary`` naturally extends this idea. In a ``NestedDictionary``, each value that is itself a dictionary
must also be a ``NestedDictionary``. This recursive structure allows for seamless nesting of dictionaries within
dictionaries.

Unlike conventional dictionaries, nested keys in a ``NestedDictionary`` are exposed as tuples. This representation
allows for easy access and manipulation of hierarchical data while maintaining compatibility with standard dictionary
operations.

.. code-block:: console

    $ a = NestedDictionary({'first': 1,
                            'second': {'1': "2:1", '2': "2:2", '3': "3:2"},
                            'third': 3,
                            'fourth': 4})

    a's keys are:
    [('first',), ('second', '1'), ('second', '2'), ('second', '3'), ('third',), ('fourth',)]

    $ a['second']['1'] = "2:1"

Understanding Paths in Nested Dictionaries
------------------------------------------

In the context of nested dictionaries, a **path** is a sequence of keys that navigates through the hierarchical
structure to access a specific value or sub-dictionary. Think of it as a trail of keys that leads you from the
outermost dictionary to a particular point within the nested structure.

For example, consider the following nested dictionary:

.. code-block:: python

    data = {
        'a': {
            'b': {
                'c': 1
            },
            'd': 2
        },
        'e': 3
    }

In this dictionary:

- The path ``['a', 'b', 'c']`` leads to the value ``1``.
- The path ``['a', 'b']`` leads to the dictionary ``{'c': 1}``.
- The path ``['a', 'd']`` leads to the value ``2``.
- The path ``['e']`` leads to the value ``3``.

By representing these paths as lists, we can easily describe and manipulate the hierarchical relationships within
the dictionary. This concept is particularly useful when working with complex nested structures, as it provides a clear
and concise way to reference specific elements.

Justification for Using Lists to Describe Paths
-----------------------------------------------

While standard dictionaries in Python use immutable types such as strings, numbers, or tuples as keys, representing
hierarchical paths with lists offers several advantages in the context of nested dictionaries:

1. **Sequential Representation**: Lists naturally represent sequences, making them ideal for capturing the order
   of keys that lead to a specific value in a nested structure. This sequential nature aligns well with the concept
   of navigating through layers of dictionaries.

2. **Flexibility**: Lists are mutable, allowing for dynamic manipulation of paths. This flexibility is beneficial when
   working with complex or evolving data structures, where paths may need to be extended, modified, or truncated.

3. **Readability**: Using lists to represent paths enhances code readability. It provides a clear and intuitive way to
   understand the hierarchical relationships within the data, making the code easier to maintain and debug.

4. **Compatibility with Recursive Operations**: Lists are well-suited for recursive operations, which are common when
   traversing nested dictionaries. They can be easily passed to and modified within recursive functions, simplifying
   the implementation of algorithms that operate on hierarchical data.

5. **Consistency with Existing Tools**: Many existing tools and libraries that deal with hierarchical data structures,
   such as JSON or XML parsers, use lists or similar structures to represent paths. By adopting this convention,
   we maintain consistency with established practices.

Introducing DictPaths
---------------------

To manage and access these paths efficiently, we provide the ``DictPaths`` class. This class offers a view object that
provides a dictionary-like interface for accessing hierarchical keys as lists. Similar to ``dict_keys``, but tailored
for hierarchical paths in a ``_StackedDict``, ``DictPaths`` allows you to:

- **Iterate** over all hierarchical paths in the ``_StackedDict`` as lists.
- **Check** if a specific hierarchical path exists within the ``_StackedDict``.
- **Retrieve** the number of hierarchical paths present in the ``_StackedDict``.

By using ``DictPaths``, you can easily navigate and manipulate complex nested dictionary structures, making your code
more readable and maintainable.


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
    .. autoattribute:: default_factory
    .. automethod:: update()
    .. automethod:: occurrences()
    .. automethod:: is_key()
    .. automethod:: key_list()
    .. automethod:: items_list()
    .. automethod:: to_dict()
        :no-index:


.. _defaultdict: https://docs.python.org/3/library/collections.html#collections.defaultdict