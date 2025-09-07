Package reference
=================

For greater convenience, the modules remain hidden inside the package. These modules are exposed for development
purposes only.

.. module::ndict_tools
   :no-index:

Exceptions
----------

.. automodule:: ndict_tools.exception
.. autoexception:: StackedAttributeError
.. autoexception:: StackedDictionaryError
.. autoexception:: StackedIndexError
.. autoexception:: StackedKeyError
.. autoexception:: StackedTypeError
.. autoexception:: StackedValueError
.. autoexception:: NestedDictionaryException

Tools
-----

.. automodule:: ndict_tools.tools
.. autofunction:: unpack_items()
.. autofunction:: from_dict()
.. autoclass:: _StackedDict

    .. autoattribute:: indent
    .. autoattribute:: default_factory
    .. autoattribute:: default_setup
    .. automethod:: __str__()
    .. automethod:: __copy__()
    .. automethod:: __deepcopy__()
    .. automethod:: __setitem__()
    .. automethod:: __getitem__()
    .. automethod:: __delitem__()
    .. automethod:: unpacked_items()
    .. automethod:: unpacked_keys()
    .. automethod:: unpacked_values()
    .. automethod:: pop()
    .. automethod:: popitem()
    .. important::

        If a nested dictionary is emptied after using popitem() and is nested in another nested dictionary. It will
        appear as an empty dictionary.

        .. code-block:: console

                $ sd = _StackedDict(indent=2, default=None)

                $ sd["x"] = "value3"
                $ sd["a"] = {"b": {"c": "value1"}}

                $ sd.popitem()
                (['a', 'b', 'c'], 'value1')

                $ sd
                _StackedDict(None, {'x': 'value3', 'a': {'b': {}}})
                
    .. automethod:: is_key()
    .. automethod:: occurrences()
    .. automethod:: key_list()
    .. automethod:: items_list()
    .. automethod:: update()
    .. automethod:: to_dict()
    .. automethod:: dict_paths()
    .. automethod:: dfs()
    .. automethod:: bfs()
    .. automethod:: is_balanced()
    .. automethod:: height()
    .. automethod:: size()
    .. automethod:: ancestors()
    .. automethod:: leaves()

.. autoclass:: DictPaths

    .. automethod:: _iterate_paths()
    .. automethod:: __iter__()
    .. automethod:: __len__()
    .. automethod:: __eq__()
    .. automethod:: __ne__()
    .. automethod:: __contains__()
    .. automethod:: __repr__()

Core
----
.. automodule:: ndict_tools.core
.. autoclass:: NestedDictionary

    .. automethod:: __str__()

.. autoclass:: StrictNestedDictionary

.. autoclass:: SmoothNestedDictionary
