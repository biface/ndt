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
.. autofunction:: compare_dict()
.. autofunction:: unpack_items()
.. autofunction:: from_dict()
.. autoclass:: _HKey

    .. automethod:: build_forest()
    .. automethod:: add_child()
    .. automethod:: add_children()
    .. automethod:: get_child()
    .. automethod:: get_child_keys()
    .. automethod:: has_children()
    .. automethod:: is_leaf()
    .. automethod:: get_path()
    .. automethod:: find_by_path()
    .. automethod:: get_all_paths()
    .. automethod:: get_descendants()
    .. automethod:: get_depth()
    .. automethod:: get_max_depth()
    .. automethod:: iter_children()
    .. automethod:: iter_leaves()
    .. automethod:: dfs_preorder()
    .. automethod:: dfs_postorder()
    .. automethod:: bfs()
    .. automethod:: dfs_find()
    .. automethod:: bfs_find()
    .. automethod:: find_all()
    .. automethod:: find_by_key()
    .. automethod:: iter_by_level()
    .. automethod:: get_nodes_at_depth()
    .. automethod:: filter_paths()
    .. automethod:: map_nodes()
    .. automethod:: prune()
    .. automethod:: get_statistics()
    .. automethod:: has_cycles()
    .. automethod:: is_dag()
    .. automethod:: is_valid_tree()
    .. automethod:: check_parent_consistency()
    .. automethod:: is_complete_tree()
    .. automethod:: is_perfect_tree()
    .. automethod:: is_balanced()
    .. automethod:: get_balance_factor()
    .. automethod:: count_nodes_by_degree()
    .. automethod:: is_binary_tree()
    .. automethod:: is_full_tree()
    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: __repr__


.. autoclass:: _StackedDict

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
    .. automethod:: paths()
    .. automethod:: dfs()
    .. automethod:: bfs()
    .. automethod:: is_balanced()
    .. automethod:: height()
    .. automethod:: size()
    .. automethod:: ancestors()
    .. automethod:: leaves()
    .. automethod:: __str__()
    .. automethod:: __copy__()
    .. automethod:: __deepcopy__()
    .. automethod:: __setitem__()
    .. automethod:: __getitem__()
    .. automethod:: __delitem__()


.. autoclass:: _Paths

    .. automethod:: get_children()
    .. automethod:: has_children()
    .. automethod:: get_subtree_paths()
    .. automethod:: filter_paths()
    .. automethod:: get_depth()
    .. automethod:: get_leaf_paths()
    .. automethod:: to_compact()
    .. automethod:: __iter__()
    .. automethod:: __len__()
    .. automethod:: __eq__()
    .. automethod:: __ne__()
    .. automethod:: __contains__()
    .. automethod:: __repr__()
    .. automethod:: _ensure_hkey()

.. autoclass:: _CPaths
    :show-inheritance:

    .. autoproperty:: structure
    .. important::
        The ``structure`` property has a setter in order to manually store compacted paths.
    .. automethod:: expand_structure
    .. automethod:: expand
    .. automethod:: is_covering
    .. automethod:: coverage
    .. automethod:: missing_paths
    .. automethod:: uncovered_paths
    .. automethod:: __repr__
    .. automethod:: __str__
    .. automethod:: _ensure_structure
    .. automethod:: _build_compact_structure
    .. automethod:: _validate_structure
    .. automethod:: _compare_path_sets

.. autoclass:: DictPaths

    .. automethod:: _iterate_paths()
    .. automethod:: __iter__()
    .. automethod:: __len__()
    .. automethod:: __contains__()
    .. automethod:: __repr__()

Core
----
.. automodule:: ndict_tools.core
.. autoclass:: PathsView
    :show-inheritance:

    .. automethod:: to_compact

.. autoclass:: CompactPathsView
    :show-inheritance:

    .. automethod:: to_paths

.. autoclass:: NestedDictionary
    :show-inheritance:

    .. automethod:: paths
    .. automethod:: compact_paths

.. autoclass:: StrictNestedDictionary
    :show-inheritance:
.. autoclass:: SmoothNestedDictionary
    :show-inheritance:
