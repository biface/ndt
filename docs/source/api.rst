Package reference
=================

For greater convenience, the modules remain hidden inside the package. These modules are exposed for development
purposes only.

.. module::ndict_tools
   :no-index:

Exceptions
----------

.. automodule:: ndict_tools.exception
    :private-members: StackedDictionaryError, StackedKeyError, StackedAttributeError
.. autoexception:: NestedDictionaryException

Tools
-----

.. automodule:: ndict_tools.tools
.. autofunction:: unpack_items()
.. autofunction:: from_dict()
.. autoclass:: _StackedDict

    .. autoattribute:: indent
    .. autoattribute:: default_factory
    .. automethod:: __str__()
    .. automethod:: unpacked_items()
    .. automethod:: unpacked_keys()
    .. automethod:: unpacked_values()
    .. automethod:: to_dict()
    .. automethod:: update()
    .. automethod:: occurrences()
    .. automethod:: is_key()
    .. automethod:: key_list()
    .. automethod:: items_list()
    .. automethod:: to_dict()

Core
----
.. automodule:: ndict_tools.core
.. autoclass:: NestedDictionary

    .. automethod:: __str__()
    .. automethod:: update()
