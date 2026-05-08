Exceptions
==========

.. automodule:: ndict_tools.exception
   :no-members:
   :no-undoc-members:

All exceptions inherit from :class:`~ndict_tools.StackedDictionaryError`,
which itself inherits from :class:`Exception`. Specialisations also inherit
from the matching standard library exception so that standard ``except``
clauses continue to work:

.. code-block:: python

    try:
        nd["missing_key"]
    except KeyError:        # catches StackedKeyError too
        ...
    except StackedKeyError: # catches only StackedKeyError
        ...

Base exception
--------------

.. autoexception:: ndict_tools.StackedDictionaryError
   :members:
   :show-inheritance:

Specialisations
---------------

.. autoexception:: ndict_tools.StackedKeyError
   :members:
   :show-inheritance:

.. autoexception:: ndict_tools.StackedAttributeError
   :members:
   :show-inheritance:

.. autoexception:: ndict_tools.StackedTypeError
   :members:
   :show-inheritance:

.. autoexception:: ndict_tools.StackedValueError
   :members:
   :show-inheritance:

.. autoexception:: ndict_tools.StackedIndexError
   :members:
   :show-inheritance:

.. autoexception:: ndict_tools.NestedDictionaryException
   :members:
   :show-inheritance:
