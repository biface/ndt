``serialize`` — Serialisation Infrastructure
=============================================

.. warning::

   **Internal module — subject to change without notice.**
   Do not import these functions directly. Use the public serialisation
   methods on :class:`~ndict_tools.NestedDictionary` instead
   (``to_json``, ``from_json``, ``to_pickle``, ``from_pickle``).

.. automodule:: ndict_tools.serialize
   :no-members:
   :no-undoc-members:

Key encoding (DD-021)
----------------------

.. autofunction:: ndict_tools.serialize._encode_key

.. autofunction:: ndict_tools.serialize._decode_key

JSON encoder
------------

.. autoclass:: ndict_tools.serialize.NestedDictionaryEncoder
   :members:
   :show-inheritance:

.. autofunction:: ndict_tools.serialize._make_decoder_hook

Pickle helpers
--------------

.. autofunction:: ndict_tools.serialize._pickle_dump

.. autofunction:: ndict_tools.serialize._pickle_load
