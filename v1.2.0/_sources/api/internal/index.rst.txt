Internal API Reference
======================

.. warning::

   This section documents **private implementation classes** (underscore-prefixed).
   They are intended for developers who extend or maintain the package.

   Internal APIs are **subject to change without notice** between any two
   releases and are not covered by the project's stability guarantees.
   Do not rely on them in production code.

.. toctree::
   :maxdepth: 1
   :hidden:

   tools
   serialize

``tools`` module
----------------

.. autosummary::

   ndict_tools.tools.compare_dict
   ndict_tools.tools.unpack_items
   ndict_tools.tools._HKey
   ndict_tools.tools._StackedDict
   ndict_tools.tools._Paths
   ndict_tools.tools._CPaths

``serialize`` module
--------------------

.. autosummary::

   ndict_tools.serialize._encode_key
   ndict_tools.serialize._decode_key
   ndict_tools.serialize.NestedDictionaryEncoder
   ndict_tools.serialize._make_decoder_hook
   ndict_tools.serialize._pickle_dump
   ndict_tools.serialize._pickle_load
