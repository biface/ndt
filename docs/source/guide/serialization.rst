Part 3 — Serialisation
=======================

**ndict-tools** provides built-in serialisation to JSON and pickle for
all three public classes. Both formats support a full round-trip: the
reconstructed object is an instance of the same class with the same
``default_setup``.

.. warning::

   Pickle files are unsafe when loaded from untrusted sources.
   Only unpickle files you created yourself or received from a trusted
   party. A :class:`UserWarning` is always emitted on both ``to_pickle``
   and ``from_pickle`` to remind you of this.


JSON round-trip
----------------

Use :meth:`~ndict_tools.NestedDictionary.to_json` to write and
:meth:`~ndict_tools.NestedDictionary.from_json` to reconstruct:

.. code-block:: python

   from ndict_tools import NestedDictionary

   nd = NestedDictionary({"project": {"name": "ndict-tools", "stars": 42}})
   nd.to_json("/tmp/nd.json", indent=2)

   nd2 = NestedDictionary.from_json(
       "/tmp/nd.json",
       default_setup={"indent": 2, "default_factory": NestedDictionary},
   )
   nd2[["project", "name"]]   # 'ndict-tools'

The ``default_setup`` parameter in ``from_json`` is required for the same
reason as in :meth:`~ndict_tools.NestedDictionary.from_dict` — the JSON
file carries no information about the target class behaviour.


Non-string keys
~~~~~~~~~~~~~~~~

JSON mandates string keys. **ndict-tools** encodes all non-string keys
automatically using a reversible convention (DD-021) — you do not need to
handle this yourself:

.. code-block:: python

   nd = NestedDictionary({42: {"label": "answer"}, "name": "demo"})
   nd.to_json("/tmp/mixed.json")

   nd2 = NestedDictionary.from_json(
       "/tmp/mixed.json",
       default_setup={"indent": 0, "default_factory": NestedDictionary},
   )
   nd2[42]["label"]   # 'answer' — integer key restored correctly

Supported key types for JSON encoding: :class:`str`, :class:`int`,
:class:`float`, :class:`bool`, flat :class:`tuple`, flat :class:`frozenset`.


Pickle round-trip
------------------

Use :meth:`~ndict_tools.NestedDictionary.to_pickle` and
:meth:`~ndict_tools.NestedDictionary.from_pickle`:

.. code-block:: python

   from ndict_tools import NestedDictionary

   nd = NestedDictionary({"a": {1: "one", 2: "two"}})
   nd.to_pickle("/tmp/nd.pkl")
   # Writes /tmp/nd.pkl and /tmp/nd.pkl.sha256 (SHA-256 integrity sidecar)

   nd2 = NestedDictionary.from_pickle("/tmp/nd.pkl")
   # Verifies the SHA-256 sidecar before unpickling
   nd2["a"][1]   # 'one'

A ``.sha256`` sidecar file is written alongside the pickle file. By
default, ``from_pickle`` reads the sidecar and raises
:class:`~ndict_tools.StackedValueError` if the digest does not match or
the sidecar is absent. To skip the check:

.. code-block:: python

   nd2 = NestedDictionary.from_pickle("/tmp/nd.pkl", verify=False)

Pickle preserves the full object state, including ``default_setup`` and
``default_factory``, without any extra arguments on load.


JSON vs pickle — when to use which
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - JSON
     - Pickle
   * - **Key types**
     - ``str``, ``int``, ``float``, ``bool``, flat ``tuple``, flat ``frozenset``
     - Any hashable type
   * - **Values**
     - JSON-serialisable only (str, int, float, bool, list, dict, None)
     - Any picklable Python object
   * - **Interoperability**
     - Any language, any tool
     - Python only
   * - **Readability**
     - Human-readable text
     - Binary
   * - **Security**
     - Safe to load from untrusted sources
     - **Never** load from untrusted sources
   * - **Integrity check**
     - None built-in
     - SHA-256 sidecar (automatic)
   * - **``default_setup`` on load**
     - Required
     - Preserved automatically

**Rule of thumb:** use JSON for configuration files, data exchange, and
anything that crosses a trust boundary. Use pickle for local persistence
of complex objects (non-string keys, custom value types) within a trusted
environment.
