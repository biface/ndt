Part 4 — Serialisation
======================

A nested dictionary can be saved to a file and read back, in JSON or with
pickle. This part saves the heating schedule of the house, which maps an
hour of the day to a temperature, and explains what each format keeps.

.. testsetup:: serialization

   import os
   import tempfile
   import warnings

   _previous_dir = os.getcwd()
   os.chdir(tempfile.mkdtemp())
   warnings.simplefilter("ignore", UserWarning)

.. testcleanup:: serialization

   os.chdir(_previous_dir)
   warnings.resetwarnings()

The thermostat of each room switches at given hours. The hours are
integers, not strings:

.. doctest:: serialization

   >>> from ndict_tools import NestedDictionary
   >>> schedule = NestedDictionary({
   ...     "living room": {"thermostat": {7: 20.5, 22: 17.0}},
   ...     "bedroom": {"thermostat": {6: 19.0, 21: 16.5}},
   ... })


JSON
----

:meth:`~ndict_tools.NestedDictionary.to_json` writes the dictionary to a
JSON file. ``indent`` makes the file easier to read:

.. doctest:: serialization

   >>> schedule.to_json("schedule.json", indent=2)
   >>> from pathlib import Path
   >>> print(Path("schedule.json").read_text())
   {
     "living room": {
       "thermostat": {
         "[7]": 20.5,
         "[22]": 17.0
       }
     },
     "bedroom": {
       "thermostat": {
         "[6]": 19.0,
         "[21]": 16.5
       }
     }
   }

JSON only allows string keys, so the hours are written as ``"[7]"``,
``"[22]"`` and so on. :meth:`~ndict_tools.NestedDictionary.from_json` reads
them back as integers:

.. doctest:: serialization

   >>> restored = NestedDictionary.from_json("schedule.json")
   >>> restored[["living room", "thermostat", 7]]
   20.5
   >>> restored == schedule
   True

The file holds the content only: neither the class nor the
``default_setup``. The result has the class on which ``from_json`` is
called and, unless ``default_setup`` is given, the default configuration of
that class, as with :meth:`~ndict_tools.NestedDictionary.from_dict`. The
same file can give a :class:`~ndict_tools.StrictNestedDictionary`, or a
dictionary printed with an indentation:

.. doctest:: serialization

   >>> from ndict_tools import StrictNestedDictionary
   >>> checked = StrictNestedDictionary.from_json("schedule.json")
   >>> checked.isomorph(schedule)
   True
   >>> printable = NestedDictionary.from_json(
   ...     "schedule.json",
   ...     default_setup={"indent": 2, "default_factory": NestedDictionary},
   ... )
   >>> printable.default_setup[0]
   ('indent', 2)

Keys of type :class:`str`, :class:`int`, :class:`float` and :class:`bool`,
and tuples and frozensets of such values, survive the round trip. A string
key that looks like an encoded key is escaped, so it stays a string:

.. doctest:: serialization

   >>> labels = NestedDictionary({7: "wake up", "[7]": "a label"})
   >>> labels.to_json("labels.json")
   >>> NestedDictionary.from_json("labels.json").to_dict()
   {7: 'wake up', '[7]': 'a label'}

A key of another type raises :class:`~ndict_tools.StackedTypeError`. The
values must be types that JSON can write: strings, numbers, booleans,
``None``, lists and dictionaries.

.. doctest:: serialization

   >>> NestedDictionary({None: "unknown hour"}).to_json("unknown.json")
   Traceback (most recent call last):
       ...
   ndict_tools.exception.StackedTypeError: JSON key encoding is not supported for type NoneType. Supported types: str, int, float, bool, tuple, frozenset. (expected: str, got: NoneType)


Pickle
------

.. warning::

   Loading a pickle file can run arbitrary code. Only load files that you
   wrote yourself or that come from a source you trust. ``to_pickle`` and
   ``from_pickle`` emit a :class:`UserWarning` as a reminder; the examples
   below hide it.

:meth:`~ndict_tools.NestedDictionary.to_pickle` writes the whole object,
and next to it a ``.sha256`` file that holds the digest of the pickle file:

.. doctest:: serialization

   >>> schedule.to_pickle("schedule.pkl")
   >>> sorted(path.name for path in Path().glob("schedule.pkl*"))
   ['schedule.pkl', 'schedule.pkl.sha256']

:meth:`~ndict_tools.NestedDictionary.from_pickle` checks the digest, then
returns the object with its class and its ``default_setup``, with no
argument to give:

.. doctest:: serialization

   >>> NestedDictionary.from_pickle("schedule.pkl") == schedule
   True

If the file was changed after it was written, or if the ``.sha256`` file is
missing, ``from_pickle`` raises :class:`~ndict_tools.StackedValueError`
before loading anything:

.. doctest:: serialization

   >>> with open("schedule.pkl", "ab") as file:
   ...     _ = file.write(b"tampered")
   >>> NestedDictionary.from_pickle("schedule.pkl")  # doctest: +IGNORE_EXCEPTION_DETAIL
   Traceback (most recent call last):
       ...
   ndict_tools.exception.StackedValueError: SHA-256 digest mismatch for 'schedule.pkl': ...

``verify=False`` skips the check, for a file whose origin you trust and
whose digest file was not kept.

The object in the file must be an instance of the class on which
``from_pickle`` is called, or of a subclass. Otherwise
:class:`~ndict_tools.StackedTypeError` is raised:

.. doctest:: serialization

   >>> schedule.to_pickle("schedule.pkl")
   >>> StrictNestedDictionary.from_pickle("schedule.pkl")
   Traceback (most recent call last):
       ...
   ndict_tools.exception.StackedTypeError: 'schedule.pkl' does not contain an instance of StrictNestedDictionary (expected: StrictNestedDictionary, got: NestedDictionary)


Choosing a format
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - JSON
     - Pickle
   * - Keys
     - ``str``, ``int``, ``float``, ``bool``, and tuples or frozensets of
       these
     - Any hashable value
   * - Values
     - Strings, numbers, booleans, ``None``, lists, dictionaries
     - Any value pickle can write
   * - Class and ``default_setup``
     - Chosen when reading
     - Kept
   * - Readable by
     - Any language and tool
     - Python only
   * - Loading from an untrusted source
     - Safe
     - Never
   * - Integrity check
     - None
     - SHA-256 digest file

Use JSON for configuration files, data exchanged with other programs, and
anything that comes from outside. Use pickle to keep Python objects between
two runs of your own program.
