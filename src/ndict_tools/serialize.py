"""
Private serialization infrastructure for ndict_tools.

This module is **not part of the public API**. It provides the low-level helpers
used by the serialization methods on ``_StackedDict`` (``to_json``, ``from_json``,
``to_pickle``, ``from_pickle``).

Contents
--------
- ``_encode_key`` / ``_decode_key`` : JSON key encoding per DD-021
- ``NestedDictionaryEncoder``       : ``json.JSONEncoder`` subclass
- ``_make_decoder_hook``            : factory for ``object_pairs_hook``
- ``_pickle_dump`` / ``_pickle_load``: pickle helpers with SHA-256 verification

Design decisions
----------------
- DD-021 : JSON key encoding strategy
- DD-022 : Serialization API placement
"""

import ast
import hashlib
import json
import pickle  # nosec B403
import warnings
from pathlib import Path
from typing import Any, Callable

from .exception import StackedTypeError, StackedValueError

# ---------------------------------------------------------------------------
# Key encoding / decoding — DD-021
# ---------------------------------------------------------------------------

#: Supported key types for JSON encoding.
_SUPPORTED_KEY_TYPES = (str, int, float, bool, tuple, frozenset)


def _encode_key(key: Any) -> str:
    """
    Encode a ``_StackedDict`` key to a JSON-safe string.

    JSON mandates string keys. This function maps any supported hashable
    Python key to a unique, reversible string representation following the
    convention defined in DD-021.

    Parameters
    ----------
    key : Any
        The key to encode. Supported types: ``str``, ``int``, ``float``,
        ``bool``, flat ``tuple`` of scalars, flat ``frozenset`` of scalars.

    Returns
    -------
    str
        A JSON-safe string representing the key.

    Raises
    ------
    StackedTypeError
        If ``key`` is of an unsupported type.

    Examples
    --------
    >>> _encode_key("hello")
    'hello'
    >>> _encode_key("[42]")
    '\\\\[42]'
    >>> _encode_key(42)
    '[42]'
    >>> _encode_key(3.14)
    '[3.14]'
    >>> _encode_key(True)
    '[True]'
    >>> _encode_key((1, 2))
    '[(1, 2)]'
    >>> _encode_key(frozenset({1, 2}))
    '[frozenset{1, 2}]'

    Notes
    -----
    - ``bool`` is tested before ``int`` because ``bool`` is a subclass of ``int``.
    - ``float`` values use ``repr()`` to preserve full precision.
    - ``frozenset`` is unordered: element order in the encoded form is not
      guaranteed. Round-trip preserves set equality, not element ordering.
    """
    # bool must be checked before int — bool is a subclass of int
    if isinstance(key, bool):
        return f"[{key}]"
    if isinstance(key, str):
        # Escape string keys that start with [ to avoid collision with encoded keys
        if key.startswith("["):
            return r"\[" + key[1:]
        return key
    if isinstance(key, int):
        return f"[{key}]"
    if isinstance(key, float):
        # repr() preserves full floating-point precision
        return f"[{repr(key)}]"
    if isinstance(key, tuple):
        # repr() produces Python native tuple syntax: (1, 2), (1.0, '2', 3)
        return f"[{repr(key)}]"
    if isinstance(key, frozenset):
        elements = ", ".join(repr(e) for e in key)
        return f"[frozenset{{{elements}}}]"
    raise StackedTypeError(
        f"JSON key encoding is not supported for type {type(key).__name__}. "
        f"Supported types: str, int, float, bool, tuple, frozenset.",
        expected_type=str,
        actual_type=type(key),
    )


def _decode_key(encoded: str) -> Any:
    """
    Decode an encoded JSON key back to its original Python type.

    Applies the five sequential decoding rules defined in DD-021.
    No ``eval()`` is used. ``ast.literal_eval()`` is used only for flat
    tuples of Python scalars, which are valid Python literals by definition.

    Parameters
    ----------
    encoded : str
        The encoded JSON key string produced by ``_encode_key``.

    Returns
    -------
    Any
        The original Python key.

    Examples
    --------
    >>> _decode_key("hello")
    'hello'
    >>> _decode_key('\\\\[42]')
    '[42]'
    >>> _decode_key("[42]")
    42
    >>> _decode_key("[3.14]")
    3.14
    >>> _decode_key("[True]")
    True
    >>> _decode_key("[(1, 2)]")
    (1, 2)
    >>> _decode_key("[frozenset{1, 2}]")
    frozenset({1, 2})

    Notes
    -----
    Decoding rules (applied in order):

    1. Starts with ``\\[`` → strip ``\\``, return as ``str``.
    2. Starts with ``[frozenset{`` and ends with ``}]`` → parse inner CSV
       of scalars, return as ``frozenset``.
    3. Starts with ``[(`` and ends with ``)]`` → ``ast.literal_eval()``,
       return as ``tuple``.
    4. Starts with ``[`` and ends with ``]`` → infer scalar type
       (``True``/``False`` → ``bool``; ``.`` or ``e`` → ``float``; else ``int``).
    5. Otherwise → return as ``str`` unchanged.
    """
    # Rule 1: escaped string starting with \[
    if encoded.startswith(r"\["):
        return "[" + encoded[2:]

    # Rule 2: frozenset — [frozenset{...}]
    if encoded.startswith("[frozenset{") and encoded.endswith("}]"):
        inner = encoded[len("[frozenset{") : -len("}]")].strip()
        if not inner:
            return frozenset()
        # ast.literal_eval on a tuple expression is safe: only scalar literals
        elements = ast.literal_eval(f"({inner},)")
        return frozenset(elements)

    # Rule 3: tuple — [(...)]\
    if encoded.startswith("[(") and encoded.endswith(")]"):
        # ast.literal_eval is safe here: content is guaranteed to be a flat
        # tuple of Python scalar literals (int, float, bool, str)
        return ast.literal_eval(encoded[1:-1])

    # Rule 4: scalar — [...]
    if encoded.startswith("[") and encoded.endswith("]"):
        inner = encoded[1:-1]
        if inner == "True":
            return True
        if inner == "False":
            return False
        # float: contains '.' or 'e'/'E' (scientific notation)
        if "." in inner or "e" in inner.lower():
            return float(inner)
        return int(inner)

    # Rule 5: plain string — unchanged
    return encoded


# ---------------------------------------------------------------------------
# JSON encoder / decoder — #43
# ---------------------------------------------------------------------------


class NestedDictionaryEncoder(json.JSONEncoder):
    """
    JSON encoder for ``_StackedDict`` instances.

    Encodes ``_StackedDict`` (and subclasses) as plain nested ``dict``,
    applying ``_encode_key`` to all non-string keys so that the JSON output
    is valid and fully reversible.

    Used internally by ``_StackedDict.to_json``. Not part of the public API.

    Examples
    --------
    >>> import json
    >>> from ndict_tools import NestedDictionary
    >>> nd = NestedDictionary({"a": {"b": 1}})
    >>> json.dumps(nd, cls=NestedDictionaryEncoder)
    '{"a": {"b": 1}}'
    """

    def default(self, obj: Any) -> Any:
        # Lazy import to avoid circular dependency at module load time
        from .tools import _StackedDict

        if isinstance(obj, _StackedDict):
            return {_encode_key(k): v for k, v in obj.items()}
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        from .tools import _StackedDict

        if isinstance(obj, _StackedDict):
            return super().encode({_encode_key(k): v for k, v in obj.items()})
        return super().encode(obj)

    def iterencode(self, obj: Any, _one_shot: bool = False):
        from .tools import _StackedDict

        if isinstance(obj, _StackedDict):
            obj = self._encode_stacked(obj)
        return super().iterencode(obj, _one_shot)

    def _encode_stacked(self, obj: Any) -> Any:
        """Recursively convert _StackedDict to plain dict with encoded keys."""
        from .tools import _StackedDict

        if isinstance(obj, _StackedDict):
            return {_encode_key(k): self._encode_stacked(v) for k, v in obj.items()}
        return obj


def _make_decoder_hook(cls: type, class_options: dict) -> Callable:
    """
    Return an ``object_pairs_hook`` that reconstructs a ``_StackedDict``
    (or subclass) from JSON key-value pairs.

    Parameters
    ----------
    cls : type
        The ``_StackedDict`` subclass to instantiate.
    class_options : dict
        Keyword arguments forwarded to ``cls.from_dict``, must include
        ``default_setup``.

    Returns
    -------
    Callable
        A hook suitable for ``json.load(..., object_pairs_hook=hook)``.
    """

    def hook(pairs: list) -> Any:
        decoded = {_decode_key(k): v for k, v in pairs}
        return cls.from_dict(decoded, **class_options)

    return hook


# ---------------------------------------------------------------------------
# Pickle helpers — #44
# ---------------------------------------------------------------------------


def _pickle_dump(
    nd: Any,
    path: "str | Path",
    protocol: int | None = None,
) -> None:
    """
    Write a ``_StackedDict`` to a pickle file with a SHA-256 sidecar.

    Writes two files:
    - ``<path>`` — the pickle file
    - ``<path>.sha256`` — hex digest of the pickle bytes

    Parameters
    ----------
    nd : _StackedDict
        The object to pickle.
    path : str or Path
        Destination file path.
    protocol : int, optional
        Pickle protocol (default: ``pickle.DEFAULT_PROTOCOL``).

    Warns
    -----
    UserWarning
        Always emits a warning reminding callers that pickle is unsafe
        with untrusted files.
    """
    warnings.warn(
        "Pickle files are unsafe when loaded from untrusted sources. "
        "Only unpickle files you created yourself or received from trusted sources.",
        UserWarning,
        stacklevel=3,
    )
    path = Path(path)
    data = pickle.dumps(nd, protocol=protocol or pickle.DEFAULT_PROTOCOL)
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(digest, encoding="utf-8")


def _pickle_load(
    path: "str | Path",
    verify: bool = True,
) -> Any:
    """
    Load a ``_StackedDict`` from a pickle file, optionally verifying its
    SHA-256 sidecar.

    Parameters
    ----------
    path : str or Path
        Path to the pickle file.
    verify : bool, optional
        If ``True`` (default), read the ``.sha256`` sidecar and raise
        ``StackedValueError`` if the digest does not match or the sidecar
        is absent.

    Returns
    -------
    Any
        The unpickled object.

    Raises
    ------
    StackedValueError
        If ``verify=True`` and the digest mismatches or the sidecar is absent.

    Warns
    -----
    UserWarning
        Always emits a warning reminding callers that pickle is unsafe
        with untrusted files.
    """
    warnings.warn(
        "Pickle files are unsafe when loaded from untrusted sources. "
        "Only unpickle files you created yourself or received from trusted sources.",
        UserWarning,
        stacklevel=3,
    )
    path = Path(path)
    data = path.read_bytes()

    if verify:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists():
            raise StackedValueError(
                f"SHA-256 sidecar not found for '{path}'. "
                "Use verify=False to skip integrity check.",
                value=str(path),
            )
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise StackedValueError(
                f"SHA-256 digest mismatch for '{path}': "
                f"expected {expected!r}, got {actual!r}. "
                "The file may be corrupted or tampered with.",
                value=str(path),
            )

    return pickle.loads(data)  # nosec B301
