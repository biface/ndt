"""
Unit tests for _encode_key and _decode_key in serialize.py — issue #48.

These tests are fully isolated: no _StackedDict, no fixtures.
They verify the encoding contract defined in DD-021.

Coverage:
- All supported scalar types: str, int, float, bool
- Container types: flat tuple, flat frozenset
- String key collision: escape of keys starting with '['
- Float precision: repr() guarantees round-trip fidelity
- frozenset: round-trip preserves set equality, not order
- StackedTypeError raised for unsupported types
- Decoding rules applied in correct order (bool before int, float detection)
"""

import pytest

from ndict_tools.exception import StackedTypeError
from ndict_tools.serialize import _decode_key, _encode_key

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def round_trip(key):
    """Encode then decode a key and return the result."""
    return _decode_key(_encode_key(key))


# ---------------------------------------------------------------------------
# String keys — pass-through and escape
# ---------------------------------------------------------------------------


class TestEncodeDecodeString:

    @pytest.mark.parametrize(
        "key",
        [
            "hello",
            "world",
            "",
            "with spaces",
            "with | pipe",
            "with unicode: é à ü",
            "0",
            "True",
            "False",
            "3.14",
        ],
    )
    def test_plain_string_passthrough(self, key):
        """Plain strings are returned unchanged."""
        assert _encode_key(key) == key
        assert _decode_key(key) == key

    @pytest.mark.parametrize(
        "key",
        [
            "[42]",
            "[hello]",
            "[(1, 2)]",
            "[frozenset{1, 2}]",
            "[True]",
            "[]",
        ],
    )
    def test_string_starting_with_bracket_escaped(self, key):
        """String keys starting with '[' are escaped with '\\['."""
        encoded = _encode_key(key)
        assert encoded.startswith(r"\[")
        assert round_trip(key) == key

    def test_escape_prefix_exact(self):
        """Escape adds exactly one backslash before the bracket."""
        assert _encode_key("[42]") == r"\[42]"
        assert _encode_key("[hello]") == r"\[hello]"


# ---------------------------------------------------------------------------
# Integer keys
# ---------------------------------------------------------------------------


class TestEncodeDecodeInt:

    @pytest.mark.parametrize("key", [0, 1, 42, -1, -100, 999999])
    def test_int_round_trip(self, key):
        assert round_trip(key) == key
        assert isinstance(round_trip(key), int)

    @pytest.mark.parametrize("key", [0, 42, -7])
    def test_int_encoding_format(self, key):
        assert _encode_key(key) == f"[{key}]"

    def test_int_decoded_as_int_not_float(self):
        """'[42]' must decode to int 42, not float 42.0."""
        result = _decode_key("[42]")
        assert result == 42
        assert type(result) is int


# ---------------------------------------------------------------------------
# Float keys
# ---------------------------------------------------------------------------


class TestEncodeDecodeFloat:

    @pytest.mark.parametrize("key", [3.14, 0.0, -1.5, 1e10, 1e-10])
    def test_float_round_trip(self, key):
        assert round_trip(key) == key
        assert isinstance(round_trip(key), float)

    def test_float_full_precision(self):
        """repr() preserves full float precision."""
        key = 1.0000000000000002
        assert round_trip(key) == key

    def test_float_encoding_uses_repr(self):
        """Float encoding uses repr(), not str()."""
        key = 1.0000000000000002
        encoded = _encode_key(key)
        assert "1.0000000000000002" in encoded

    @pytest.mark.parametrize("key", [1.5, -0.1, 3.14])
    def test_float_decoded_as_float(self, key):
        result = round_trip(key)
        assert type(result) is float

    def test_scientific_notation_decoded_as_float(self):
        """Scientific notation (e.g. 1e10) is decoded as float."""
        key = 1e10
        result = round_trip(key)
        assert result == key
        assert type(result) is float


# ---------------------------------------------------------------------------
# Bool keys
# ---------------------------------------------------------------------------


class TestEncodeDecodeBool:

    @pytest.mark.parametrize("key", [True, False])
    def test_bool_round_trip(self, key):
        assert round_trip(key) == key
        assert type(round_trip(key)) is bool

    def test_true_encoding(self):
        assert _encode_key(True) == "[True]"

    def test_false_encoding(self):
        assert _encode_key(False) == "[False]"

    def test_bool_not_decoded_as_int(self):
        """[True] must decode to bool True, not int 1."""
        assert _decode_key("[True]") is True
        assert _decode_key("[False]") is False
        assert type(_decode_key("[True]")) is bool

    def test_bool_encoded_before_int(self):
        """bool is a subclass of int — must be encoded as bool, not int."""
        assert _encode_key(True) == "[True]"
        assert _encode_key(True) != "[1]"
        assert _encode_key(False) == "[False]"
        assert _encode_key(False) != "[0]"


# ---------------------------------------------------------------------------
# Tuple keys (flat)
# ---------------------------------------------------------------------------


class TestEncodeDecodeTuple:

    @pytest.mark.parametrize(
        "key",
        [
            (1, 2),
            (1, 2, 3),
            ("a", "b"),
            (1, "a", True),
            (1.0, "2", 3),
            (True, "a"),
            ("env", "production"),
            ("env", "dev"),
            ("metrics", "cpu"),
            ("logs", "level"),
            ("security", "encryption"),
        ],
    )
    def test_tuple_round_trip(self, key):
        assert round_trip(key) == key
        assert isinstance(round_trip(key), tuple)

    def test_tuple_encoding_format(self):
        """Tuple uses Python native repr syntax wrapped in [...]."""
        assert _encode_key((1, 2)) == "[(1, 2)]"

    def test_single_element_tuple(self):
        """Single-element tuple has trailing comma in repr."""
        key = (42,)
        assert round_trip(key) == key
        assert type(round_trip(key)) is tuple

    def test_empty_tuple(self):
        key = ()
        assert round_trip(key) == key


# ---------------------------------------------------------------------------
# Frozenset keys (flat)
# ---------------------------------------------------------------------------


class TestEncodeDecodeFrozenset:

    @pytest.mark.parametrize(
        "key",
        [
            frozenset({1, 2}),
            frozenset({"a", 1}),
            frozenset({"cache", "redis"}),
            frozenset({True, 42}),
        ],
    )
    def test_frozenset_round_trip(self, key):
        result = round_trip(key)
        assert result == key
        assert isinstance(result, frozenset)

    def test_empty_frozenset(self):
        key = frozenset()
        result = round_trip(key)
        assert result == key
        assert isinstance(result, frozenset)

    def test_frozenset_encoding_format(self):
        """Frozenset uses [frozenset{...}] convention."""
        encoded = _encode_key(frozenset({1, 2}))
        assert encoded.startswith("[frozenset{")
        assert encoded.endswith("}]")

    def test_frozenset_order_irrelevant(self):
        """Element order in encoded form is not guaranteed — set equality only."""
        key = frozenset({1, 2, 3})
        assert round_trip(key) == key

    def test_frozenset_with_strings(self):
        key = frozenset({"cache", "redis"})
        result = round_trip(key)
        assert result == key


# ---------------------------------------------------------------------------
# Keys from _SYSTEM_CONFIG (real-world fixture)
# ---------------------------------------------------------------------------


class TestRealWorldKeys:
    """
    Verify encoding/decoding on all key types present in _SYSTEM_CONFIG.
    These are the exact keys used in integration tests.
    """

    @pytest.mark.parametrize(
        "key",
        [
            ("env", "production"),
            ("env", "dev"),
            ("metrics", "cpu"),
            ("logs", "level"),
            ("security", "encryption"),
            frozenset(["cache", "redis"]),
            1,
            2,
            42,
            54,
            12,
            34,
            "monitoring",
            "global_settings",
            "database",
            "api",
        ],
    )
    def test_system_config_keys_round_trip(self, key):
        assert round_trip(key) == key


# ---------------------------------------------------------------------------
# Unsupported types → StackedTypeError
# ---------------------------------------------------------------------------


class TestUnsupportedTypes:

    @pytest.mark.parametrize(
        "key",
        [
            [1, 2],  # list — not hashable
            {"a": 1},  # dict — not hashable
            object(),  # generic object
            bytes(b"hello"),  # bytes — post-1.2.0
            complex(1, 2),  # complex — post-1.2.0
        ],
    )
    def test_unsupported_type_raises(self, key):
        with pytest.raises((StackedTypeError, TypeError)):
            _encode_key(key)


# ---------------------------------------------------------------------------
# Decoding rules order
# ---------------------------------------------------------------------------


class TestDecodingRulesOrder:
    """Verify that the 5 decoding rules are applied in the correct order."""

    def test_rule_1_escaped_bracket(self):
        """Rule 1: \\[ → plain string."""
        assert _decode_key(r"\[42]") == "[42]"
        assert _decode_key(r"\[frozenset{1}]") == "[frozenset{1}]"

    def test_rule_2_frozenset(self):
        """Rule 2: [frozenset{...}] → frozenset."""
        assert _decode_key("[frozenset{1, 2}]") == frozenset({1, 2})

    def test_rule_3_tuple(self):
        """Rule 3: [(...)] → tuple via ast.literal_eval."""
        assert _decode_key("[(1, 2)]") == (1, 2)

    def test_rule_4_bool_before_int(self):
        """Rule 4: True/False decoded before int inference."""
        assert _decode_key("[True]") is True
        assert _decode_key("[False]") is False

    def test_rule_4_float_detected_by_dot(self):
        """Rule 4: presence of '.' triggers float."""
        assert _decode_key("[3.14]") == 3.14

    def test_rule_4_int_fallback(self):
        """Rule 4: no '.', not bool → int."""
        assert _decode_key("[42]") == 42

    def test_rule_5_plain_string(self):
        """Rule 5: anything else is a plain string."""
        assert _decode_key("hello") == "hello"
        assert _decode_key("") == ""
