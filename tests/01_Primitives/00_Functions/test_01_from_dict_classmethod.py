"""
Tests for _StackedDict.from_dict() @classmethod — issue #42.

Covers:
- Basic construction via classmethod on all four variants
- Correct type propagation to nested dicts
- _default_setup preservation through the classmethod
- Already-instantiated _StackedDict values preserved as-is
- StackedKeyError when default_setup is absent
- DeprecationWarning emitted by the free function
- Round-trip: from_dict(to_dict()) produces a structurally equal dict
"""

import re

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.exception import StackedKeyError
from ndict_tools.tools import _StackedDict, from_dict

# ---------------------------------------------------------------------------
# Classmethod — basic construction
# ---------------------------------------------------------------------------


class TestFromDictClassmethod:

    @pytest.mark.parametrize(
        "cls",
        [
            _StackedDict,
            NestedDictionary,
            StrictNestedDictionary,
            SmoothNestedDictionary,
        ],
    )
    def test_returns_correct_type(self, cls, function_system_config):
        """from_dict classmethod returns an instance of the calling class."""
        result = cls.from_dict(
            function_system_config,
            default_setup={"indent": 0, "default_factory": None},
        )
        assert isinstance(result, cls)

    @pytest.mark.parametrize(
        "cls",
        [
            _StackedDict,
            NestedDictionary,
            StrictNestedDictionary,
            SmoothNestedDictionary,
        ],
    )
    def test_nested_dicts_have_correct_type(self, cls, function_system_config):
        """Nested dicts are converted to the same cls, not the base class."""
        result = cls.from_dict(
            function_system_config,
            default_setup={"indent": 0, "default_factory": None},
        )
        for value in result.values():
            if isinstance(value, _StackedDict):
                assert type(value) is cls

    @pytest.mark.parametrize(
        "default_setup, keys",
        [
            ({"indent": 2, "default_factory": None}, []),
            ({"indent": 2, "default_factory": _StackedDict}, ["monitoring"]),
        ],
    )
    def test_default_setup_propagated(
        self, function_system_config, default_setup, keys
    ):
        """_default_setup is propagated to all nested instances."""
        result = _StackedDict.from_dict(
            function_system_config,
            default_setup=default_setup,
        )
        d = result
        for key in keys:
            d = d[key]
        assert isinstance(d, _StackedDict)
        assert d._default_setup == set(default_setup.items())

    def test_already_stacked_dict_preserved(self, function_system_config):
        """An already-instantiated _StackedDict value is preserved as-is."""
        existing = _StackedDict(
            {"x": 1},
            default_setup={"indent": 0, "default_factory": None},
        )
        data = {"nested": existing, "scalar": 42}
        result = _StackedDict.from_dict(
            data,
            default_setup={"indent": 0, "default_factory": None},
        )
        assert result["nested"] is existing

    def test_scalar_values_assigned_directly(self):
        """Non-dict scalar values are assigned without conversion."""
        data = {"a": 1, "b": 3.14, "c": True, "d": "hello", "e": [1, 2, 3]}
        result = _StackedDict.from_dict(
            data,
            default_setup={"indent": 0, "default_factory": None},
        )
        assert result["a"] == 1
        assert result["b"] == 3.14
        assert result["c"] is True
        assert result["d"] == "hello"
        assert result["e"] == [1, 2, 3]

    # -----------------------------------------------------------------------
    # Error cases
    # -----------------------------------------------------------------------

    def test_missing_default_setup_raises(self, function_system_config):
        """StackedKeyError raised when default_setup is absent."""
        with pytest.raises(
            StackedKeyError,
            match=re.escape("The key 'default_setup' must be present in class options"),
        ):
            _StackedDict.from_dict(function_system_config, none_setup={})

    # -----------------------------------------------------------------------
    # Round-trip
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "cls",
        [NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary],
    )
    def test_round_trip(self, cls, function_system_config):
        """from_dict(to_dict()) produces a structurally equal dictionary."""
        original = cls.from_dict(
            function_system_config,
            default_setup={"indent": 0, "default_factory": None},
        )
        restored = cls.from_dict(
            original.to_dict(),
            default_setup={"indent": 0, "default_factory": None},
        )
        assert original.to_dict() == restored.to_dict()


# ---------------------------------------------------------------------------
# Free function deprecation
# ---------------------------------------------------------------------------


class TestFromDictFreeFunction:

    def test_deprecation_warning_emitted(self, function_system_config):
        """The free function emits DeprecationWarning mentioning 1.5.0."""
        with pytest.warns(DeprecationWarning, match="1.5.0"):
            from_dict(
                function_system_config,
                _StackedDict,
                default_setup={"indent": 0, "default_factory": None},
            )

    def test_free_function_still_works(self, function_system_config):
        """The free function still returns a valid result despite the warning."""
        with pytest.warns(DeprecationWarning):
            result = from_dict(
                function_system_config,
                _StackedDict,
                default_setup={"indent": 0, "default_factory": None},
            )
        assert isinstance(result, _StackedDict)
