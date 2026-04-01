"""
Serialization tests for NestedDictionary, StrictNestedDictionary,
SmoothNestedDictionary, and custom subclasses — issues #43, #44.

Design principles:
- All three public variants are tested via @pytest.mark.parametrize.
- Behavioral differences (strict/smooth on unknown keys) are tested after
  deserialization to verify that the reconstructed instance behaves correctly.
- Subclass tests verify that from_json / from_pickle / __reduce__ preserve
  the subclass type and custom attributes.
- xfail is used where a behavior is expected to differ between variants.

Placed in: tests/02_Nesting/05_Serialization/
Fixtures used: function_system_config, tmp_function_file (global conftest)
"""

import pickle
from copy import deepcopy

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.exception import StackedValueError
from ndict_tools.tools import _StackedDict


# ---------------------------------------------------------------------------
# Parametrize configuration for all three public variants
# ---------------------------------------------------------------------------

#: (class, expected default_factory, initial factory for make())
#: Each class overrides default_factory in __init__ — only NestedDictionary
#: respects the passed factory; Strict forces None, Smooth forces itself.
VARIANTS = [
    pytest.param(
        NestedDictionary,
        NestedDictionary,
        NestedDictionary,
        id="NestedDictionary",
    ),
    pytest.param(
        StrictNestedDictionary,
        None,
        None,
        id="StrictNestedDictionary",
    ),
    pytest.param(
        SmoothNestedDictionary,
        SmoothNestedDictionary,
        None,
        id="SmoothNestedDictionary",
    ),
]

def _setup(factory):
    """Return a fresh default_setup dict to avoid in-place mutation by subclasses."""
    return {"indent": 0, "default_factory": factory}


def make(cls, data, factory=None):
    """Create an instance of cls from a dict using from_dict classmethod."""
    return cls.from_dict(deepcopy(data), default_setup=_setup(factory))


# ---------------------------------------------------------------------------
# Subclass used in subclass-specific tests
# ---------------------------------------------------------------------------

class _CustomDict(_StackedDict):
    """Minimal subclass with a custom attribute for testing."""

    def __init__(self, *args, **kwargs):
        self.custom = False
        settings = kwargs.pop("default_setup", {})
        settings.setdefault("indent", 0)
        settings.setdefault("default_factory", None)
        settings["custom"] = True
        super().__init__(*args, **kwargs, default_setup=settings)


# ---------------------------------------------------------------------------
# JSON — all variants
# ---------------------------------------------------------------------------

class TestJsonAllVariants:

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_to_json_creates_file(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """to_json writes a non-empty file for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}.json"
        nd.to_json(path)
        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_json_round_trip_structure(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """from_json produces a structurally equal instance for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_rt.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        assert restored.to_dict() == nd.to_dict()

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_from_json_returns_correct_type(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """from_json returns an instance of the calling class."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_type.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        assert type(restored) is cls

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_from_json_default_factory(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Each variant forces its own default_factory regardless of class_options."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_factory.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        assert restored.default_factory is expected_factory

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_json_non_string_keys_preserved(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Tuple, frozenset, and int keys survive JSON round-trip for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_keys.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        assert ("env", "production") in restored
        assert ("env", "dev") in restored
        assert frozenset(["cache", "redis"]) in restored
        assert "monitoring" in restored

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_json_nested_int_keys_preserved(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Nested integer keys survive JSON round-trip."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_int_keys.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        env_prod = restored[("env", "production")]
        assert 1 in env_prod["database"]["replicas"]
        assert 2 in env_prod["database"]["replicas"]

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_json_nested_tuple_keys_preserved(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Nested tuple keys survive JSON round-trip."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_tuple_keys.json"
        nd.to_json(path)
        restored = cls.from_json(path, default_setup=_setup(init_factory))
        monitoring = restored["monitoring"]
        assert ("metrics", "cpu") in monitoring
        assert ("logs", "level") in monitoring

    # --- Behavioral verification after deserialization ---

    def test_strict_raises_on_unknown_key_after_json(self, function_system_config, tmp_function_file):
        """StrictNestedDictionary raises KeyError on unknown key after from_json."""
        nd = make(StrictNestedDictionary, function_system_config, None)
        path = tmp_function_file / "strict_behavior.json"
        nd.to_json(path)
        restored = StrictNestedDictionary.from_json(path, default_setup=_setup(None))
        with pytest.raises(KeyError):
            _ = restored["nonexistent_key"]

    def test_smooth_returns_empty_on_unknown_key_after_json(self, function_system_config, tmp_function_file):
        """SmoothNestedDictionary returns empty instance on unknown key after from_json."""
        nd = make(SmoothNestedDictionary, function_system_config, None)
        path = tmp_function_file / "smooth_behavior.json"
        nd.to_json(path)
        restored = SmoothNestedDictionary.from_json(path, default_setup=_setup(None))
        result = restored["nonexistent_key"]
        assert isinstance(result, SmoothNestedDictionary)


# ---------------------------------------------------------------------------
# Pickle — all variants
# ---------------------------------------------------------------------------

class TestPickleAllVariants:

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_to_pickle_creates_files(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """to_pickle creates both pickle file and .sha256 sidecar."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}.pkl"
        with pytest.warns(UserWarning, match="unsafe"):
            nd.to_pickle(path)
        assert path.exists()
        assert path.with_suffix(".pkl.sha256").exists()

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_pickle_round_trip_structure(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Pickle round-trip preserves structure for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_rt.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = cls.from_pickle(path)
        assert restored.to_dict() == nd.to_dict()

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_from_pickle_returns_correct_type(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """from_pickle returns an instance of the original class."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_type.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = cls.from_pickle(path)
        assert type(restored) is cls

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_pickle_preserves_default_factory(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """Pickle round-trip preserves default_factory for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_factory.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = cls.from_pickle(path)
        assert restored.default_factory is expected_factory

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_pickle_all_key_types_preserved(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """All key types from _SYSTEM_CONFIG survive pickle round-trip."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_keys.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = cls.from_pickle(path)
        assert ("env", "production") in restored
        assert frozenset(["cache", "redis"]) in restored
        assert "monitoring" in restored

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_tampered_pickle_raises(self, cls, expected_factory, init_factory, function_system_config, tmp_function_file):
        """StackedValueError raised on tampered pickle for every variant."""
        nd = make(cls, function_system_config, init_factory)
        path = tmp_function_file / f"{cls.__name__}_tamper.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        path.write_bytes(path.read_bytes() + b"\x00\x01\x02")
        with pytest.warns(UserWarning):
            with pytest.raises(StackedValueError, match="digest mismatch"):
                cls.from_pickle(path, verify=True)

    # --- Behavioral verification after deserialization ---

    def test_strict_raises_on_unknown_key_after_pickle(self, function_system_config, tmp_function_file):
        """StrictNestedDictionary raises KeyError on unknown key after from_pickle."""
        nd = make(StrictNestedDictionary, function_system_config, None)
        path = tmp_function_file / "strict_behavior.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = StrictNestedDictionary.from_pickle(path)
        with pytest.raises(KeyError):
            _ = restored["nonexistent_key"]

    def test_smooth_returns_empty_on_unknown_key_after_pickle(self, function_system_config, tmp_function_file):
        """SmoothNestedDictionary returns empty instance on unknown key after from_pickle."""
        nd = make(SmoothNestedDictionary, function_system_config, None)
        path = tmp_function_file / "smooth_behavior.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = SmoothNestedDictionary.from_pickle(path)
        result = restored["nonexistent_key"]
        assert isinstance(result, SmoothNestedDictionary)


# ---------------------------------------------------------------------------
# Native pickle (__reduce__) — all variants
# ---------------------------------------------------------------------------

class TestNativePickleAllVariants:

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_native_pickle_round_trip(self, cls, expected_factory, init_factory, function_system_config):
        """stdlib pickle.dumps/loads round-trip for every variant."""
        nd = make(cls, function_system_config, init_factory)
        restored = pickle.loads(pickle.dumps(nd))  # noqa: S301
        assert type(restored) is cls
        assert restored.to_dict() == nd.to_dict()

    @pytest.mark.parametrize("cls,expected_factory,init_factory", VARIANTS)
    def test_native_pickle_preserves_default_factory(self, cls, expected_factory, init_factory, function_system_config):
        """__reduce__ preserves default_factory for every variant."""
        nd = make(cls, function_system_config, init_factory)
        restored = pickle.loads(pickle.dumps(nd))  # noqa: S301
        assert restored.default_factory is expected_factory


# ---------------------------------------------------------------------------
# Subclass serialization
# ---------------------------------------------------------------------------

class TestSerializationSubclass:
    """
    Verify that from_json / from_pickle / __reduce__ work correctly
    when called on a custom _StackedDict subclass.
    """

    def test_subclass_from_json_returns_subclass_type(self, function_system_config, tmp_function_file):
        """_CustomDict.from_json returns a _CustomDict instance."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        path = tmp_function_file / "custom.json"
        nd.to_json(path)
        restored = _CustomDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert type(restored) is _CustomDict

    def test_subclass_from_json_preserves_structure(self, function_system_config, tmp_function_file):
        """_CustomDict.from_json produces structurally equal content."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        path = tmp_function_file / "custom_rt.json"
        nd.to_json(path)
        restored = _CustomDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert restored.to_dict() == nd.to_dict()

    def test_subclass_native_pickle_returns_subclass_type(self, function_system_config):
        """stdlib pickle preserves _CustomDict type via __reduce__."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        restored = pickle.loads(pickle.dumps(nd))  # noqa: S301
        assert type(restored) is _CustomDict

    def test_subclass_native_pickle_preserves_structure(self, function_system_config):
        """stdlib pickle preserves _CustomDict content via __reduce__."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        restored = pickle.loads(pickle.dumps(nd))  # noqa: S301
        assert restored.to_dict() == nd.to_dict()

    def test_subclass_custom_attribute_preserved_through_pickle(self, function_system_config):
        """Custom attribute (balanced/custom) is restored by __init__ after pickle."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        assert nd.custom is True
        restored = pickle.loads(pickle.dumps(nd))  # noqa: S301
        # __init__ is called by _reconstruct → cls(**class_options)
        # _CustomDict.__init__ sets self.custom = True
        assert restored.custom is True

    def test_subclass_to_pickle_round_trip(self, function_system_config, tmp_function_file):
        """to_pickle / from_pickle work on _CustomDict."""
        nd = _CustomDict.from_dict(
            deepcopy(function_system_config),
            default_setup={"indent": 0, "default_factory": None},
        )
        path = tmp_function_file / "custom.pkl"
        with pytest.warns(UserWarning):
            nd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _CustomDict.from_pickle(path)
        assert type(restored) is _CustomDict
        assert restored.to_dict() == nd.to_dict()
