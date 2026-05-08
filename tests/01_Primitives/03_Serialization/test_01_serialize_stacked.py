"""
Serialization tests on _StackedDict — issues #43, #44.

Covers to_json / from_json / to_pickle / from_pickle on _StackedDict
using _SYSTEM_CONFIG as test data (tuple, frozenset, int keys).

Structure:
- TestJsonSerializationStackedDict  : JSON round-trip, key encoding, error cases
- TestPickleSerializationStackedDict: pickle round-trip, SHA-256, tamper detection
- TestNativePickle                  : __reduce__ via stdlib pickle.dumps/loads
"""

import pickle
from copy import deepcopy
from pathlib import Path

import pytest

from ndict_tools.exception import StackedValueError
from ndict_tools.tools import _StackedDict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_strict(data: dict) -> _StackedDict:
    return _StackedDict.from_dict(
        data, default_setup={"indent": 0, "default_factory": None}
    )


def make_smooth(data: dict) -> _StackedDict:
    return _StackedDict.from_dict(
        data, default_setup={"indent": 0, "default_factory": _StackedDict}
    )


# ---------------------------------------------------------------------------
# JSON — _StackedDict
# ---------------------------------------------------------------------------


class TestJsonSerializationStackedDict:

    # --- Round-trip: strict mode ---

    def test_to_json_creates_file(self, function_system_config, tmp_function_file):
        """to_json writes a file at the given path."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.json"
        sd.to_json(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_json_round_trip_strict(self, function_system_config, tmp_function_file):
        """from_json reconstructs a structurally equal _StackedDict (strict)."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_strict.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert isinstance(restored, _StackedDict)
        assert restored.to_dict() == sd.to_dict()

    def test_json_round_trip_smooth(self, function_system_config, tmp_function_file):
        """from_json reconstructs a structurally equal _StackedDict (smooth)."""
        sd = make_smooth(function_system_config)
        path = tmp_function_file / "sd_smooth.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": _StackedDict}
        )
        assert restored.to_dict() == sd.to_dict()

    # --- Non-string keys ---

    @pytest.mark.parametrize(
        "key",
        [
            ("env", "production"),
            ("env", "dev"),
            frozenset(["cache", "redis"]),
            "monitoring",
            "global_settings",
        ],
    )
    def test_top_level_keys_preserved(
        self, function_system_config, tmp_function_file, key
    ):
        """All top-level keys from _SYSTEM_CONFIG survive JSON round-trip."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_keys.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert key in restored

    def test_nested_int_keys_preserved(self, function_system_config, tmp_function_file):
        """Integer keys nested inside the dict survive JSON round-trip."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_int_keys.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        # replicas uses int keys 1 and 2
        env_prod = restored[("env", "production")]
        assert 1 in env_prod["database"]["replicas"]
        assert 2 in env_prod["database"]["replicas"]

    def test_nested_tuple_keys_preserved(
        self, function_system_config, tmp_function_file
    ):
        """Tuple keys nested at multiple levels survive JSON round-trip."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_tuple_keys.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        monitoring = restored["monitoring"]
        assert ("metrics", "cpu") in monitoring
        assert ("logs", "level") in monitoring

    def test_frozenset_key_preserved(self, function_system_config, tmp_function_file):
        """frozenset key survives JSON round-trip with set equality."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_frozenset.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert frozenset(["cache", "redis"]) in restored

    # --- Values preserved ---

    def test_scalar_values_preserved(self, function_system_config, tmp_function_file):
        """Scalar values (str, int, bool, list) survive JSON round-trip."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_values.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        env_prod = restored[("env", "production")]
        assert env_prod["database"]["host"] == "prod-db.company.com"
        assert env_prod["database"]["port"] == 5432
        assert env_prod["api"]["rate_limit"] == 10000

    def test_list_values_preserved(self, function_system_config, tmp_function_file):
        """List values survive JSON round-trip unchanged."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_lists.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        env_prod = restored[("env", "production")]
        assert env_prod["database"]["pools"] == [5, 10, 15]

    # --- indent option ---

    def test_indent_option(self, function_system_config, tmp_function_file):
        """indent parameter controls JSON pretty-printing."""
        sd = make_strict(function_system_config)
        path_compact = tmp_function_file / "compact.json"
        path_pretty = tmp_function_file / "pretty.json"
        sd.to_json(path_compact, indent=None)
        sd.to_json(path_pretty, indent=2)
        assert path_pretty.stat().st_size > path_compact.stat().st_size

    # --- from_json uses cls ---

    def test_from_json_returns_stackeddict(
        self, function_system_config, tmp_function_file
    ):
        """from_json returns an instance of _StackedDict (or calling class)."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.json"
        sd.to_json(path)
        restored = _StackedDict.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert type(restored) is _StackedDict


# ---------------------------------------------------------------------------
# Pickle — _StackedDict
# ---------------------------------------------------------------------------


class TestPickleSerializationStackedDict:

    # --- Round-trip ---

    def test_to_pickle_creates_files(self, function_system_config, tmp_function_file):
        """to_pickle writes both pickle file and .sha256 sidecar."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning, match="unsafe"):
            sd.to_pickle(path)
        assert path.exists()
        assert path.with_suffix(".pkl.sha256").exists()

    def test_pickle_round_trip_strict(self, function_system_config, tmp_function_file):
        """Pickle round-trip preserves structure and default_factory=None."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_strict.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert isinstance(restored, _StackedDict)
        assert restored.to_dict() == sd.to_dict()

    def test_pickle_round_trip_smooth(self, function_system_config, tmp_function_file):
        """Pickle round-trip preserves structure and default_factory=_StackedDict."""
        sd = make_smooth(function_system_config)
        path = tmp_function_file / "sd_smooth.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert restored.to_dict() == sd.to_dict()

    def test_pickle_preserves_default_factory_none(
        self, function_system_config, tmp_function_file
    ):
        """default_factory=None (strict) is preserved across pickle."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "strict.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert restored.default_factory is None

    def test_pickle_preserves_default_factory_class(
        self, function_system_config, tmp_function_file
    ):
        """default_factory=_StackedDict (smooth) is preserved across pickle."""
        sd = make_smooth(function_system_config)
        path = tmp_function_file / "smooth.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert restored.default_factory is _StackedDict

    def test_pickle_all_key_types_preserved(
        self, function_system_config, tmp_function_file
    ):
        """All key types from _SYSTEM_CONFIG survive pickle round-trip."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd_keys.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert ("env", "production") in restored
        assert frozenset(["cache", "redis"]) in restored
        assert "monitoring" in restored

    # --- SHA-256 verification ---

    def test_verify_true_accepts_intact_file(
        self, function_system_config, tmp_function_file
    ):
        """verify=True passes when file is intact."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path, verify=True)
        assert restored is not None

    def test_tampered_file_raises(self, function_system_config, tmp_function_file):
        """StackedValueError raised when pickle file is tampered."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        # Tamper with the pickle file
        path.write_bytes(path.read_bytes() + b"\x00\x01\x02")
        with pytest.warns(UserWarning):
            with pytest.raises(StackedValueError, match="digest mismatch"):
                _StackedDict.from_pickle(path, verify=True)

    def test_missing_sidecar_raises(self, function_system_config, tmp_function_file):
        """StackedValueError raised when .sha256 sidecar is absent."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        sidecar = path.with_suffix(".pkl.sha256")
        sidecar.unlink()
        with pytest.warns(UserWarning):
            with pytest.raises(StackedValueError, match="not found"):
                _StackedDict.from_pickle(path, verify=True)

    def test_verify_false_skips_hash_check(
        self, function_system_config, tmp_function_file
    ):
        """verify=False loads without checking the sidecar."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        sidecar = path.with_suffix(".pkl.sha256")
        sidecar.unlink()
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path, verify=False)
        assert restored is not None

    def test_userwarning_on_dump(self, function_system_config, tmp_function_file):
        """to_pickle always emits UserWarning about unsafe pickle."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning, match="unsafe"):
            sd.to_pickle(path)

    def test_userwarning_on_load(self, function_system_config, tmp_function_file):
        """from_pickle always emits UserWarning about unsafe pickle."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / "sd.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path)
        with pytest.warns(UserWarning, match="unsafe"):
            _StackedDict.from_pickle(path)

    # --- Protocol ---

    @pytest.mark.parametrize("protocol", [2, 3, 4, 5])
    def test_pickle_protocol_explicit(
        self, function_system_config, tmp_function_file, protocol
    ):
        """Explicit pickle protocols all produce valid round-trips."""
        sd = make_strict(function_system_config)
        path = tmp_function_file / f"sd_proto{protocol}.pkl"
        with pytest.warns(UserWarning):
            sd.to_pickle(path, protocol=protocol)
        with pytest.warns(UserWarning):
            restored = _StackedDict.from_pickle(path)
        assert restored.to_dict() == sd.to_dict()


# ---------------------------------------------------------------------------
# Native pickle via __reduce__
# ---------------------------------------------------------------------------


class TestNativePickle:
    """
    Verify __reduce__ works correctly with stdlib pickle.dumps/pickle.loads.
    This tests the reconstruction callable (_reconstruct) directly.
    """

    def test_native_pickle_round_trip_strict(self, function_system_config):
        """stdlib pickle.dumps/loads round-trip on strict _StackedDict."""
        sd = make_strict(function_system_config)
        data = pickle.dumps(sd)
        restored = pickle.loads(data)  # noqa: S301
        assert isinstance(restored, _StackedDict)
        assert restored.to_dict() == sd.to_dict()

    def test_native_pickle_round_trip_smooth(self, function_system_config):
        """stdlib pickle.dumps/loads round-trip on smooth _StackedDict."""
        sd = make_smooth(function_system_config)
        data = pickle.dumps(sd)
        restored = pickle.loads(data)  # noqa: S301
        assert restored.to_dict() == sd.to_dict()

    def test_native_pickle_preserves_default_factory_none(self, function_system_config):
        """__reduce__ preserves default_factory=None."""
        sd = make_strict(function_system_config)
        restored = pickle.loads(pickle.dumps(sd))  # noqa: S301
        assert restored.default_factory is None

    def test_native_pickle_preserves_default_factory_class(
        self, function_system_config
    ):
        """__reduce__ preserves default_factory=_StackedDict."""
        sd = make_smooth(function_system_config)
        restored = pickle.loads(pickle.dumps(sd))  # noqa: S301
        assert restored.default_factory is _StackedDict

    def test_native_pickle_preserves_class_type(self, function_system_config):
        """Reconstructed object is the exact same class as the original."""
        sd = make_strict(function_system_config)
        restored = pickle.loads(pickle.dumps(sd))  # noqa: S301
        assert type(restored) is type(sd)

    @pytest.mark.parametrize("protocol", [2, 3, 4, 5])
    def test_native_pickle_all_protocols(self, function_system_config, protocol):
        """__reduce__ works with all pickle protocols."""
        sd = make_strict(function_system_config)
        restored = pickle.loads(pickle.dumps(sd, protocol=protocol))  # noqa: S301
        assert restored.to_dict() == sd.to_dict()
