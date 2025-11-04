import pytest

from ndict_tools import NestedDictionary


class TestCreateNestedDictionary:

    def test_initialize_strict(self, strict_c_nd):
        assert isinstance(strict_c_nd, NestedDictionary)
        assert strict_c_nd.default_factory is None

    def test_initialize_smooth(self, smooth_c_nd):
        assert isinstance(smooth_c_nd, NestedDictionary)
        assert smooth_c_nd.default_factory == NestedDictionary

    @pytest.mark.parametrize(
        "path",
        [
            [("env", "production")],
            [("env", "production"), "database"],
            [("env", "dev"), "database", "instances"],
            [frozenset(["cache", "redis"])],
        ],
    )
    def test_initialize_strict_nested(self, path, strict_c_nd):
        assert isinstance(strict_c_nd[path], NestedDictionary)
        assert strict_c_nd[path].default_factory is None

    @pytest.mark.parametrize(
        "path",
        [
            [("env", "production")],
            [("env", "production"), "database"],
            [("env", "dev"), "database", "instances"],
            [frozenset(["cache", "redis"])],
        ],
    )
    def test_initialize_smooth_nested(self, path, smooth_c_nd):
        assert isinstance(smooth_c_nd[path], NestedDictionary)
        assert smooth_c_nd[path].default_factory == NestedDictionary
