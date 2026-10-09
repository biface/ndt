"""
Configuration handling (``default_setup``): normalization, setter, propagation.
"""

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StackedAttributeError,
    StackedKeyError,
    StrictNestedDictionary,
)
from ndict_tools.tools import _StackedDict

VARIANTS = [
    (NestedDictionary, NestedDictionary),
    (StrictNestedDictionary, None),
    (SmoothNestedDictionary, SmoothNestedDictionary),
]


def _levels(nd):
    """All _StackedDict levels of ``nd``, root included (no cycles expected)."""
    yield nd
    for value in nd.values():
        if isinstance(value, _StackedDict):
            yield from _levels(value)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls, factory", VARIANTS)
def test_default_configuration(cls, factory):
    nd = cls({"a": {"b": 1}})
    for level in _levels(nd):
        assert level.indent == 0
        assert level.default_factory is factory


@pytest.mark.parametrize("cls", [StrictNestedDictionary, SmoothNestedDictionary])
def test_constructor_does_not_modify_caller_setup(cls):
    setup = {"indent": 3, "default_factory": NestedDictionary}
    nd = cls({"a": {"b": 1}}, default_setup=setup)
    assert setup == {"indent": 3, "default_factory": NestedDictionary}
    assert nd.indent == 3


@pytest.mark.parametrize("cls", [StrictNestedDictionary, SmoothNestedDictionary])
def test_constructor_indent_defaults_to_zero(cls):
    nd = cls(default_setup={"default_factory": NestedDictionary})
    assert nd.indent == 0


@pytest.mark.parametrize("cls, factory", VARIANTS[1:])
def test_constructor_forces_factory(cls, factory):
    nd = cls(default_setup={"indent": 1, "default_factory": NestedDictionary})
    assert nd.default_factory is factory


def test_stacked_dict_requires_setup():
    with pytest.raises(StackedKeyError):
        _StackedDict()


# ---------------------------------------------------------------------------
# Setter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls, factory", VARIANTS)
def test_setter_applies_and_propagates(cls, factory):
    nd = cls({"a": {"b": {"c": 1}}, "d": 2})
    nd.default_setup = {"indent": 4, "default_factory": factory}
    for level in _levels(nd):
        assert level.indent == 4
        assert level.default_factory is factory
        assert level.default_setup == [("indent", 4), ("default_factory", factory)]


@pytest.mark.parametrize("cls, factory", VARIANTS[1:])
def test_setter_keeps_variant_factory(cls, factory):
    nd = cls({"a": {"b": 1}})
    nd.default_setup = {"indent": 2, "default_factory": NestedDictionary}
    for level in _levels(nd):
        assert level.indent == 2
        assert level.default_factory is factory


def test_setter_accepts_pairs_and_does_not_modify_value():
    nd = NestedDictionary({"a": {"b": 1}})
    value = [("indent", 5), ("default_factory", None)]
    nd.default_setup = value
    assert value == [("indent", 5), ("default_factory", None)]
    assert nd["a"].indent == 5
    assert nd["a"].default_factory is None


def test_setter_missing_key_leaves_instance_unchanged():
    nd = NestedDictionary(
        {"a": {"b": 1}}, default_setup={"indent": 1, "default_factory": None}
    )
    with pytest.raises(StackedKeyError):
        nd.default_setup = {"indent": 4}
    assert nd.default_setup == [("indent", 1), ("default_factory", None)]
    assert nd["a"].indent == 1


def test_setter_unknown_attribute_leaves_instance_unchanged():
    nd = NestedDictionary(
        {"a": 1}, default_setup={"indent": 1, "default_factory": None}
    )
    with pytest.raises(StackedAttributeError):
        nd.default_setup = {"indent": 4, "default_factory": None, "unknown": 0}
    assert nd.indent == 1
    assert nd.default_setup == [("indent", 1), ("default_factory", None)]


def test_setter_nested_variant_keeps_its_factory():
    nd = NestedDictionary({"a": 1})
    nd["strict"] = StrictNestedDictionary({"x": {"y": 1}})
    nd.default_setup = {"indent": 3, "default_factory": NestedDictionary}
    for level in _levels(nd["strict"]):
        assert level.indent == 3
        assert level.default_factory is None


def test_setter_self_reference():
    nd = NestedDictionary({"a": {"b": 1}})
    nd["self"] = nd
    nd.default_setup = {"indent": 6, "default_factory": None}
    assert nd.indent == 6
    assert nd["a"].indent == 6


def test_setter_shared_substructure():
    shared = NestedDictionary({"x": 1})
    nd = NestedDictionary({"a": 1})
    nd["left"] = shared
    nd["right"] = shared
    nd.default_setup = {"indent": 2, "default_factory": None}
    assert shared.indent == 2
    assert shared.default_factory is None


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_propagates_to_all_levels_of_inserted_value():
    nd = NestedDictionary(default_setup={"indent": 2, "default_factory": None})
    value = NestedDictionary({"b": {"c": {"d": 1}}})
    nd.update({"a": value})
    assert nd["a"] is value
    for level in _levels(nd):
        assert level.indent == 2
        assert level.default_factory is None
        assert level.default_setup == [("indent", 2), ("default_factory", None)]


def test_update_keeps_inserted_variant_factory():
    nd = NestedDictionary(
        default_setup={"indent": 2, "default_factory": NestedDictionary}
    )
    nd.update(strict=StrictNestedDictionary({"x": {"y": 1}}))
    for level in _levels(nd["strict"]):
        assert level.indent == 2
        assert level.default_factory is None
