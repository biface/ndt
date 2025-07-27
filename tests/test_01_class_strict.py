"""
Testing class with strict option
================================

Smooth option defines default_factory as the NestedDictionary that is to say :

 * if nd is a NestedDictionary and smooth option is True
 * if nd['b'] exists, with a value for 'a' key
 * if nd['b']['b'] does not exist, an KeyError is raised
"""

import pytest

from ndict_tools import NestedDictionary, StrictNestedDictionary


@pytest.fixture(scope="module")
def nd():
    return NestedDictionary(
        {"a": 1, "b": {"a": 2}}, default_setup={"indent": 0, "default_factory": None}
    )


@pytest.fixture(scope="module")
def snd():
    return StrictNestedDictionary(
        {"a": 1, "b": {"a": 2}},
        default_setup={"indent": 0, "default_factory": NestedDictionary},
    )


# Testing NestedDictionary with strict default factory (None)


def test_class_instance(nd):
    assert isinstance(nd, NestedDictionary)


def test_class_attributes_instances(nd):
    assert isinstance(nd["a"], int)
    assert isinstance(nd["b"]["a"], int)
    assert isinstance(nd[["b", "a"]], int)


def test_class_attributes_values(nd):
    assert nd["a"] == 1
    assert nd["b"]["a"] == 2
    assert nd[["b", "a"]] == 2


def test_nested_strict_option(nd):
    assert hasattr(nd, "default_factory")
    assert nd.default_factory is None


def test_nested_strict_behavior(nd):
    with pytest.raises(KeyError):
        value = nd["b"]["b"]


def test_nested(nd):
    nd_2 = NestedDictionary(nd, default_setup={"indent": 2, "default_factory": None})
    assert nd.indent == 0
    assert nd_2.indent == 2
    assert nd_2["a"] == nd["a"]


# Testing StrictNestedDictionary - default factory is always None


def test_strict_class_none():
    lnd = StrictNestedDictionary()
    assert lnd.default_factory is None
    assert lnd.indent == 0


def test_strict_class_instance(snd):
    assert isinstance(snd, StrictNestedDictionary)
    assert isinstance(snd, NestedDictionary)


def test_strict_class_attributes_instances(snd):
    assert isinstance(snd["a"], int)
    assert isinstance(snd["b"]["a"], int)
    assert isinstance(snd[["b", "a"]], int)


def test_strict_class_attributes_values(snd):
    assert snd["a"] == 1
    assert snd["b"]["a"] == 2
    assert snd[["b", "a"]] == 2


def test_strict_class_option(snd):
    assert hasattr(snd, "default_factory")
    assert snd.default_factory is None


def test_strict_class_behavior(snd):
    with pytest.raises(KeyError):
        value = snd["b"]["b"]


def test_strict_nested(snd):
    nd_2 = StrictNestedDictionary(
        snd, default_setup={"indent": 2, "default_factory": None}
    )
    assert snd.indent == 0
    assert nd_2.indent == 2
    assert nd_2["a"] == snd["a"]
    assert nd_2.default_factory is None
