"""
Testing class with smooth option
================================

Smooth option defines default_factory as the NestedDictionary that is to say :

 * if nd is a NestedDictionary and smooth option is True
 * if nd['b'] exists, with a value for 'a' key
 * if nd['b']['b'] does not exist, an empty NestedDictionary is returned
"""

import pytest

from ndict_tools import NestedDictionary, SmoothNestedDictionary


@pytest.fixture(scope="module")
def nd():
    return NestedDictionary(
        {"a": 1, "b": {"a": 2}},
        default_setup={"indent": 0, "default_factory": NestedDictionary},
    )


@pytest.fixture(scope="module")
def snd():
    return SmoothNestedDictionary(
        {"a": 1, "b": {"a": 2}}, default_setup={"indent": 0, "default_factory": None}
    )


# Test NestedDictionary with a smooth option (default is NestedDictionary)


def test_class_instance(nd):
    assert isinstance(nd, NestedDictionary)


def test_class_attributes_instances(nd):
    assert isinstance(nd["a"], int)
    assert isinstance(nd["b"]["a"], int)


def test_class_attributes_values(nd):
    assert nd["a"] == 1
    assert nd["b"]["a"] == 2


def test_nested_smoot_option(nd):
    assert hasattr(nd, "default_factory")
    assert nd.default_factory == NestedDictionary


def test_nested_smooth_behavior(nd):
    value = nd["b"]["b"]
    assert isinstance(nd["b"]["b"], NestedDictionary)


# Testing SmoothNestedDictionary where default_factory is always SmoothNestedDictionary


def test_smooth_class_none():
    lnd = SmoothNestedDictionary()
    assert isinstance(lnd, NestedDictionary)
    assert isinstance(lnd, SmoothNestedDictionary)
    assert lnd.default_factory is SmoothNestedDictionary
    assert lnd.indent == 0


def test_smooth_class_instance(snd):
    assert isinstance(snd, SmoothNestedDictionary)
    assert isinstance(snd, NestedDictionary)


def test_smooth_class_attributes_instances(snd):
    assert isinstance(snd["a"], int)
    assert isinstance(snd["b"]["a"], int)


def test_smooth_class_attributes_values(snd):
    assert snd["a"] == 1
    assert snd["b"]["a"] == 2


def test_smooth_nested_option(snd):
    assert hasattr(snd, "default_factory")
    assert snd.default_factory == SmoothNestedDictionary


def test_smooth_nested_behavior(snd):
    value = snd["b"]["b"]
    assert isinstance(snd["b"]["b"], SmoothNestedDictionary)
