"""
Testing class with strict option
================================

Smooth option defines default_factory as the NestedDictionary that is to say :

 * if nd is a NestedDictionary and smooth option is True
 * if nd['b'] exists, with a value for 'a' key
 * if nd['b']['b'] does not exist, an KeyError is raised
"""
import pytest

from ndict_tools import NestedDictionary

nd = NestedDictionary(strict=True)
nd['a'] = 1
nd['b'] = NestedDictionary(strict=True)
nd['b']['a'] = 2


def test_class_instance():
    assert isinstance(nd, NestedDictionary)


def test_class_attributes_instances():
    assert isinstance(nd['a'], int)
    assert isinstance(nd['b']['a'], int)


def test_class_attributes_values():
    assert nd['a'] == 1
    assert nd['b']['a'] == 2


def test_strict_option():
    assert hasattr(nd, 'default_factory')
    assert nd.default_factory is None


def test_strict_behavior():
    with pytest.raises(KeyError):
        value = nd['b']['b']
