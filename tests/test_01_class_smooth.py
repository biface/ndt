"""
Testing class with smooth option
================================

Smooth option defines default_factory as the NestedDictionary that is to say :

 * if nd is a NestedDictionary and smooth option is True
 * if nd['b'] exists, with a value for 'a' key
 * if nd['b']['b'] does not exist, an empty NestedDictionary is returned
"""
from ndict_tools import NestedDictionary

nd = NestedDictionary()
nd['a'] = 1
nd['b']['a'] = 2


def test_class_instance():
    assert isinstance(nd, NestedDictionary)


def test_class_attributes_instances():
    assert isinstance(nd['a'], int)
    assert isinstance(nd['b']['a'], int)


def test_class_attributes_values():
    assert nd['a'] == 1
    assert nd['b']['a'] == 2


def test_smoot_option():
    assert hasattr(nd, 'default_factory')
    assert nd.default_factory == NestedDictionary


def test_smooth_behavior():
    value = nd['b']['b']
    assert isinstance(nd['b']['b'], NestedDictionary)
