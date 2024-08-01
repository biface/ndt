import pytest

from ndict_tools.tools import _StackedDict, from_dict
from ndict_tools.core import NestedDictionary
from ndict_tools.exception import StackedKeyError, StackedAttributeError


def test_stacked_dict_init_error():
    with pytest.raises(StackedKeyError):
        _StackedDict()
        _StackedDict(indent=0)
        _StackedDict(default=None)


def test_stacked_dict_init_success():
    sd = _StackedDict(indent=0, default=None)
    assert isinstance(sd, _StackedDict)
    assert sd.indent == 0
    assert hasattr(sd, 'default_factory')
    assert sd.default_factory is None


def test_stacked_dict_any_keys():
    sd = _StackedDict(indent=0, default=None)
    sd[1] = "integer"
    sd[(1, 2)] = "tuple"
    assert sd[1] == "integer"
    assert sd[(1, 2)] == "tuple"


def test_stacked_dict_typeerror_key_list():
    sd = _StackedDict(indent=0, default=None)
    with pytest.raises(TypeError):
        assert sd[[1, 2]] == "list"


def test_stacked_dict_typeerror_key_dict():
    sd = _StackedDict(indent=0, default=None)
    with pytest.raises(TypeError):
        assert sd[{1, 2}] == "dict"


def test_from_dict():
    nd = from_dict({1: 'first', 2: {'first': 1, 'second': 2}, 3: 3}, NestedDictionary,
                   init={'indent': 2, 'strict': True})
    assert isinstance(nd, NestedDictionary)
    assert nd.indent == 2
    assert nd.default_factory is None


def test_from_nested_dict():
    nd = from_dict({1: 'first', 2: {'first': 1, 'second': 2}, 3: 3}, NestedDictionary,
                   init={'indent': 2, 'strict': True})
    nd2 = from_dict({1: nd, 2: {'first': 1, 'second': 2}, 3: 3}, NestedDictionary, init={'indent': 4})
    assert isinstance(nd2, NestedDictionary)
    assert nd2.indent == 4
    assert nd2.default_factory is NestedDictionary
    assert isinstance(nd2[1], NestedDictionary)
    assert nd2[1].default_factory is None
    assert nd2[1].indent == 2


def test_from_dict_attribute_error():
    with pytest.raises(StackedAttributeError):
        from_dict({1: 'first', 2: {'first': 1, 'second': 2}, 3: 3}, NestedDictionary,
                  init={'indent': 2, 'strict': True}, attributes={'factor': True})
