import re
import pytest
from jinja2.runtime import async_exported

from ndict_tools.tools import _StackedDict


@pytest.mark.parametrize("test_dict, keys, factory, setup", [
    ("strict_f_sd", [],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_c_sd", [],
     None, [("indent", 2), ("default_factory", None)]),
    ("strict_f_sd", [frozenset(['cache', 'redis'])],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [frozenset(['cache', 'redis'])],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_f_sd", [frozenset(['cache', 'redis']), 'config'],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_c_sd", ["monitoring", ('logs', 'level')],
     _StackedDict, [("indent", 2), ("default_factory", _StackedDict)]),
])
def test_attributes(test_dict, keys, factory, setup, request):
    d = request.getfixturevalue(test_dict)
    for key in keys:
        d = d[key]
    assert isinstance(d, _StackedDict)
    if factory:
        assert d.default_factory == factory
    else:
        assert d.default_factory is None
    assert d.default_setup == setup


@pytest.mark.parametrize("test_dict, keys, factory, setup", [
    ("strict_f_sd", [],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_c_sd", [],
     None, [("indent", 2), ("default_factory", None)]),
    ("strict_f_sd", [frozenset(['cache', 'redis'])],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [frozenset(['cache', 'redis'])],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_f_sd", [frozenset(['cache', 'redis']), 'config'],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_c_sd", ["monitoring", ('logs', 'level')],
     _StackedDict, [("indent", 2), ("default_factory", _StackedDict)]),
])
def test_default_dict_setup(test_dict, keys, factory, setup, request):
    d = request.getfixturevalue(test_dict)
    for key in keys:
        d = d[key]
    assert isinstance(d, _StackedDict)
    assert d.default_setup == setup
    d.default_setup = {"indent": 10, "default_factory": d.default_factory}
    assert d.default_setup == [("indent", 10), ("default_factory", d.default_factory)]

@pytest.mark.parametrize("test_dict, keys, factory, setup", [
    ("strict_f_sd", [],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_c_sd", [],
     None, [("indent", 2), ("default_factory", None)]),
    ("strict_f_sd", [frozenset(['cache', 'redis'])],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [frozenset(['cache', 'redis'])],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_f_sd", [frozenset(['cache', 'redis']), 'config'],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_c_sd", ["monitoring", ('logs', 'level')],
     _StackedDict, [("indent", 2), ("default_factory", _StackedDict)]),
])
def test_default_list_setup(test_dict, keys, factory, setup, request):
    d = request.getfixturevalue(test_dict)
    for key in keys:
        d = d[key]
    assert isinstance(d, _StackedDict)
    assert d.default_setup == setup
    d.default_setup = [("indent", 10), ("default_factory", d.default_factory), ("indent", 5)]
    # first ("indent", 10) will be removed as it is a set
    assert d.default_setup == [("indent", 5), ("default_factory", d.default_factory)]


@pytest.mark.parametrize("test_dict, keys, factory, setup", [
    ("strict_f_sd", [],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_c_sd", [],
     None, [("indent", 2), ("default_factory", None)]),
    ("strict_f_sd", [frozenset(['cache', 'redis'])],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_f_sd", [frozenset(['cache', 'redis'])],
     _StackedDict, [("indent", 0), ("default_factory", _StackedDict)]),
    ("strict_f_sd", [frozenset(['cache', 'redis']), 'config'],
     None, [("indent", 0), ("default_factory", None)]),
    ("smooth_c_sd", ["monitoring", ('logs', 'level')],
     _StackedDict, [("indent", 2), ("default_factory", _StackedDict)]),
])
def test_default_set_setup(test_dict, keys, factory, setup, request):
    d = request.getfixturevalue(test_dict)
    for key in keys:
        d = d[key]
    assert isinstance(d, _StackedDict)
    assert d.default_setup == setup
    d.default_setup = {("indent", 10), ("default_factory", d.default_factory)}
    assert d.default_setup == [("indent", 10), ("default_factory", d.default_factory)]