import re
from copy import copy, deepcopy

import pytest


@pytest.mark.parametrize("dict_name", ["strict_f_sd", "smooth_f_sd"])
def test_copy_dict(dict_name, request):
    dict_source = request.getfixturevalue(dict_name)
    dict_copy = dict_source.copy()
    dict_copy_f = copy(dict_source)
    assert dict_copy == dict_source
    assert dict_copy_f == dict_source
    assert dict_copy == dict_copy_f


@pytest.mark.parametrize("dict_name", ["strict_f_sd", "smooth_f_sd"])
def test_deepcopy_dict(dict_name, request):
    dict_source = request.getfixturevalue(dict_name)
    dict_copy = dict_source.deepcopy()
    assert dict_copy == dict_source


@pytest.mark.parametrize(
    "dict_name, path, value",
    [
        ("strict_f_sd", ["global_settings", "security", "encryption"], "optional"),
        ("smooth_f_sd", ["global_settings", "security", "encryption"], "optional"),
    ],
)
def test_copy_dict_change(dict_name, path, value, request):
    dict_source = request.getfixturevalue(dict_name)
    dict_copy = dict_source.copy()
    dict_copy_f = copy(dict_source)
    assert dict_copy == dict_source
    assert dict_copy_f == dict_source
    assert dict_copy == dict_copy_f
    dict_source[path] = value
    assert dict_copy[path] == value
    assert dict_copy_f[path] == value


@pytest.mark.parametrize(
    "dict_name, path, value",
    [
        ("strict_f_sd", ["global_settings", "security", "encryption"], "optional"),
        ("smooth_f_sd", ["global_settings", "security", "encryption"], "optional"),
    ],
)
def test_deepcopy_dict_change(dict_name, path, value, request):
    dict_source = request.getfixturevalue(dict_name)
    dict_copy = dict_source.deepcopy()
    assert dict_copy == dict_source
    dict_source[path] = value
    assert dict_copy[path] != value
