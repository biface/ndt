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


@pytest.mark.parametrize("dict_name", ["strict_f_sd", "smooth_f_sd"])
def test_deepcopy_protocol(dict_name, request):
    """copy.deepcopy() goes through __deepcopy__(memo) (regression #127)."""
    dict_source = request.getfixturevalue(dict_name)
    dict_copy = deepcopy(dict_source)
    assert dict_copy == dict_source
    assert type(dict_copy) is type(dict_source)
    assert dict_copy.default_setup == dict_source.default_setup


@pytest.mark.parametrize("dict_name", ["strict_f_sd", "smooth_f_sd"])
def test_deepcopy_mutable_leaf(dict_name, request):
    """Mutable leaf values are copied, not shared (regression #127)."""
    dict_source = request.getfixturevalue(dict_name)
    dict_source[["copy_test", "items"]] = [1, 2]
    for dict_copy in (dict_source.deepcopy(), deepcopy(dict_source)):
        dict_copy[["copy_test", "items"]].append(3)
        assert dict_source[["copy_test", "items"]] == [1, 2]
