import pytest

from ndict_tools.core import NestedDictionary
from ndict_tools.exception import (
    StackedAttributeError,
    StackedDictionaryError,
    StackedKeyError,
    StackedTypeError,
)
from ndict_tools.tools import _StackedDict, from_dict


@pytest.fixture
def stacked_dict():
    return _StackedDict(default_setup={"indent": 0, "default_factory": None})


def test_shallow_copy_dict(stacked_dict):
    stacked_dict[1] = "Integer"
    stacked_dict[(1, 2)] = "Tuple"
    stacked_dict["2"] = {"first": 1, "second": 2}
    sd_copy = stacked_dict.copy()
    assert sd_copy[1] == "Integer"
    assert sd_copy[(1, 2)] == "Tuple"
    assert isinstance(sd_copy["2"], dict)
    assert not isinstance(sd_copy["2"], _StackedDict)
    sd_copy[1] = "Changed in string"
    assert stacked_dict[1] == "Integer"
    assert sd_copy[1] == "Changed in string"
    assert isinstance(sd_copy["2"]["second"], int)
    stacked_dict["2"]["second"] = "3"
    assert sd_copy["2"]["second"] == "3"
    assert isinstance(sd_copy["2"]["second"], str)


def test_deep_copy_dict(stacked_dict):
    stacked_dict[1] = "Integer"
    stacked_dict[(1, 2)] = "Tuple"
    stacked_dict["2"] = {"first": 1, "second": 2}
    sd_copy = stacked_dict.deepcopy()
    assert sd_copy[1] == "Integer"
    assert sd_copy[(1, 2)] == "Tuple"
    assert isinstance(sd_copy["2"], dict)
    sd_copy[1] = "Changed in string"
    assert stacked_dict[1] == "Integer"
    assert sd_copy[1] == "Changed in string"
    assert isinstance(sd_copy["2"]["second"], int)
    stacked_dict["2"]["second"] = "3"
    assert sd_copy["2"]["second"] == 2
    assert isinstance(sd_copy["2"]["second"], int)
