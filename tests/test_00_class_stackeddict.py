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


# These tests must be suppressed in vers 1.2 of the package - managing deprecated parameters


@pytest.mark.parametrize(
    "parameters, expected, error, expected_error",
    [
        (
            {"indent": 10, "default": None},
            [("indent", 10), ("default_factory", None)],
            False,
            None,
        ),
        ({"indent": 10}, [("indent", 10), ("default_factory", None)], False, None),
        (
            {"indent": 10, "default": _StackedDict},
            [("indent", 10), ("default_factory", _StackedDict)],
            False,
            None,
        ),
        (
            {"default": None},
            [("indent", 10), ("default_factory", None)],
            True,
            StackedKeyError,
        ),
    ],
)
def test_deprecated_parameters(parameters, expected, error, expected_error):
    if not error:
        dp = _StackedDict(**parameters)
        for attribute, value in expected:
            assert dp.__getattribute__(attribute) == value
    else:
        with pytest.raises(expected_error):
            dp = _StackedDict(**parameters)


# End of deprecated parameters tests



def test_unused_error():
    e = StackedDictionaryError("This is an unused class", 1000)
    assert str(e) == "This is an unused class"
    assert e.error_code == 1000


@pytest.mark.parametrize("parameters", [{}, {"indent": 0}, {"default_factory": None}])
def test_stacked_dict_init_error(parameters):
    with pytest.raises(StackedKeyError):
        _StackedDict(default_setup=parameters)


def test_stacked_dict_init_success(stacked_dict):
    assert isinstance(stacked_dict, _StackedDict)
    assert stacked_dict.indent == 0
    assert hasattr(stacked_dict, "default_factory")
    assert stacked_dict.default_factory is None


def test_stacked_dict_any_keys(stacked_dict):
    stacked_dict[1] = "integer"
    stacked_dict[(1, 2)] = "tuple"
    assert stacked_dict[1] == "integer"
    assert stacked_dict[(1, 2)] == "tuple"


def test_stacked_dict_typeerror_key_dict(stacked_dict):
    with pytest.raises(TypeError):
        assert stacked_dict[{1, 2}] == "dict"
        assert stacked_dict[[1, [1, 2]]] == "dict"


def test_from_dict():
    nd = from_dict(
        {1: "first", 2: {"first": 1, "second": 2}, 3: 3},
        NestedDictionary,
        default_setup={"indent": 2, "default_factory": None},
    )
    assert isinstance(nd, NestedDictionary)
    assert nd.indent == 2
    assert nd.default_factory is None


def test_unpacked_values(stacked_dict):
    stacked_dict[1] = "first"
    stacked_dict[2] = {"first": 1, "second": 2}
    stacked_dict[3] = 3
    assert list(stacked_dict.unpacked_keys()) == [
        (1,),
        (2, "first"),
        (2, "second"),
        (3,),
    ]
    assert list(stacked_dict.unpacked_values()) == ["first", 1, 2, 3]


def test_from_nested_dict():
    nd = from_dict(
        {1: "first", 2: {"first": 1, "second": 2}, 3: 3},
        NestedDictionary,
        default_setup={"indent": 2, "default_factory": NestedDictionary},
    )
    nd2 = from_dict(
        {1: nd, 2: {"first": 1, "second": 2}, 3: 3},
        NestedDictionary,
        default_setup={"indent": 4, "default_factory": None},
    )
    assert isinstance(nd2, NestedDictionary)
    assert nd2.indent == 4
    assert nd2.default_factory is None
    assert isinstance(nd2[1], NestedDictionary)
    assert nd2[1].default_factory is NestedDictionary
    assert nd2[1].indent == 2


def test_from_dict_attribute_error():
    with pytest.raises(StackedAttributeError):
        from_dict(
            {1: "first", 2: {"first": 1, "second": 2}, 3: 3},
            NestedDictionary,
            default_setup={"indent": 0, "default_factory": None, "factor": True},
        )
    with pytest.raises(StackedKeyError):
        from_dict({1: "first", 2: {"first": 1, "second": 2}, 3: 3}, NestedDictionary)


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
