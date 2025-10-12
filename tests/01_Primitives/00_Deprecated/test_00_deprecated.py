# These tests must be suppressed in version 1.2 of the package - managing deprecated parameters

import pytest

from ndict_tools.exception import StackedKeyError
from ndict_tools.tools import _StackedDict


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (
            {"indent": 10, "default": None},
            [("indent", 10), ("default_factory", None)],
        ),
        ({"indent": 10}, [("indent", 10), ("default_factory", None)]),
        (
            {"indent": 10, "default": _StackedDict},
            [("indent", 10), ("default_factory", _StackedDict)],
        ),
    ],
)
def test_deprecated_parameters(parameters, expected):
    dp = _StackedDict(**parameters)
    for attribute, value in expected:
        assert dp.__getattribute__(attribute) == value


@pytest.mark.parametrize(
    "parameters, expected, expected_error",
    [
        (
            {"default": None},
            [("indent", 10), ("default_factory", None)],
            StackedKeyError,
        ),
    ],
)
def test_deprecated_parameters_failed(parameters, expected, expected_error):
    with pytest.raises(expected_error):
        dp = _StackedDict(**parameters)
