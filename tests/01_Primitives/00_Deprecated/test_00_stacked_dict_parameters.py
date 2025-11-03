# These tests must be suppressed in version 1.2 of the package - managing deprecated parameters


import re
import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(expected_error):
            dp = _StackedDict(**parameters)


class TestWarningMessage:

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
    def test_stacked_dict_raises_warning(self, parameters, expected):
        with pytest.warns(
            DeprecationWarning,
            match=re.escape(
                "indent and default parameters are obsolete since version 0.8.0"
                "and will be remove in version 1.2.0. Use kwargs default_setup dictionary instead"
            ),
        ):
            dp = _StackedDict(**parameters)
            for attribute, value in expected:
                assert dp.__getattribute__(attribute) == value
