import re

import pytest

from ndict_tools.exception import StackedAttributeError, StackedKeyError
from ndict_tools.tools import _StackedDict


@pytest.fixture
def stacked_c_result(standard_strict_c_setup):
    return _StackedDict(
        zip(
            ["first", "second", "third", "fourth"],
            [1, {"1": "2:1", "2": "2:2", "3": "3:2"}, 3, 4],
        ),
        default_setup=standard_strict_c_setup,
    )


class TestPrioritySetUp:

    def test_priority(self):
        instance = _StackedDict(
            {}, default_setup={"default_factory": None, "indent": 10}
        )
        assert instance.indent == 10
        assert instance.default_factory is None
        assert instance._default_setup == {("default_factory", None), ("indent", 10)}


class TestParametersSD:

    def test_param_smooth_index(self, smooth_c_sd):
        assert isinstance(smooth_c_sd, _StackedDict)
        assert smooth_c_sd.indent == 2

    def test_param_smooth_default_factory(self, smooth_c_sd):
        assert hasattr(smooth_c_sd, "default_factory")
        assert smooth_c_sd.default_factory == _StackedDict

    def test_param_smooth_defaults_setup(self, smooth_c_sd):
        assert hasattr(smooth_c_sd, "default_setup")
        assert smooth_c_sd.default_setup == [
            ("indent", 2),
            ("default_factory", _StackedDict),
        ]

    def test_param_strict_index(self, strict_c_sd):
        assert isinstance(strict_c_sd, _StackedDict)
        assert strict_c_sd.indent == 2

    def test_param_strict_default_factory(self, strict_c_sd):
        assert hasattr(strict_c_sd, "default_factory")
        assert strict_c_sd.default_factory is None

    def test_param_strict_defaults_setup(self, strict_c_sd):
        assert hasattr(strict_c_sd, "default_setup")
        assert strict_c_sd.default_setup == [("indent", 2), ("default_factory", None)]


class TestInitSD:

    # From documentation https://ndict-tools.readthedocs.io/en/latest/usage.html

    def test_init_dict(self, standard_strict_c_setup, stacked_c_result):
        sd = _StackedDict(
            {
                "first": 1,
                "second": {"1": "2:1", "2": "2:2", "3": "3:2"},
                "third": 3,
                "fourth": 4,
            },
            default_setup=standard_strict_c_setup,
        )
        assert sd.indent == 2
        assert sd.default_factory is None
        assert sd == stacked_c_result

    def test_init_zip(self, standard_strict_c_setup, stacked_c_result):
        sd = _StackedDict(
            zip(
                ["first", "second", "third", "fourth"],
                [1, {"1": "2:1", "2": "2:2", "3": "3:2"}, 3, 4],
            ),
            default_setup=standard_strict_c_setup,
        )
        assert sd.indent == 2
        assert sd.default_factory is None
        assert sd == stacked_c_result

    def test_init_list(self, standard_strict_c_setup, stacked_c_result):
        sd = _StackedDict(
            [
                ("first", 1),
                ("second", {"1": "2:1", "2": "2:2", "3": "3:2"}),
                ("third", 3),
                ("fourth", 4),
            ],
            default_setup=standard_strict_c_setup,
        )
        assert sd.indent == 2
        assert sd.default_factory is None
        assert sd == stacked_c_result

    def test_init_unordered_list(self, standard_strict_c_setup, stacked_c_result):
        sd = _StackedDict(
            [
                ("third", 3),
                ("first", 1),
                ("second", {"1": "2:1", "2": "2:2", "3": "3:2"}),
                ("fourth", 4),
            ],
            default_setup=standard_strict_c_setup,
        )
        assert sd.indent == 2
        assert sd.default_factory is None
        assert sd == stacked_c_result

    def test_init_hybrid(self, standard_strict_c_setup, stacked_c_result):
        sd = _StackedDict(
            [("first", 1), ("fourth", 4)],
            third=3,
            second={"1": "2:1", "2": "2:2", "3": "3:2"},
            default_setup=standard_strict_c_setup,
        )
        assert sd.indent == 2
        assert sd.default_factory is None
        assert sd == stacked_c_result


class TestErrorsSD:

    @pytest.mark.parametrize(
        "parameters, error, error_msg",
        [
            (
                {},
                StackedKeyError,
                "Missing 'indent' argument in default settings (key: indent)",
            ),
            (
                {"indent": 0},
                StackedKeyError,
                "Missing 'default_factory' argument in default settings (key: default_factory)",
            ),
            (
                {"default_factory": None},
                StackedKeyError,
                "Missing 'indent' argument in default settings (key: indent)",
            ),
            (
                {"indent": 0, "default_factory": None, "balanced_factory": True},
                StackedAttributeError,
                "The key balanced_factory is not an attribute of the <class 'ndict_tools.tools._StackedDict'> class. (attribute: balanced_factory)",
            ),
        ],
    )
    def test_stacked_dict_init_error(self, parameters, error, error_msg):
        with pytest.raises(error, match=re.escape(error_msg)):
            _StackedDict(default_setup=parameters)
