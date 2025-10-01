import re

import pytest

from ndict_tools.exception import StackedKeyError
from ndict_tools.tools import _StackedDict, from_dict


class TestFromDict:

    @pytest.mark.parametrize(
        "default_setup, keys",
        [
            ({"indent": 2, "default_factory": None}, []),
            ({"indent": 2, "default_factory": _StackedDict}, ["monitoring"]),
        ],
    )
    def test_from_dict(self, function_system_config, default_setup, keys):
        d = from_dict(function_system_config, _StackedDict, default_setup=default_setup)
        for key in keys:
            d = d[key]
        assert isinstance(d, _StackedDict)
        assert d._default_setup == set(list(default_setup.items()))

    def test_from_already_stacked_dict(self, function_system_config):
        d = from_dict(
            _StackedDict(
                function_system_config,
                default_setup={"indent": 2, "default_factory": None},
            ),
            _StackedDict,
            default_setup={"indent": 2, "default_factory": None},
        )
        assert isinstance(d, _StackedDict)
        assert isinstance(d["global_settings"], _StackedDict)

    def test_default_setup_failed(self, function_system_config):
        with pytest.raises(
            StackedKeyError,
            match=re.escape(
                "The key 'default_setup' must be present in class options : {'none_setup': {}} (key: default_setup)"
            ),
        ):
            from_dict(function_system_config, _StackedDict, none_setup={})
