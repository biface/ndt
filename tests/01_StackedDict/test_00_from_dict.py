import pytest

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
