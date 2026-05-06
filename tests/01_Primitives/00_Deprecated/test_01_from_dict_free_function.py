import re
from collections import defaultdict

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StackedTypeError,
    StrictNestedDictionary,
)
from ndict_tools.tools import _StackedDict, from_dict


@pytest.mark.parametrize(
    "class_name",
    [_StackedDict, NestedDictionary, SmoothNestedDictionary, StrictNestedDictionary],
)
def test_from_dict(class_name, function_system_config):
    with pytest.warns(DeprecationWarning, match="1.5.0"):
        dictionary = from_dict(
            function_system_config,
            class_name,
            default_setup={"indent": 1, "default_factory": None},
        )
    assert isinstance(dictionary, class_name)


@pytest.mark.parametrize("class_name", [defaultdict, list])
def test_from_dict_failed(class_name, function_system_config):
    with pytest.raises(
        StackedTypeError,
        match=re.escape(f"class_name must be a _StackedDict class, got <class 'type'>"),
    ):
        with pytest.warns(DeprecationWarning, match="1.5.0"):
            from_dict(
                function_system_config,
                class_name,
                default_setup={"indent": 1, "default_factory": None},
            )
