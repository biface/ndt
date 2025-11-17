import re
import pytest
from black.trans import defaultdict

from ndict_tools.tools import _StackedDict, from_dict
from ndict_tools import NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary, StackedTypeError


@pytest.mark.parametrize("class_name", [
    _StackedDict, NestedDictionary, SmoothNestedDictionary, StrictNestedDictionary])
def test_from_dict(class_name, function_system_config):
    dictionary = from_dict(function_system_config, class_name, default_setup={'indent': 1, 'default_factory': None})
    assert isinstance(dictionary, class_name)


@pytest.mark.parametrize("class_name", [
    defaultdict, list
])
def test_from_dict_failed(class_name, function_system_config):
    with pytest.raises(StackedTypeError, match=re.escape(f"class_name must be a _StackedDict class, got <class 'type'>")):
        from_dict(function_system_config, class_name, default_setup={'indent': 1, 'default_factory': None})