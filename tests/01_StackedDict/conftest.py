import pytest

from ndict_tools.tools import _StackedDict
from copy import deepcopy

@pytest.fixture(scope="function")
def strict_f_sd(function_system_config):
    return _StackedDict(deepcopy(function_system_config), default_setup={"indent": 0, "default_factory": None})

@pytest.fixture(scope="function")
def smooth_f_sd(function_system_config):
    return _StackedDict(deepcopy(function_system_config), default_setup={"indent": 0, "default_factory": _StackedDict})

@pytest.fixture(scope="class")
def strict_c_sd(class_system_config):
    return _StackedDict(deepcopy(class_system_config), default_setup={"indent": 2, "default_factory": None})

@pytest.fixture(scope="class")
def smooth_c_sd(class_system_config):
    return _StackedDict(deepcopy(class_system_config), default_setup={"indent": 2, "default_factory": _StackedDict})