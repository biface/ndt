from copy import deepcopy

import pytest

from ndict_tools.tools import _StackedDict


@pytest.fixture(scope="function")
def standard_smooth_f_setup():
    return {"indent": 0, "default_factory": _StackedDict}


@pytest.fixture(scope="function")
def standard_strict_f_setup():
    return {"indent": 0, "default_factory": None}


@pytest.fixture(scope="function")
def strict_f_sd(function_system_config, standard_strict_f_setup):
    return _StackedDict(
        deepcopy(function_system_config),
        default_setup=standard_strict_f_setup,
    )


@pytest.fixture(scope="function")
def smooth_f_sd(function_system_config, standard_smooth_f_setup):
    return _StackedDict(
        deepcopy(function_system_config),
        default_setup=standard_smooth_f_setup,
    )


@pytest.fixture(scope="class")
def standard_smooth_c_setup():
    return {"indent": 2, "default_factory": _StackedDict}


@pytest.fixture(scope="class")
def standard_strict_c_setup():
    return {"indent": 2, "default_factory": None}


@pytest.fixture(scope="class")
def strict_c_sd(class_system_config, standard_strict_c_setup):
    return _StackedDict(
        deepcopy(class_system_config),
        default_setup=standard_strict_c_setup,
    )


@pytest.fixture(scope="class")
def smooth_c_sd(class_system_config, standard_smooth_c_setup):
    return _StackedDict(
        deepcopy(class_system_config),
        default_setup=standard_smooth_c_setup,
    )
