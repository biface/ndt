import pytest

from ndict_tools import SmoothNestedDictionary


@pytest.fixture(scope="function")
def smooth_f_snd(class_system_config):
    return SmoothNestedDictionary(
        class_system_config, default_setup={"indent": 3, "default_factory": None}
    )


@pytest.fixture(scope="class")
def smooth_c_snd(class_system_config):
    return SmoothNestedDictionary(
        class_system_config,
        default_setup={"indent": 3, "default_factory": SmoothNestedDictionary},
    )


@pytest.fixture(scope="class")
def empty_c_smooth_snd():
    return SmoothNestedDictionary(default_setup={"indent": 3})
