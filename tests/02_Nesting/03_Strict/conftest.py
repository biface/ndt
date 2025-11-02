import pytest

from ndict_tools import StrictNestedDictionary


@pytest.fixture(scope="function")
def strict_f_snd(class_system_config):
    return StrictNestedDictionary(
        class_system_config,
        default_setup={"indent": 3, "default_factory": StrictNestedDictionary},
    )


@pytest.fixture(scope="class")
def strict_c_snd(class_system_config):
    return StrictNestedDictionary(
        class_system_config,
        default_setup={"indent": 3, "default_factory": StrictNestedDictionary},
    )


@pytest.fixture(scope="class")
def empty_c_strict_snd():
    return StrictNestedDictionary(default_setup={"indent": 3})
