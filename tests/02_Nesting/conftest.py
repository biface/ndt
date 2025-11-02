import pytest

from ndict_tools import NestedDictionary


@pytest.fixture(scope="function")
def nested_strict_f_setup():
    return {"indent": 0, "default_factory": None}


@pytest.fixture(scope="function")
def nested_smooth_f_setup():
    return {"indent": 0, "default_factory": NestedDictionary}


@pytest.fixture(scope="function")
def strict_f_nd(class_system_config, nested_strict_f_setup):
    return NestedDictionary(class_system_config, default_setup=nested_strict_f_setup)


@pytest.fixture(scope="function")
def smooth_f_nd(class_system_config, nested_smooth_f_setup):
    return NestedDictionary(class_system_config, default_setup=nested_smooth_f_setup)


@pytest.fixture(scope="class")
def nested_strict_c_setup():
    return {"indent": 0, "default_factory": None}


@pytest.fixture(scope="class")
def nested_smooth_c_setup():
    return {"indent": 0, "default_factory": NestedDictionary}


@pytest.fixture(scope="class")
def strict_c_nd(class_system_config, nested_strict_c_setup):
    return NestedDictionary(class_system_config, default_setup=nested_strict_c_setup)


@pytest.fixture(scope="class")
def smooth_c_nd(class_system_config, nested_smooth_c_setup):
    return NestedDictionary(class_system_config, default_setup=nested_smooth_c_setup)


@pytest.fixture(scope="class")
def empty_c_smooth_nd(nested_smooth_c_setup):
    return NestedDictionary({}, default_setup=nested_smooth_c_setup)


@pytest.fixture(scope="class")
def empty_c_strict_nd(nested_strict_c_setup):
    return NestedDictionary({}, default_setup=nested_strict_c_setup)
