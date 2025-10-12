import pytest

from ndict_tools.tools import _HKey


@pytest.fixture(scope="class")
def children():
    return "".join([chr(lettre) for lettre in range(97, 123)])


@pytest.fixture(scope="class")
def root():
    return _HKey(key=None, is_root=True)


@pytest.fixture(scope="class")
def key_tree(strict_c_sd):
    return _HKey.build_forest(strict_c_sd)
