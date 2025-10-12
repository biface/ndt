"""
Test extensions
"""

import pytest

from ndict_tools.tools import _StackedDict


class BDict(_StackedDict):

    def __init__(self, *args, **kwargs):

        self.balanced = False

        settings = kwargs.pop("default_setup", {})
        settings["indent"] = 4
        settings["default_factory"] = None
        settings["balanced"] = True

        super().__init__(*args, **kwargs, default_setup=settings)


@pytest.fixture
def bdict_test():
    return BDict({1: {1: 11, 2: 12}, 2: {1: 21, 2: 22}})


def test_bdict_class_extension(bdict_test):
    assert bdict_test.balanced == True


def test_bdict_class_default_setup(bdict_test):
    assert bdict_test.default_setup == [
        ("indent", 4),
        ("default_factory", None),
        ("balanced", True),
    ]


def test_bdict_class_nested(bdict_test):
    assert isinstance(bdict_test, BDict)
    assert isinstance(bdict_test[1], BDict)
    assert bdict_test[1].balanced == True
    assert isinstance(bdict_test[2], BDict)
    assert bdict_test[2].balanced == True


def test_bdict_value(bdict_test):
    assert bdict_test[[1, 1]] == 11
    assert bdict_test[1][1] == 11
    assert bdict_test[[2, 1]] == 21
