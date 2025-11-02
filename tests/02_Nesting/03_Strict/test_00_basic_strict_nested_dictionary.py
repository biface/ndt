import pytest

from ndict_tools import StrictNestedDictionary


class TestStrictNestedDictionary:

    def test_default_setup(self, strict_c_snd):
        assert strict_c_snd.default_factory is not StrictNestedDictionary
        assert strict_c_snd.indent == 3
