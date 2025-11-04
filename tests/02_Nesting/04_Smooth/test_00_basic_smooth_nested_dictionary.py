import pytest


class TestSmoothNestedDictionary:

    def test_default_setup(self, smooth_c_snd):
        assert smooth_c_snd.default_factory is not None
        assert smooth_c_snd.indent == 3
