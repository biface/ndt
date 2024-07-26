import pytest

from ndict_tools.tools import _StackedDict
from ndict_tools.exception import StackedKeyError


def test_stacked_dict():
    with pytest.raises(StackedKeyError):
        sd = _StackedDict()
        sd = _StackedDict(indent=0)
        sd = _StackedDict(default=None)
    sd = _StackedDict(indent=0, default=None)
    assert isinstance(sd, _StackedDict)
    assert sd.indent == 0
    assert hasattr(sd, 'default_factory')
    assert sd.default_factory is None
