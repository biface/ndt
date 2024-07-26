import pytest
from ndict_tools import NestedDictionary

d = {'1': 1, '2': {'1': "2:1", '2': "2:2", '3': "3:2"}, '3': 3, '4': 4}

ref_smooth_nd = NestedDictionary(d)


def test_verify_smooth_ref():
    assert ref_smooth_nd.default_factory == NestedDictionary
    assert ref_smooth_nd.indent == 0


def test_smooth_zip_source():
    zip_nd = NestedDictionary(zip(['1', '2', '3', '4'], [1, {'1': "2:1", '2': "2:2", '3': "3:2"}, 3, 4]))
    assert zip_nd == ref_smooth_nd


def test_smooth_list_source():
    list_nd = NestedDictionary([('1', 1), ('2', {'1': "2:1", '2': "2:2", '3': "3:2"}), ('3', 3), ('4', 4)])
    assert list_nd == ref_smooth_nd


def test_smooth_unordered_source():
    ulist_nd = NestedDictionary([('3', 3), ('1', 1), ('2', {'1': "2:1", '2': "2:2", '3': "3:2"}), ('4', 4)])
    assert ulist_nd == ref_smooth_nd


ref_strict_nd = NestedDictionary(d, indent=2, strict=True)


def test_verify_strict_ref():
    assert ref_strict_nd.default_factory is None
    assert ref_strict_nd.indent == 2


def test_strict_zip_source():
    zip_nd = NestedDictionary(zip(['1', '2', '3', '4'], [1, {'1': "2:1", '2': "2:2", '3': "3:2"}, 3, 4]),
                              indent=2, strict=True)
    assert zip_nd == ref_strict_nd


def test_strict_list_source():
    list_nd = NestedDictionary([('1', 1), ('2', {'1': "2:1", '2': "2:2", '3': "3:2"}), ('3', 3), ('4', 4)],
                               indent=2, strict=True)
    assert list_nd == ref_smooth_nd


def test_strict_unordered_source():
    ulist_nd = NestedDictionary([('3', 3), ('1', 1), ('2', {'1': "2:1", '2': "2:2", '3': "3:2"}), ('4', 4)],
                                indent=2, strict=True)
    assert ulist_nd == ref_strict_nd


d = {'first': 1, 'second': {'1': "2:1", '2': "2:2", '3': "3:2"}, 'third': 3, 'fourth': 4}

ref_nd = NestedDictionary(d, indent=2, strict=True)


def test_mixed_sources():
    mixed_nd = NestedDictionary([('first', 1), ('fourth', 4)], third = 3, indent=2,
                                second = {'1': "2:1", '2': "2:2", '3': "3:2"}, strict=True)
    assert mixed_nd == ref_nd

