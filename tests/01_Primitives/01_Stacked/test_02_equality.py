"""Comparisons of nested dictionaries (DD-031).

``equal()`` and ``==``: same class, same ``default_setup``, same content.
``isomorph()``: same content, any class of the ``_StackedDict`` family.
``similar()``: same content only; plain dicts accepted.
"""

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.tools import _StackedDict

SOURCES = ["strict_f_sd", "smooth_f_sd"]

SAME_SETUP = [
    ("strict_f_sd", "standard_strict_f_setup"),
    ("smooth_f_sd", "standard_smooth_f_setup"),
]

OTHER_SETUP = [
    ("strict_f_sd", "standard_smooth_f_setup"),
    ("smooth_f_sd", "standard_strict_f_setup"),
]


# ``==`` and ``!=``


@pytest.mark.parametrize("source_name, setup_name", SAME_SETUP)
def test_eq_same_class_setup_and_content(
    source_name, setup_name, function_system_config, request
):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source == dictionary
    assert not dict_source != dictionary


@pytest.mark.parametrize("source_name, setup_name", OTHER_SETUP)
def test_eq_other_setup_is_false(
    source_name, setup_name, function_system_config, request
):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert not dict_source == dictionary
    assert dict_source != dictionary


@pytest.mark.parametrize("source_name", SOURCES)
def test_eq_plain_dict_is_false_both_ways(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source == function_system_config
    assert not function_system_config == dict_source
    assert dict_source != function_system_config
    assert function_system_config != dict_source


@pytest.mark.parametrize("source_name", SOURCES)
def test_ne_empty(source_name, request):
    dict_source = request.getfixturevalue(source_name)
    assert dict_source != {}


# ``equal()``


@pytest.mark.parametrize("source_name, setup_name", SAME_SETUP)
def test_equality(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source.equal(dictionary)


@pytest.mark.parametrize("source_name, setup_name", OTHER_SETUP)
def test_not_equality_same_class(
    source_name, setup_name, function_system_config, request
):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert not dict_source.equal(dictionary)


@pytest.mark.parametrize("source_name", SOURCES)
def test_not_equality_simple_dict(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.equal(function_system_config)


def test_equal_requires_the_same_class():
    """equal() is symmetric: an instance of a subclass is never equal to its parent."""

    class Inventory(NestedDictionary):
        pass

    data = {"kitchen": {"lights": "ceiling"}}
    parent, child = NestedDictionary(data), Inventory(data)
    assert parent.default_setup == child.default_setup
    assert not parent.equal(child)
    assert not child.equal(parent)
    assert parent != child
    assert child != parent
    assert parent.isomorph(child) and child.isomorph(parent)


# ``isomorph()``


@pytest.mark.parametrize("source_name, setup_name", OTHER_SETUP)
def test_isomorph_other_setup(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source.isomorph(dictionary)


@pytest.mark.parametrize("source_name", SOURCES)
def test_not_isomorph_with_simple_dict(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.isomorph(function_system_config)


@pytest.mark.parametrize("source_name", SOURCES)
def test_not_isomorph_with_non_dict(source_name, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.isomorph(["test", "not", "dict"])


# ``similar()``


@pytest.mark.parametrize("source_name, setup_name", OTHER_SETUP)
def test_similar_other_setup(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source.similar(dictionary)


@pytest.mark.parametrize("source_name", SOURCES)
def test_similar_with_simple_dict(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert dict_source.similar(function_system_config)


@pytest.mark.parametrize("source_name", SOURCES)
def test_not_similar_with_non_dict(source_name, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.similar(["test", "not", "dict"])


# The three levels on the public classes


@pytest.mark.parametrize(
    "other_class", [StrictNestedDictionary, SmoothNestedDictionary]
)
def test_public_classes_are_isomorphic_not_equal(other_class):
    data = {"kitchen": {"lights": "ceiling", "heating": {"type": "radiator"}}}
    nested, other = NestedDictionary(data), other_class(data)
    assert not nested.equal(other) and not other.equal(nested)
    assert nested != other
    assert nested.isomorph(other) and other.isomorph(nested)
    assert nested.similar(other) and other.similar(nested)
    assert nested.similar(data) and not nested.isomorph(data)


@pytest.mark.parametrize(
    "dict_class", [NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary]
)
def test_different_content_fails_every_comparison(dict_class):
    first = dict_class({"kitchen": {"lights": "ceiling"}})
    second = dict_class({"kitchen": {"lights": "spots"}})
    assert first != second
    assert not first.equal(second)
    assert not first.isomorph(second)
    assert not first.similar(second)
    assert not first.similar({"kitchen": {"lights": "spots"}})
