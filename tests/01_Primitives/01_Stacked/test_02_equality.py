import pytest

from ndict_tools.tools import _StackedDict


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_eq(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert dict_source == function_system_config


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_eq_stacked_dictionary(
    source_name, function_system_config, standard_smooth_f_setup, request
):
    dict_source = request.getfixturevalue(source_name)
    dictionary = _StackedDict(
        function_system_config, default_setup=standard_smooth_f_setup
    )
    assert dict_source == dictionary

@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_eq(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert dict_source != {}


@pytest.mark.parametrize(
    "source_name, setup_name",
    [
        ("strict_f_sd", "standard_strict_f_setup"),
        ("smooth_f_sd", "standard_smooth_f_setup"),
    ],
)
def test_equality(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source.equal(dictionary)


@pytest.mark.parametrize(
    "source_name, setup_name",
    [
        ("strict_f_sd", "standard_smooth_f_setup"),
        ("smooth_f_sd", "standard_strict_f_setup"),
    ],
)
def test_not_equality(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert not dict_source.equal(dictionary)


@pytest.mark.parametrize(
    "source_name, setup_name",
    [
        ("strict_f_sd", "standard_smooth_f_setup"),
        ("smooth_f_sd", "standard_strict_f_setup"),
    ],
)
def test_similar(source_name, setup_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    default_setup = request.getfixturevalue(setup_name)
    dictionary = _StackedDict(function_system_config, default_setup=default_setup)
    assert dict_source.similar(dictionary)


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_not_similar_with_simple_dict(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.similar(function_system_config)


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_isomorph(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert dict_source.isomorph(function_system_config)


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_isomorph_stacked_dictionary(
    source_name, function_system_config, standard_smooth_f_setup, request
):
    dict_source = request.getfixturevalue(source_name)
    dictionary = _StackedDict(
        function_system_config, default_setup=standard_smooth_f_setup
    )
    assert dict_source.isomorph(dictionary)


@pytest.mark.parametrize("source_name", ["strict_f_sd", "smooth_f_sd"])
def test_not_isomorph(source_name, function_system_config, request):
    dict_source = request.getfixturevalue(source_name)
    assert not dict_source.isomorph(["test", "not", "dict"])
