import pytest

from ndict_tools import NestedDictionary


@pytest.fixture
def ref_smooth_nd():
    return NestedDictionary(
        {"1": 1, "2": {"1": "2:1", "2": "2:2", "3": "3:2"}, "3": 3, "4": 4},
        default_setup={
            "indent": 3,
            "default_factory": NestedDictionary,
        },
    )


@pytest.fixture
def ref_strict_nd():
    return NestedDictionary(
        {"1": 1, "2": {"1": "2:1", "2": "2:2", "3": "3:2"}, "3": 3, "4": 4},
        default_setup={
            "indent": 10,
            "default_factory": None,
        },
    )


@pytest.fixture
def ref_mixed_nd():
    return NestedDictionary(
        {
            "first": 1,
            "second": {"1": "2:1", "2": "2:2", "3": "3:2"},
            "third": 3,
            "fourth": 4,
        },
        default_setup={
            "indent": 5,
            "default_factory": True,
        },
    )


@pytest.fixture(scope="function")
def nested_strict_f_setup():
    return {"indent": 3, "default_factory": None}


@pytest.fixture(scope="function")
def nested_smooth_f_setup():
    return {"indent": 3, "default_factory": NestedDictionary}


def test_verify_smooth_ref(ref_smooth_nd):
    assert ref_smooth_nd.default_factory == NestedDictionary
    assert ref_smooth_nd.indent == 3


@pytest.mark.parametrize(
    "source",
    [
        zip(["1", "2", "3", "4"], [1, {"1": "2:1", "2": "2:2", "3": "3:2"}, 3, 4]),
        [("1", 1), ("2", {"1": "2:1", "2": "2:2", "3": "3:2"}), ("3", 3), ("4", 4)],
        [("3", 3), ("1", 1), ("2", {"1": "2:1", "2": "2:2", "3": "3:2"}), ("4", 4)],
    ],
)
def test_smooth_eq_sources(ref_smooth_nd, source, nested_smooth_f_setup):
    source_nd = NestedDictionary(source, default_setup=nested_smooth_f_setup)
    assert ref_smooth_nd == source_nd


@pytest.mark.parametrize(
    "source",
    [
        zip(["1", "2", "3", "5"], [1, {"1": "2:1", "2": "2:2", "3": "3:2"}, 3, 4]),
        [
            ("1", 1),
            ("2", {"1": "2:1", "2": "2:2", "3": "3:2", 4: (4, 2)}),
            ("3", 3),
            ("4", 4),
        ],
        [
            ("3", 3),
            ("1", 1),
            ("2", {"1": "2:1", "2": "2:2", "3": "3:2"}),
            ("4", {1: (4, 1), 2: (4, 2)}),
        ],
    ],
)
def test_smooth_neq_sources(ref_smooth_nd, source, nested_smooth_f_setup):
    source_nd = NestedDictionary(source, default_setup=nested_smooth_f_setup)
    assert ref_smooth_nd != source_nd


def test_verify_strict_ref(ref_strict_nd):
    assert ref_strict_nd.default_factory is None
    assert ref_strict_nd.indent == 10


@pytest.mark.parametrize(
    "source",
    [
        zip(["1", "2", "3", "4"], [1, {"1": "2:1", "2": "2:2", "3": "3:2"}, 3, 4]),
        [("1", 1), ("2", {"1": "2:1", "2": "2:2", "3": "3:2"}), ("3", 3), ("4", 4)],
        [("3", 3), ("1", 1), ("2", {"1": "2:1", "2": "2:2", "3": "3:2"}), ("4", 4)],
    ],
)
def test_strict_sources(ref_strict_nd, source):
    source_nd = NestedDictionary(
        source, default_setup={"indent": 10, "default_factory": None}
    )
    assert source_nd == ref_strict_nd


def test_mixed_sources(ref_mixed_nd):
    mixed_nd = NestedDictionary(
        [("first", 1), ("fourth", 4)],
        third=3,
        second={"1": "2:1", "2": "2:2", "3": "3:2"},
        default_setup={"indent": 5, "default_factory": True},
    )
    assert mixed_nd == ref_mixed_nd
