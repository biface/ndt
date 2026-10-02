"""
Type check on the objects returned by from_json and from_pickle — issue #75.

from_json and from_pickle are annotated ``-> Self``. Their loaders return
``Any``: a pickle file carries its own class, and a JSON document whose root
is not an object is not converted by the decoder hook. Both methods therefore
check the result with ``isinstance(result, cls)`` and raise
``StackedTypeError`` otherwise. Instances of a subclass of ``cls`` are
accepted.

Placed in: tests/02_Nesting/05_Serialization/
Fixtures used: tmp_function_file (global conftest)
"""

import json

import pytest

from ndict_tools import (
    NestedDictionary,
    SmoothNestedDictionary,
    StrictNestedDictionary,
)
from ndict_tools.exception import StackedDictionaryError, StackedTypeError
from ndict_tools.serialize import _pickle_dump

VARIANTS = [NestedDictionary, StrictNestedDictionary, SmoothNestedDictionary]
VARIANT_IDS = [cls.__name__ for cls in VARIANTS]

#: (class written to the file, class used to load it)
MISMATCHES = [
    pytest.param(NestedDictionary, StrictNestedDictionary, id="Nested-as-Strict"),
    pytest.param(NestedDictionary, SmoothNestedDictionary, id="Nested-as-Smooth"),
    pytest.param(StrictNestedDictionary, SmoothNestedDictionary, id="Strict-as-Smooth"),
    pytest.param(SmoothNestedDictionary, StrictNestedDictionary, id="Smooth-as-Strict"),
]


def _make(cls):
    """Small instance of cls; Strict and Smooth force their own factory."""
    return cls.from_dict(
        {"a": {"b": 1}}, default_setup={"indent": 0, "default_factory": None}
    )


def _write_pickle(obj, path):
    """Write obj with its SHA-256 sidecar, as to_pickle does."""
    with pytest.warns(UserWarning):
        _pickle_dump(obj, path)


class TestFromPickleTypeCheck:

    @pytest.mark.parametrize("written,loader", MISMATCHES)
    def test_other_variant_is_rejected(self, written, loader, tmp_function_file):
        """A file holding another variant raises StackedTypeError."""
        path = tmp_function_file / f"{written.__name__}-{loader.__name__}.pkl"
        _write_pickle(_make(written), path)
        with pytest.warns(UserWarning):
            with pytest.raises(StackedTypeError) as exc_info:
                loader.from_pickle(path)
        assert exc_info.value.expected_type is loader
        assert exc_info.value.actual_type is written

    @pytest.mark.parametrize(
        "written", [StrictNestedDictionary, SmoothNestedDictionary]
    )
    def test_subclass_is_accepted(self, written, tmp_function_file):
        """NestedDictionary.from_pickle accepts its subclasses and keeps their type."""
        path = tmp_function_file / f"{written.__name__}-sub.pkl"
        _write_pickle(_make(written), path)
        with pytest.warns(UserWarning):
            restored = NestedDictionary.from_pickle(path)
        assert type(restored) is written

    @pytest.mark.parametrize("loader", VARIANTS, ids=VARIANT_IDS)
    def test_non_dictionary_is_rejected(self, loader, tmp_function_file):
        """A pickled list raises StackedTypeError for every variant."""
        path = tmp_function_file / f"list-{loader.__name__}.pkl"
        _write_pickle([1, 2], path)
        with pytest.warns(UserWarning):
            with pytest.raises(StackedTypeError) as exc_info:
                loader.from_pickle(path)
        assert exc_info.value.actual_type is list

    def test_error_is_a_type_error(self, tmp_function_file):
        """The error stays catchable as TypeError and StackedDictionaryError."""
        path = tmp_function_file / "list-catch.pkl"
        _write_pickle([1, 2], path)
        with pytest.warns(UserWarning):
            with pytest.raises(TypeError) as exc_info:
                NestedDictionary.from_pickle(path)
        assert isinstance(exc_info.value, StackedDictionaryError)
        assert str(path) in str(exc_info.value)


class TestFromJsonTypeCheck:

    @pytest.mark.parametrize("loader", VARIANTS, ids=VARIANT_IDS)
    @pytest.mark.parametrize(
        "document,root_type",
        [
            pytest.param([{"a": 1}], list, id="list"),
            pytest.param("text", str, id="string"),
            pytest.param(3, int, id="number"),
            pytest.param(None, type(None), id="null"),
        ],
    )
    def test_non_object_root_is_rejected(
        self, loader, document, root_type, tmp_function_file
    ):
        """A JSON root that is not an object raises StackedTypeError."""
        path = tmp_function_file / f"root-{root_type.__name__}-{loader.__name__}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(StackedTypeError) as exc_info:
            loader.from_json(path, default_setup={"indent": 0, "default_factory": None})
        assert exc_info.value.expected_type is loader
        assert exc_info.value.actual_type is root_type

    @pytest.mark.parametrize("loader", VARIANTS, ids=VARIANT_IDS)
    def test_object_root_gives_calling_class(self, loader, tmp_function_file):
        """A JSON object is rebuilt as the calling class, whatever wrote it."""
        path = tmp_function_file / f"object-{loader.__name__}.json"
        _make(NestedDictionary).to_json(path)
        restored = loader.from_json(
            path, default_setup={"indent": 0, "default_factory": None}
        )
        assert type(restored) is loader
        assert restored.to_dict() == {"a": {"b": 1}}
