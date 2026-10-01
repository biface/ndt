"""
Arity of the tree predicates of ``_HKey`` (#135).

``is_complete_tree``, ``is_perfect_tree`` and ``is_full_tree`` take an arity
``n`` (default 2). The minimum is 2 for complete and perfect, 1 for full.
Nodes created with ``is_root=True`` are not checked.
"""

import pytest

from ndict_tools import StackedValueError
from ndict_tools.tools import _HKey


def _tree(spec, key="root", is_root=True):
    """Build a tree from nested ``{key: {child: {...}}}`` dictionaries."""
    node = _HKey(key, is_root=is_root)
    _add(node, spec)
    return node


def _add(node, spec):
    for key, sub in spec.items():
        _add(node.add_child(key), sub)


CHAIN = _tree({"a": {"b": {"c": {}}}})
SINGLE = _tree({})
PERFECT_TERNARY = _tree(
    {
        "a": {"x": {}, "y": {}, "z": {}},
        "b": {"x": {}, "y": {}, "z": {}},
    }
)
COMPLETE_TERNARY = _tree({"a": {"x": {}, "y": {}, "z": {}}, "b": {"x": {}}, "c": {}})


# ---------------------------------------------------------------------------
# Docstring examples
# ---------------------------------------------------------------------------


def test_docstring_example_complete():
    root = _HKey("a")
    b = root.add_child("b")
    root.add_child("c")
    b.add_child("d")
    assert root.is_complete_tree()


def test_docstring_example_not_complete():
    root = _HKey("a")
    root.add_child("b")
    c = root.add_child("c")
    c.add_child("d")
    assert not root.is_complete_tree()


# ---------------------------------------------------------------------------
# Arity validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, n, minimum",
    [
        ("is_complete_tree", 1, 2),
        ("is_complete_tree", 0, 2),
        ("is_perfect_tree", 1, 2),
        ("is_perfect_tree", -1, 2),
        ("is_full_tree", 0, 1),
        ("is_full_tree", -2, 1),
    ],
)
def test_arity_below_minimum_raises(method, n, minimum):
    with pytest.raises(StackedValueError, match=f"n >= {minimum}"):
        getattr(PERFECT_TERNARY, method)(n=n)


def test_full_tree_accepts_arity_one():
    assert CHAIN.is_full_tree(n=1)
    assert not PERFECT_TERNARY.is_full_tree(n=1)


# ---------------------------------------------------------------------------
# Default arity (binary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["is_complete_tree", "is_perfect_tree", "is_full_tree"]
)
def test_single_node_is_trivially_true(method):
    assert getattr(SINGLE, method)()


@pytest.mark.parametrize("method", ["is_complete_tree", "is_perfect_tree"])
def test_chain_is_not_binary_complete_or_perfect(method):
    assert not getattr(CHAIN, method)()


@pytest.mark.parametrize(
    "method", ["is_complete_tree", "is_perfect_tree", "is_full_tree"]
)
def test_ternary_tree_is_not_binary(method):
    """A node with more than n children makes the tree not n-ary."""
    assert not getattr(PERFECT_TERNARY, method)()


# ---------------------------------------------------------------------------
# Explicit arity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["is_complete_tree", "is_perfect_tree", "is_full_tree"]
)
def test_perfect_ternary_with_n_3(method):
    assert getattr(PERFECT_TERNARY, method)(n=3)


def test_complete_ternary_with_n_3():
    assert COMPLETE_TERNARY.is_complete_tree(n=3)
    assert not COMPLETE_TERNARY.is_perfect_tree(n=3)
    assert not COMPLETE_TERNARY.is_full_tree(n=3)


def test_root_arity_is_not_checked():
    """Three top-level keys under a binary tree: the root is free."""
    tree = _tree(
        {"a": {"x": {}, "y": {}}, "b": {"x": {}, "y": {}}, "c": {"x": {}, "y": {}}}
    )
    assert tree.is_complete_tree()
    assert tree.is_perfect_tree()
    assert tree.is_full_tree()
