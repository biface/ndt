"""
``CompactPathsView.structure`` setter with a plain dict (#106).
"""

from ndict_tools import CompactPathsView, NestedDictionary


def test_structure_setter_accepts_plain_dict():
    cpaths = CompactPathsView(NestedDictionary({"a": 1}))
    cpaths.structure = {"x": {"y": 1, "z": {"t": 2}}, "w": 3}
    assert cpaths.structure == [["x", "y", ["z", "t"]], "w"]
    assert sorted(cpaths.to_paths()) == sorted(
        [["x"], ["x", "y"], ["x", "z"], ["x", "z", "t"], ["w"]]
    )
