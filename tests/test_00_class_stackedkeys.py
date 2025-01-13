import pytest
from ndict_tools.tools import _StackedDict

@pytest.fixture
def stacked_dict():
    """Fixture to initialize a _StackedDict with default parameters."""
    return _StackedDict(indent=2, default=None)


def test_hierarchical_key_assignment(stacked_dict):
    """Test: Assigning a value with a list as a hierarchy."""
    stacked_dict[[1, 2]] = "value"
    assert 1 in stacked_dict
    assert 2 in stacked_dict[1]
    assert stacked_dict[1][2] == "value"


def test_flat_key_assignment(stacked_dict):
    """Test: Assigning a value with a tuple as a flat key."""
    stacked_dict[(1, 2)] = "tuple_value"
    assert (1, 2) in stacked_dict
    assert stacked_dict[(1, 2)] == "tuple_value"


def test_mixed_key_access(stacked_dict):
    """Test: Assigning and accessing hierarchical and flat keys."""
    stacked_dict[[1, 2]] = "hierarchical_value"
    stacked_dict[(1, 2)] = "tuple_value"
    assert stacked_dict[[1, 2]] == "hierarchical_value"
    assert stacked_dict[(1, 2)] == "tuple_value"


def test_hierarchical_key_deletion(stacked_dict):
    """Test: Deleting a key with a hierarchy."""
    stacked_dict[[1, 2]] = "value"
    del stacked_dict[[1, 2]]
    assert 1 not in stacked_dict


def test_flat_key_deletion(stacked_dict):
    """Test: Deleting a flat key."""
    stacked_dict[(1, 2)] = "tuple_value"
    del stacked_dict[(1, 2)]
    assert (1, 2) not in stacked_dict


def test_conflicting_keys(stacked_dict):
    """Test: Ensure hierarchical and flat keys do not conflict."""
    stacked_dict[[1, 2]] = "hierarchical_value"
    stacked_dict[(1, 2)] = "tuple_value"
    assert stacked_dict[[1, 2]] == "hierarchical_value"
    assert stacked_dict[(1, 2)] == "tuple_value"


def test_empty_structure_cleaning(stacked_dict):
    """Test: Verify that empty structures are cleaned after deletion."""
    stacked_dict[[1, 2]] = "value"
    del stacked_dict[[1, 2]]
    assert len(stacked_dict) == 0


def test_invalid_key_type(stacked_dict):
    """Test: Ensure a non-hashable key raises a TypeError."""
    with pytest.raises(TypeError):
        stacked_dict[[1, [2, 3]]] = "invalid"


def test_hierarchical_access_error(stacked_dict):
    """Test: Accessing a non-existent hierarchy should raise a KeyError."""
    with pytest.raises(KeyError):
        _ = stacked_dict[[1, 2]]

def test_simple_list_as_hierarchy(stacked_dict):
    """Test: A flat list should create a hierarchical structure."""
    stacked_dict[[1, 2, 3]] = "value"
    assert stacked_dict[1][2][3] == "value"

def test_nested_list_access_error(stacked_dict):
    """Test: Accessing with a nested list should raise a TypeError."""
    stacked_dict[[1, 2, 3]] = "value"
    with pytest.raises(TypeError):
        _ = stacked_dict[[1, [2, 3]]]

def test_nested_list_key_error(stacked_dict):
    """Test: A nested list within the key should raise a TypeError."""
    with pytest.raises(TypeError):
        stacked_dict[[1, [2, 3]]] = "value"


def test_flat_access_error(stacked_dict):
    """Test: Accessing a non-existent flat key should raise a KeyError."""
    with pytest.raises(KeyError):
        _ = stacked_dict[(1, 2)]


def test_subclass_behavior():
    """Test: Verify that subclasses of _StackedDict behave correctly."""

    class CustomStackedDict(_StackedDict):
        pass

    sd = CustomStackedDict(indent=2, default=lambda: None)
    sd[[1, 2]] = "hierarchical_value"
    sd[(1, 2)] = "tuple_value"

    assert isinstance(sd[1], CustomStackedDict)
    assert sd[[1, 2]] == "hierarchical_value"
    assert sd[(1, 2)] == "tuple_value"