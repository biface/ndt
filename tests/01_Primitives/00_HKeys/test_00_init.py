import pytest

class TestHKeySimpleInit:

    def test_alone(self, root):
        assert not root.has_children()

    def test_init(self, root, children):
        for child in children:
            root.add_child(child)
        assert root.has_children()

    @pytest.mark.parametrize("key", ["a","b","g","i","m","n","o","r","s","t","w","x","z"])
    def test_children(self, root, key):
        assert root.get_child(key).is_root is False


    @pytest.mark.parametrize("key", ["c","d","e","f","h","j","k","l","p","q","u","v","y"])
    def test_readd_child(self, root, key):
        root.add_child(key)
        assert root.get_child(key) is not None


class TestHkeyListInit:
    def test_init(self, root, children):
        root.add_children(list(children))
        assert root.has_children()
        assert len(root) ==  26

    def test_keys_list(self, root, children):
        assert root.get_child_keys() == list(children)

    @pytest.mark.parametrize("key", ["a","b","g","i","m","n","o","r","s","t","w","x","z"])
    def test_children(self, root, key):
        assert root.get_child(key).is_root is False

    @pytest.mark.parametrize("key", ["c", "d", "e", "f", "h", "j", "k", "l", "p", "q", "u", "v", "y"])
    def test_readd_child(self, root, children, key):
        root.add_children(list(children))
        assert root.get_child(key) is not None