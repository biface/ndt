"""
Test NestedDictionary updates


"""

from ndict_tools import NestedDictionary

head = {'hat': {'type': "bowl hat", 'color': "blue", 'size': "large"}, 'glasses': None}
top = {'coat': {}, 'jacket': {}, 'vest': {}}
bottom = {'pants': {}, 'socks': {}, 'shoes': {}}

suit = NestedDictionary()


def test_dictionary():
    """
    This test is using yhe default_factory property to initialise a NestedDictionary
    """
    assert isinstance(suit['head'], NestedDictionary)
    assert isinstance(suit['top'], NestedDictionary)


def test_head():
    suit['head'].update(head)
    assert suit['head']['glasses'] is None
    assert suit['head']['hat'].default_factory == NestedDictionary
    assert suit['head']['hat']['color'] == 'blue'
    suit['head']['hat']['color'] = "black"
    assert suit['head']['hat']['color'] == 'black'


def test_top():
    suit['top'].update(top)
    assert isinstance(suit['top']['coat'], NestedDictionary)
    assert isinstance(suit['top']['jacket'], NestedDictionary)
    assert isinstance(suit['top']['vest'], NestedDictionary)


def test_bottom():
    suit['bottom'].update(bottom)
    assert isinstance(suit['bottom']['pants'], NestedDictionary)
    assert isinstance(suit['bottom']['socks'], NestedDictionary)
    suit['bottom']['socks']['color'] = "blue"
    assert suit.is_key('color') is True
    assert suit.key_list('color') == [('head', 'hat', 'color'), ('bottom', 'socks', 'color')]
    assert suit.occurrences('color') == 2
    assert isinstance(suit['bottom']['shoes'], NestedDictionary)
