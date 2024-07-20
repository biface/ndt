from __future__ import annotations

"""
Python module for creating nested directories, since standard python does not have nested directories.
"""
from collections import defaultdict
from json import dumps
from .exception import StackedKeyError

"""Internal functions"""


def unpack_items(dictionary: dict):
    """
    de-stacking items from a nested dictionary
    :param dictionary:
    :type dictionary: dict
    :return: generator that yields items from a nested dictionary
    :rtype: generator
    """
    for key in dictionary.keys():
        value = dictionary[key]
        if hasattr(value, 'keys'):
            for stacked_key, stacked_value in unpack_items(value):
                yield (key,) + stacked_key, stacked_value
        else:
            yield (key,), value


def from_dict(dictionary: dict) -> NestedDictionary:
    """
    This recursive function is used to transform a dictionary into a nested dictionary.
    :param dictionary: a dict object, even nested dict., to be transformed into a nested dictionary object class .
    :type dictionary: dict
    :return: a nested dictionary object
    :rtype: NestedDictionary
    """

    nested = NestedDictionary()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            nested[key] = from_dict(value)
        else:
            nested[key] = value
    return nested


"""Classes section"""


class _StackedDict(defaultdict):
    """
    Internal upper class to stacked nested dictionaries.

    This class is technical and is used to manage the processing of nested dictionaries.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = 0

    def __str__(self) -> str:
        """
        Converts a nested dictionary to a string in json format
        :return: a string in json format
        :rtype: str
        """
        return dumps(self.to_dict(), indent=self.indent)

    def unpacked_items(self):
        """
        De-stacking items from a nested dictionary
        :return: generator that yields items from a nested dictionary
        """
        for key, value in unpack_items(self):
            yield key, value

    def unpacked_keys(self):
        """
        De-stacking keys from a nested dictionary
        :return: generator that yields keys from a nested dictionary
        """
        for key, value in unpack_items(self):
            yield key

    def unpacked_values(self):
        """
        De-stacking values from a nested dictionary
        :return: generator that yields values from a nested dictionary
        """
        for key, value in unpack_items(self):
            yield value

    def to_dict(self):
        """
        Converts a nested dictionary to a dictionary
        :return: dict
        """
        unpacked_dict = {}
        for key in self.keys():
            if isinstance(self[key], _StackedDict):
                unpacked_dict[key] = self[key].to_dict()
            else:
                unpacked_dict[key] = self[key]
        return unpacked_dict

    def update(self, **kwargs):
        """
        Updates a stacked dictionary with key/value pairs.
        :param kwargs: key/value pairs where values are _StackedDict instances.
        :type kwargs: dict
        :return: None
        """
        if 'key' in kwargs and 'value' in kwargs:
            if isinstance(kwargs['value'], _StackedDict):
                self[kwargs['key']] = kwargs['value']
            else:
                raise StackedKeyError("Cannot update a stacked dictionary with an invalid key/value types")
        else:
            raise KeyError("Malformed dictionary parameters key and value are missing")

    def is_key(self, key: str) -> bool:
        """
        Checks if a key is stacked or not.
        :param key: A possible key in a stacked dictionary.
        :type key: str
        :return: True if key is a stacked key, False otherwise
        :rtype: bool
        """
        __flag = False
        for keys in self.unpacked_keys():
            if key in keys:
                __flag = True
        return __flag

    def occurrences(self, key: str) -> int:
        """
        Returns the number of occurrences of a key in a stacked dictionary.
        :param key: A possible key in a stacked dictionary.
        :type key: str
        :return: Number of occurrences of a key in a stacked dictionary including 0 if the key is not a keys in a
        stacked dictionary.
        :rtype: int
        """
        __occurrences = 0
        for stacked_keys in self.unpacked_keys():
            if key in stacked_keys:
                for occ in stacked_keys:
                    if occ == key:
                        __occurrences += 1
        return __occurrences

    def key_list(self, key: str) -> list:
        """
        returns the list of unpacked keys containing the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).
        :param key: a possible key in a stacked dictionary.
        :type key: str
        :return: A list of unpacked keys containing the key from the stacked dictionary.
        :rtype: list
        """
        __key_list = []

        if self.is_key(key):
            for keys in self.unpacked_keys():
                if key in keys:
                    __key_list.append(keys)
        else:
            raise StackedKeyError("Cannot find the key : {} in a stacked dictionary : ".format(key))

        return __key_list

    def items_list(self, key: str) -> list:
        """
        returns the list of unpacked items associated to the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).
        :param key: a possible key in a stacked dictionary.
        :type key: str
        :return: A list of unpacked items associated the key from the stacked dictionary.
        :rtype: list
        """
        __items_list = []

        if self.is_key(key):
            for items in self.unpacked_items():
                if key in items[0]:
                    __items_list.append(items[1])
        else:
            raise StackedKeyError("Cannot find the key : {} in a stacked dictionary : ".format(key))

        return __items_list


class NestedDictionary(_StackedDict):
    """
    Nested directory class.

    This class is designed as a stacked dictionary. It represents a nest of dictionaries, that is to say that each
    key is a value or a nested dictionary. And so on...
    """

    def __init__(self, *args, **kwargs):
        """
        This function initializes a nested dictionary.
        :param args: the first one of the list must be a dictionary to instantiate an object.
        :param kwargs: enrichments settings
            * indent : indentation of the printable nested dictionary (used by json.dumps() function)
            * strict : strict mode (False by default) define default answer to unknown key
        :type kwargs: dict
        """
        indent = 0

        if kwargs and 'indent' in kwargs:
            indent = kwargs['indent']
            del kwargs['indent']

        if kwargs and 'strict' in kwargs:
            if kwargs.pop('strict') is True:
                default_class = None
            else:
                default_class = NestedDictionary
        else:
            default_class = NestedDictionary

        super().__init__(**kwargs)

        if len(args) == 1:
            if isinstance(args[0], dict):
                nested = from_dict(args[0])
                self.update(nested)

        self.indent = indent
        self.default_factory = default_class

    def update(self, dictionary: dict) -> None:
        """
        Updates a stacked dictionary with key/value pairs.
        :param dictionary: a simple dict.
        :type dictionary: dict
        :return: None
        """
        for key, value in dictionary.items():
            if isinstance(value, NestedDictionary):
                value.indent = self.indent
                value.default_factory = self.default_factory
                super().update(key=key, value=value)
            elif isinstance(value, dict):
                nested_dict = from_dict(value)
                nested_dict.indent = self.indent
                nested_dict.default_factory = self.default_factory
                super().update(key=key, value=nested_dict)
            else:
                self[key] = value
