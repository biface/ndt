from __future__ import annotations

"""
Python module for creating nested dictionaries, since standard python does not have nested dictionaries.
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


"""Classes section"""


class _StackedDict(defaultdict):
    """
    Internal upper class to stacked nested dictionaries.

    This class is technical and is used to manage the processing of nested dictionaries.
    """
    indent: int = 0
    "indent is used to print dictionary with json indent."

    def __init__(self, *args, **kwargs):

        if not ('indent' in kwargs and 'default' in kwargs):
            raise StackedKeyError("Missing 'indent' or 'default' arguments")
        else:
            indent = kwargs.pop("indent")
            default = kwargs.pop("default")
            super().__init__(*args, **kwargs)
            self.indent = indent
            self.default_factory = default

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
        :rtype: generator
        """
        for key, value in unpack_items(self):
            yield key, value

    def unpacked_keys(self):
        """
        De-stacking keys from a nested dictionary

        :return: generator that yields keys from a nested dictionary
        :rtype: generator
        """
        for key, value in unpack_items(self):
            yield key

    def unpacked_values(self):
        """
        De-stacking values from a nested dictionary

        :return: generator that yields values from a nested dictionary
        :rtype: generator
        """
        for key, value in unpack_items(self):
            yield value

    def to_dict(self) -> dict:
        """
        Converts a nested dictionary to a dictionary

        :return: a simple dictionary
        :rtype: dict
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
        Returns the number of occurrences of a key in a stacked dictionary including 0 if the key is not a keys in a
        stacked dictionary.

        :param key: A possible key in a stacked dictionary.
        :type key: str
        :return: Number of occurrences of a key or 0
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
