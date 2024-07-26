from __future__ import annotations

"""
Python module for creating nested dictionaries, since standard python does not have nested dictionaries.
"""

from .tools import _StackedDict

"""Internal functions"""


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


class NestedDictionary(_StackedDict):
    """
    Nested dictionary class.

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

        super().__init__(indent=indent, default=default_class)

        if len(args) >= 1:
            for item in args:
                if isinstance(item, dict):
                    nested = from_dict(item)
                    self.update(nested)
                else:
                    nested = from_dict(dict(item))
                    self.update(nested)

        if kwargs:
            nested = from_dict(kwargs)
            self.update(nested)

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
