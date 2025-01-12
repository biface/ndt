"""
This module provides tools and class for creating nested dictionaries, since standard python does not have nested
dictionaries.
"""

from __future__ import annotations
from .tools import _StackedDict, from_dict

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
        :type args: Iterable
        :param kwargs: enrichments settings and

            * indent : indentation of the printable nested dictionary (used by json.dumps() function)
            * strict : strict mode (False by default) define default answer to unknown key
        :type kwargs: dict

        Example
        -------

        ``NestedDictionary({'first': 1,'second': {'1': "2:1", '2': "2:2", '3': "3:2"}, 'third': 3, 'fourth': 4})``

        ``NestedDictionary(zip(['first','second', 'third', 'fourth'],
        [1, {'1': "2:1", '2': "2:2", '3': "3:2"}, 3, 4]))``

        ``NestedDictionary([('first', 1), ('second', {'1': "2:1", '2': "2:2", '3': "3:2"}),
        ('third', 3), ('fourth', 4)])``


        """
        indent = 0

        if kwargs and "indent" in kwargs:
            indent = kwargs["indent"]
            del kwargs["indent"]

        if kwargs and "strict" in kwargs:
            if kwargs.pop("strict") is True:
                strict = True
                default_class = None
            else:
                strict = False
                default_class = NestedDictionary
        else:
            strict = False
            default_class = NestedDictionary

        options = {"indent": indent, "strict": strict}
        super().__init__(indent=indent, default=default_class)

        if len(args):
            for item in args:
                if isinstance(item, NestedDictionary):
                    nested = item.deepcopy()
                elif isinstance(item, dict):
                    nested = from_dict(item, NestedDictionary, init=options)
                else:
                    nested = from_dict(dict(item), NestedDictionary, init=options)
                self.update(nested)

        if kwargs:
            nested = from_dict(kwargs, NestedDictionary, init=options)
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
                nested_dict = from_dict(
                    value,
                    NestedDictionary,
                    attributes={
                        "indent": self.indent,
                        "default_factory": self.default_factory,
                    },
                )
                super().update(key=key, value=nested_dict)
            else:
                self[key] = value
