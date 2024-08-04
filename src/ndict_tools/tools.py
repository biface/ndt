"""
This module provides an intermediate technical class and tools for manipulating nested dictionaries.

Although this module is hidden from the package's external view, its contents are important. The ``_StackedDict`` object
class orchestrates the basic attributes, functions and methods required to initialize and manage nested dictionaries.

This class could have been eliminated in favor of building all methods and tools into the main module containing the
``NestedDictionary`` object class. However, this choice will enable us to build stacks of different dictionaries in the
future, without necessarily using the properties specific to these dictionaries.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Union, List, Any, Tuple, Generator
from json import dumps
from .exception import StackedKeyError, StackedAttributeError

"""Internal functions"""


def unpack_items(dictionary: dict) -> Generator:
    """
    This functions de-stacks items from a nested dictionary

    :param dictionary:
    :type dictionary: dict
    :return: generator that yields items from a nested dictionary
    :rtype: Generator
    """
    for key in dictionary.keys():
        value = dictionary[key]
        if hasattr(value, "keys"):
            for stacked_key, stacked_value in unpack_items(value):
                yield (key,) + stacked_key, stacked_value
        else:
            yield (key,), value


def from_dict(dictionary: dict, class_name: object, **class_options) -> _StackedDict:
    """This recursive function is used to transform a dictionary into a stacked dictionary.

    This function enhances and replaces the previous from_dict() function in core module of this package.
    It allows you to create an object subclasses of a _StackedDict with initialization options if requested and
    attributes to be set.

    :param dictionary: dictionary to transform
    :type dictionary: dict
    :param class_name: name of the class to return
    :type class_name: object
    :param class_options: options to pass to the class or attributes of the class to be set

        * init : parameters to initialize instances of the class, this should be from ``__init__`` function of the class
        * attributes : attributes to set the class attributes
    :type class_options: dict
    :return: stacked dictionary or of subclasses of _StackedDict
    :rtype: _StackedDict
    :raise StackedKeyError: if attribute called is not an attribute of the hierarchy of classes
    """

    options = {"indent": 0, "strict": False}

    if "init" in class_options:
        options = class_options["init"]

    dict_object = class_name(**options)

    if "attributes" in class_options:
        for attribute in class_options["attributes"]:
            if hasattr(dict_object, attribute):
                dict_object.__setattr__(
                    attribute, class_options["attributes"][attribute]
                )
            else:
                raise StackedAttributeError(
                    "The key {} is not present in the class attributes".format(
                        attribute
                    )
                )

    for key, value in dictionary.items():
        if isinstance(value, _StackedDict):
            dict_object[key] = value
        elif isinstance(value, dict):
            dict_object[key] = from_dict(value, class_name, **class_options)
        else:
            dict_object[key] = value

    return dict_object


"""Classes section"""


class _StackedDict(defaultdict):
    """
    This class is an internal class for stacking nested dictionaries. This class is technical and is used to manage
    the processing of nested dictionaries. It inherits from defaultdict.
    """

    indent: int = 0
    "indent is used to print the dictionary with json indentation"

    def __init__(self, *args, **kwargs):
        """
        At instantiation, it has two mandatory parameters for its creation:

            * **indent**, which is used to format the object's display.
            * **default**, which initializes the default_factory attribute of its parent class defaultdict.


        These parameters are passed using the kwargs dictionary.

        :param args:
        :type args: iterator
        :param kwargs:
        :type kwargs: dict
        """

        if not ("indent" in kwargs and "default" in kwargs):
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

    def __copy__(self) -> _StackedDict:
        """
        This internal function is used to create a shallow copy of a stacked dictionary.
        :return: a shallow copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        new = self.__class__(indent=self.indent, default=self.default_factory)
        for key, value in self.items():
            new[key] = value
        return new

    def __deepcopy__(self) -> _StackedDict:
        """
        This internal function is used to create a complete copy of a stacked dictionary.
        :return: a complete copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        return from_dict(
            self.to_dict(),
            self.__class__,
            init={"indent": self.indent, "default": self.default_factory},
        )

    def unpacked_items(self) -> Generator:
        """
        This method de-stacks items from a nested dictionary. It calls internal unpack_items() function.

        :return: generator that yields items from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield key, value

    def unpacked_keys(self) -> Generator:
        """
        This method de-stacks keys from a nested dictionary and return them as keys. It calls internal unpack_items()
        function.

        :return: generator that yields keys from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield key

    def unpacked_values(self) -> Generator:
        """
        This method de-stacks values from a nested dictionary and return them as values. It calls internal
        unpack_items() function.

        :return: generator that yields values from a nested dictionary
        :rtype: Generator
        """
        for key, value in unpack_items(self):
            yield value

    def to_dict(self) -> dict:
        """
        This method converts a nested dictionary to a classical dictionary

        :return: a dictionary
        :rtype: dict
        """
        unpacked_dict = {}
        for key in self.keys():
            if isinstance(self[key], _StackedDict):
                unpacked_dict[key] = self[key].to_dict()
            else:
                unpacked_dict[key] = self[key]
        return unpacked_dict

    def copy(self) -> _StackedDict:
        """
        This method copies stacked dictionaries to a copy of the dictionary.
        :return: a shallow copy of the dictionary
        :rtype: _StackedDict: a _StackedDict of subclasses of _StackedDict
        """
        return self.__copy__()

    def deepcopy(self) -> _StackedDict:
        """
        This method copies a stacked dictionaries to a deep copy of the dictionary.

        :return: a deep copy of the dictionary
        :rtype: _StackedDict: a _StackedDict of subclasses of _StackedDict
        """

        return self.__deepcopy__()

    def update(self, **kwargs):
        """
        Updates a stacked dictionary with key/value pairs.

        :param kwargs: key/value pairs where values are _StackedDict instances.
        :type kwargs: dict
        :return: None
        :raise StackedKeyError: if any of the key/value pairs cannot be updated:
        :raise KeyError: if key/value are missing or invalid.
        """
        if "key" in kwargs and "value" in kwargs:
            if isinstance(kwargs["value"], _StackedDict):
                self[kwargs["key"]] = kwargs["value"]
            else:
                raise StackedKeyError(
                    "Cannot update a stacked dictionary with an invalid key/value types"
                )
        else:
            raise KeyError("Malformed dictionary parameters key and value are missing")

    def is_key(self, key: Any) -> bool:
        """
        Checks if a key is stacked or not.

        :param key: A possible key in a stacked dictionary.
        :type key: Any
        :return: True if key is a stacked key, False otherwise
        :rtype: bool
        """
        __flag = False
        for keys in self.unpacked_keys():
            if key in keys:
                __flag = True
        return __flag

    def occurrences(self, key: Any) -> int:
        """
        Returns the Number of occurrences of a key in a stacked dictionary including 0 if the key is not a keys in a
        stacked dictionary.

        :param key: A possible key in a stacked dictionary.
        :type key: Any
        :return: Number of occurrences or 0
        :rtype: int
        """
        __occurrences = 0
        for stacked_keys in self.unpacked_keys():
            if key in stacked_keys:
                for occ in stacked_keys:
                    if occ == key:
                        __occurrences += 1
        return __occurrences

    def key_list(self, key: Any) -> list:
        """
        returns the list of unpacked keys containing the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).

        :param key: a possible key in a stacked dictionary.
        :type key: Any
        :return: A list of unpacked keys containing the key from the stacked dictionary.
        :rtype: list
        :raise StackedKeyError: if a key is not in a stacked dictionary.
        """
        __key_list = []

        if self.is_key(key):
            for keys in self.unpacked_keys():
                if key in keys:
                    __key_list.append(keys)
        else:
            raise StackedKeyError(
                "Cannot find the key : {} in a stacked dictionary : ".format(key)
            )

        return __key_list

    def items_list(self, key: Any) -> list:
        """
        returns the list of unpacked items associated to the key from the stacked dictionary. If the key is not in the
        dictionary, it raises StackedKeyError (not a key).

        :param key: a possible key in a stacked dictionary.
        :type key: Any
        :return: A list of unpacked items associated the key from the stacked dictionary.
        :rtype: list
        :raise StackedKeyError: if a key is not in a stacked dictionary.
        """
        __items_list = []

        if self.is_key(key):
            for items in self.unpacked_items():
                if key in items[0]:
                    __items_list.append(items[1])
        else:
            raise StackedKeyError(
                "Cannot find the key : {} in a stacked dictionary : ".format(key)
            )

        return __items_list
