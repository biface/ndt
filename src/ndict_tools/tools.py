"""
This module provides an intermediate technical class and tools for manipulating nested dictionaries.

Although this module is hidden from the package's external view, its contents are important. The ``_StackedDict`` object
class orchestrates the basic attributes, functions and methods required to initialize and manage nested dictionaries.

This class could have been eliminated in favor of building all methods and tools into the main module containing the
``NestedDictionary`` object class. However, this choice will enable us to build stacks of different dictionaries in the
future, without necessarily using the properties specific to these dictionaries.
"""

from __future__ import annotations

from textwrap import indent
from collections import defaultdict
from typing import Union, List, Any, Tuple, Generator

from .exception import StackedKeyError, StackedAttributeError

"""Internal functions"""


def unpack_items(dictionary: dict) -> Generator:
    """
    This function de-stacks items from a nested dictionary.

    :param dictionary: Dictionary to unpack.
    :type dictionary: dict
    :return: Generator that yields items from a nested dictionary.
    :rtype: Generator
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            if not value:  # Handle empty dictionaries
                yield (key,), value
            else:  # Recursive case for non-empty dictionaries
                for stacked_key, stacked_value in unpack_items(value):
                    yield (key,) + stacked_key, stacked_value
        else:  # Base case for non-dictionary values
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
        ind: int = 0
        default = None

        if not "indent" in kwargs:
            raise StackedKeyError("Missing 'indent' arguments")
        else:
            ind = kwargs.pop("indent")

        if not "default" in kwargs:
            default = None
        else:
            default = kwargs.pop("default")

        super().__init__(*args, **kwargs)
        self.indent = ind
        self.default_factory = default

    def __str__(self, padding=0) -> str:
        """
        Override __str__ to converts a nested dictionary to a string in json like format

        :param padding: whitespace indentation of dictionary content
        :type padding: int
        :return: a string in json like format
        :rtype: str
        """

        d_str = "{\n"
        padding += self.indent

        for key, value in self.items():
            if isinstance(value, _StackedDict):
                d_str += indent(
                    str(key) + " : " + value.__str__(padding), padding * " "
                )
            else:
                d_str += indent(str(key) + " : " + str(value), padding * " ")
            d_str += ",\n"

        d_str += "}"

        return d_str

    def __copy__(self) -> _StackedDict:
        """
        Override __copy__ to create a shallow copy of a stacked dictionary.

        :return: a shallow copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        new = self.__class__(indent=self.indent, default=self.default_factory)
        for key, value in self.items():
            new[key] = value
        return new

    def __deepcopy__(self) -> _StackedDict:
        """
        Override __deepcopy__ to create a complete copy of a stacked dictionary.

        :return: a complete copy of a stacked dictionary
        :rtype: _StackedDict or a subclass of _StackedDict
        """

        return from_dict(
            self.to_dict(),
            self.__class__,
            init={"indent": self.indent, "default": self.default_factory},
        )

    def __setitem__(self, key, value) -> None:
        """
        Override __setitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :param value: value to set
        :type value: object
        :return: None
        :rtype: None
        :raises TypeError: if a nested list is found within the key
        """
        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise TypeError("Nested lists are not allowed as keys in _StackedDict.")

            # Handle hierarchical keys
            current = self
            for sub_key in key[:-1]:  # Traverse the hierarchy
                if sub_key not in current or not isinstance(current[sub_key], _StackedDict):
                    current[sub_key] = self.__class__(indent=self.indent)
                    current[sub_key].__setattr__("default_factory", self.default_factory)
                current = current[sub_key]
            current[key[-1]] = value
        else:
            # Flat keys are handled as usual
            super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Override __getitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :return: value
        :rtype: object
        :raises TypeError: if a nested list is found within the key
        """
        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise TypeError("Nested lists are not allowed as keys in _StackedDict.")

            # Handle hierarchical keys
            current = self
            for sub_key in key:
                current = current[sub_key]
            return current
        return super().__getitem__(key)

    def __delitem__(self, key):
        """
        Override __delitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :return: None
        :rtype: None
        """
        if isinstance(key, list):  # Une liste est interprétée comme une hiérarchie de clés
            current = self
            parents = []
            for sub_key in key[:-1]:  # Parcourt tous les sous-clés sauf la dernière
                parents.append((current, sub_key))  # Garde une trace des parents pour nettoyer ensuite
                current = current[sub_key]
            del current[key[-1]]  # Supprime la dernière clé
            # Nettoie les parents s'ils deviennent vides
            for parent, sub_key in reversed(parents):
                if not parent[sub_key]:
                    del parent[sub_key]
        else:  # Autres types traités comme des clés simples
            super().__delitem__(key)

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
        Checks if a key exists at any level in the _StackedDict hierarchy using unpack_items().
        This works for both flat keys (e.g., 1) and hierarchical keys (e.g., [1, 2, 3]).

        :param key: A key to check. Can be a single key or a part of a hierarchical path.
        :return: True if the key exists at any level, False otherwise.
        """
        # Normalize the key (convert lists to tuples for uniform comparison)
        if isinstance(key, list):
            raise StackedKeyError("This function manage only atomic keys")

        # Check directly if the key exists in unpacked keys
        return any(key in keys for keys in self.unpacked_keys())

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
