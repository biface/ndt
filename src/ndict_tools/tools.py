"""
This module provides an intermediate technical class and tools for manipulating nested dictionaries.

Although this module is hidden from the package's external view, its contents are important. The ``_StackedDict`` object
class orchestrates the basic attributes, functions and methods required to initialize and manage nested dictionaries.

This class could have been eliminated in favor of building all methods and tools into the main module containing the
``NestedDictionary`` object class. However, this choice will enable us to build stacks of different dictionaries in the
future, without necessarily using the properties specific to these dictionaries.
"""

from __future__ import annotations

from collections import defaultdict, deque
from textwrap import indent
from typing import Any, Generator, List, Tuple, Union

from .exception import (
    StackedAttributeError,
    StackedIndexError,
    StackedKeyError,
    StackedTypeError,
    StackedValueError,
)

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
    """
    This recursive function is used to transform a dictionary into a stacked dictionary.

    This function enhances and replaces the previous from_dict() function in the core module of this package.
    It allows you to create an object subclass of a _StackedDict with initialization options if requested and
    attributes to be set.

    :param dictionary: The dictionary to transform
    :type dictionary: dict
    :param class_name: name of the class to return
    :type class_name: object
    :param class_options: options to pass to the class or attributes of the class to be set

        * init : parameters to initialize instances of the class, this should be from ``__init__`` function of the class
        * attributes : attributes to set the class attributes
    :type class_options: dict
    :return: stacked dictionary or of subclasses of _StackedDict
    :rtype: _StackedDict
    :raise StackedKeyError: if attribute called is not an attribute of the class hierarchy.
    """

    # FIXME : Improving _StackedDict and subclasses inner attributes management

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
                    f"The key {attribute} is not present in the class attributes",
                    attribute=attribute,
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

    # indent: int = 0
    # "indent is used to print the dictionary with json indentation"
    # default_setup: list = []

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
        setup = []

        self.indent: int = 0
        "indent is used to print the dictionary with json indentation"
        self.default_setup: list = []
        "default_setup is ued to disseminate default parameters to stacked objects"

        # FIXME : Improving inner class and subclasses

        if "setup" not in kwargs:
            if "indent" not in kwargs:
                raise StackedKeyError("Missing 'indent' arguments", key="indent")
            else:
                ind = kwargs.pop("indent")

            if "default" not in kwargs:
                default = None
            else:
                default = kwargs.pop("default")
            setup = [("indent", ind), ("default_factory", default)]
        else:
            settings = kwargs.pop("setup")
            for key, value in settings.items():
                setup.append((key, value))

        super().__init__(*args, **kwargs)
        self.default_setup = setup
        for key, value in self.default_setup:
            self.__setattr__(key, value)

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
        :raises StackedTypeError: if a nested list is found within the key
        """
        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise StackedTypeError(
                        "Nested lists are not allowed as keys in _StackedDict.",
                        expected_type=str,
                        actual_type=list,
                        path=key[: key.index(sub_key)],
                    )

            # Handle hierarchical keys
            current = self
            for sub_key in key[:-1]:  # Traverse the hierarchy
                if sub_key not in current or not isinstance(
                    current[sub_key], _StackedDict
                ):
                    current[sub_key] = self.__class__(indent=self.indent)
                    current[sub_key].__setattr__(
                        "default_factory", self.default_factory
                    )
                    current[sub_key].__setattr__("default_setup", self.default_setup)
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
        :raises StackedTypeError: if a nested list is found within the key
        """

        if isinstance(key, list):
            # Check for nested lists and raise an error
            for sub_key in key:
                if isinstance(sub_key, list):
                    raise StackedTypeError(
                        "Nested lists are not allowed as keys in _StackedDict.",
                        expected_type=str,
                        actual_type=list,
                        path=key[: key.index(sub_key)],
                    )

            # Handle hierarchical keys
            current = self
            for sub_key in key:
                current = current[sub_key]
            return current

        if isinstance(key, str) and key in self.__dict__.keys():
            return self.__getattribute__(key)
        else:
            return super().__getitem__(key)

    def __delitem__(self, key):
        """
        Override __delitem__ to handle hierarchical keys.

        :param key: key to set
        :type key: object
        :return: None
        :rtype: None
        """
        if isinstance(
            key, list
        ):  # Une liste est interprétée comme une hiérarchie de clés
            current = self
            parents = []
            for sub_key in key[:-1]:  # Parcourt tous les sous-clés sauf la dernière
                parents.append(
                    (current, sub_key)
                )  # Garde une trace des parents pour nettoyer ensuite
                current = current[sub_key]
            del current[key[-1]]  # Supprime la dernière clé
            # Nettoie les parents s'ils deviennent vides
            for parent, sub_key in reversed(parents):
                if not parent[sub_key]:
                    del parent[sub_key]
        else:  # Autres types traités comme des clés simples
            # FIXME : Must not delete an attribute - to be controlled as in __getitem__ and raise an error
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

    def pop(self, key: Union[Any, List[Any]], default=None) -> Any:
        """
        Removes the specified key (or hierarchical key) and returns its value.
        If the key does not exist, returns the default value if provided, or raises a KeyError.

        :param key: The key or hierarchical key to remove.
        :type key: Union[Any, List[Any]]
        :param default: The value to return if the key does not exist.
        :type default: Any
        :return: The value associated with the removed key.
        :rtype: Any
        :raises StackedKeyError: If the key does not exist and no default is provided.
        """
        if isinstance(key, list):
            # Handle hierarchical keys
            current = self
            parents = []  # Track parent dictionaries for cleanup
            for sub_key in key[:-1]:  # Traverse up to the last key
                if sub_key not in current:
                    if default is not None:
                        return default
                    raise StackedKeyError(
                        f"Key path {key} does not exist.", key=key, path=key[:-1]
                    )
                parents.append((current, sub_key))
                current = current[sub_key]

            # Pop the final key
            if key[-1] in current:
                value = current.pop(key[-1])
                # Clean up empty parents
                for parent, sub_key in reversed(parents):
                    if not parent[sub_key]:  # Remove empty dictionaries
                        parent.pop(sub_key)
                return value
            else:
                if default is not None:
                    return default
                raise StackedKeyError(
                    f"Key path {key} does not exist.", key=key[-1], path=key[:-1]
                )
        else:
            # Handle flat keys
            return super().pop(key, default)

    def popitem(self):
        """
        Removes and returns the last item in the most deeply nested dictionary as a (path, value) pair.
        The path is represented as a list of keys leading to the value.
        If the dictionary is empty, raises a KeyError.

        The method follows a depth-first search (DFS) traversal to locate the last item,
        removing it from the nested structure before returning.

        :return: A tuple containing the hierarchical path (list of keys) and the value.
        :rtype: tuple
        :raises StackedIndexError: If the dictionary is empty.
        """
        if not self:  # Handle empty dictionary
            raise StackedIndexError("popitem(): _StackedDict is empty")

        # Initialize a stack to traverse the dictionary
        stack = [(self, [])]  # Each entry is (current_dict, current_path)

        while stack:
            current, path = stack.pop()  # Get the current dictionary and path

            if isinstance(current, dict):  # Ensure we are at a dictionary level
                keys = list(current.keys())
                if keys:  # If there are keys in the current dictionary
                    key = keys[-1]  # Select the last key
                    new_path = path + [key]  # Update the path
                    stack.append((current[key], new_path))  # Continue with this branch
            else:
                # If the current value is not a dictionary, we have reached a leaf
                break

        # Remove the item from the dictionary using the found path
        container = self  # Start from the root dictionary
        for key in path[:-1]:  # Traverse to the parent of the target key
            container = container[key]
        value = container.pop(path[-1])  # Remove the last key-value pair

        return path, value

    def update(self, dictionary: dict = None, **kwargs) -> None:
        """
        Updates a stacked dictionary with key/value pairs from a dictionary or keyword arguments.

        :param dictionary: A dictionary with key/value pairs to update.
        :type dictionary: dict
        :param kwargs: Additional key/value pairs to update.
        :type kwargs: dict
        :return: None
        """
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, _StackedDict):
                    value.indent = self.indent
                    value.default_factory = self.default_factory
                    self[key] = value
                elif isinstance(value, dict):
                    nested_dict = from_dict(
                        value,
                        self.__class__,
                        init={
                            "indent": self.indent,
                        },
                        attributes={
                            "default_factory": self.default_factory,
                        },
                    )
                    self[key] = nested_dict
                else:
                    self[key] = value

        for key, value in kwargs.items():
            if isinstance(value, _StackedDict):
                value.indent = self.indent
                value.default_factory = self.default_factory
                self[key] = value
            elif isinstance(value, dict):
                nested_dict = from_dict(
                    value,
                    self.__class__,
                    init={
                        "indent": self.indent,
                    },
                    attributes={
                        "default_factory": self.default_factory,
                    },
                )
                self[key] = nested_dict
            else:
                self[key] = value

    def is_key(self, key: Any) -> bool:
        """
        Checks if a key exists at any level in the _StackedDict hierarchy using unpack_items().
        This works for both flat keys (e.g., 1) and hierarchical keys (e.g., [1, 2, 3]).

        :param key: A key to check. Can be a single key or a part of a hierarchical path.
        :return: True if the key exists at any level, False otherwise.
        """
        # Normalize the key (convert lists to tuples for uniform comparison)
        if isinstance(key, list):
            raise StackedKeyError("This function manages only atomic keys", key=key)

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
                f"Cannot find the key: {key} in the stacked dictionary", key=key
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
                f"Cannot find the key: {key} in the stacked dictionary", key=key
            )

        return __items_list

    def dict_paths(self):
        """
        Returns a view object for all hierarchical paths in the _StackedDict.
        """
        return DictPaths(self)

    def dfs(self, node=None, path=None) -> Generator[Tuple[List, Any], None, None]:
        """
        Depth-First Search (DFS) traversal of the stacked dictionary.

        This method recursively traverses the dictionary in a depth-first manner.
        It yields each hierarchical path as a list and its corresponding value.

        :param node: The current dictionary node being traversed. Defaults to the root if None.
        :type node: Optional[dict]
        :param path: The current hierarchical path being constructed. Defaults to an empty list if None.
        :type path: Optional[List]
        :return: A generator that yields tuples of hierarchical paths and their corresponding values.
        :rtype: Generator[Tuple[List, Any], None, None]
        """
        if node is None:
            node = self
        if path is None:
            path = []

        for key, value in node.items():
            current_path = path + [key]
            yield (current_path, value)
            if isinstance(value, dict):  # Check if the value is a nested dictionary
                yield from self.dfs(
                    value, current_path
                )  # Recursively traverse the nested dictionary

    def bfs(self) -> Generator[Tuple[Tuple, Any], None, None]:
        """
        Breadth-First Search (BFS) traversal of the stacked dictionary.

        This method iteratively traverses the dictionary in a breadth-first manner.
        It uses a queue to ensure that all nodes at a given depth are visited before moving deeper.

        :return: A generator that yields tuples of hierarchical paths (as tuples) and their corresponding values.
        :rtype: Generator[Tuple[Tuple, Any], None, None]
        """
        queue = deque(
            [((), self)]
        )  # Start with an empty path and the top-level dictionary
        while queue:
            path, current_dict = queue.popleft()  # Dequeue the first dictionary
            for key, value in current_dict.items():
                new_path = path + (key,)  # Extend the path with the current key
                if isinstance(
                    value, _StackedDict
                ):  # Check if the value is a nested _StackedDict
                    queue.append(
                        (new_path, value)
                    )  # Enqueue the nested dictionary with its path
                else:
                    yield new_path, value  # Yield the current path and value

    def height(self) -> int:
        """
        Computes the height of the _StackedDict, defined as the length of the longest path.

        :return: The height of the dictionary.
        :rtype: int
        """
        return max((len(path) for path in self.dict_paths()), default=0)

    def size(self) -> int:
        """
        Computes the size of the _StackedDict, defined as the total number of keys (nodes) in the structure.

        :return: The total number of nodes in the dictionary.
        :rtype: int
        """
        return sum(1 for _ in self.unpacked_items())

    def leaves(self) -> list:
        """
        Extracts the leaf nodes of the _StackedDict.

        :return: A list of leaf values.
        :rtype: list
        """
        return [value for _, value in self.dfs() if not isinstance(value, _StackedDict)]

    def is_balanced(self) -> bool:
        """
        Checks if the _StackedDict is balanced.
        A balanced dictionary is one where the height difference between any two subtrees is at most 1.

        :return: True if balanced, False otherwise.
        :rtype: bool
        """

        def check_balance(node):
            if not isinstance(node, _StackedDict) or not node:
                return 0, True  # Height, is_balanced
            heights = []
            for key in node:
                height, balanced = check_balance(node[key])
                if not balanced:
                    return 0, False
                heights.append(height)
            if not heights:
                return 1, True
            return max(heights) + 1, max(heights) - min(heights) <= 1

        _, balanced = check_balance(self)
        return balanced

    def ancestors(self, value):
        """
        Finds the ancestors (keys) of a given value in the nested dictionary.

        :param value: The value to search for in the nested dictionary.
        :type value: Any
        :return: A list of keys representing the path to the value.
        :rtype: List[Any]
        :raises ValueError: If the value is not found in the dictionary.
        """
        for path, val in self.dfs():
            if val == value:
                return path[
                    :-1
                ]  # Return all keys except the last one (the direct key of the value)

        raise StackedValueError(
            f"Value {value} not found in the dictionary.", value=value
        )


class DictPaths:
    """
    A view object that provides a dict-like interface for accessing hierarchical keys as lists.
    Similar to `dict_keys`, but for hierarchical paths in a _StackedDict.
    """

    def __init__(self, stacked_dict):
        self._stacked_dict = stacked_dict

    def __iter__(self):
        """
        Iterates over all hierarchical paths in the _StackedDict as lists.
        """
        yield from self._iterate_paths(self._stacked_dict)

    def _iterate_paths(self, current_dict, current_path=None):
        """
        Recursively iterates over all hierarchical paths in the _StackedDict.

        This function now records both intermediate nodes and leaves.

        :param current_dict: The current dictionary being traversed.
        :param current_path: The path accumulated so far.
        :yield: A list representing the hierarchical path.
        """
        if current_path is None:
            current_path = []

        for key, value in current_dict.items():
            new_path = current_path + [key]  # Current path including this key
            yield new_path  # Register the node itself

            if isinstance(value, dict) and not isinstance(value, _StackedDict):
                value = _StackedDict(value)  # Convert normal dicts to _StackedDict

            if isinstance(value, _StackedDict):
                yield from self._iterate_paths(value, new_path)  # Continue recursion

    def __len__(self):
        """
        Returns the number of hierarchical paths in the _StackedDict.
        """
        return sum(1 for _ in self)

    def __contains__(self, path) -> bool:
        """
        Checks if a hierarchical path exists in the _StackedDict.

        A path is considered valid if it leads to a stored value or a sub-dictionary.

        :param path: A list representing a hierarchical path.
        :type path: List
        :return: True if the path exists, False otherwise.
        :rtype: bool
        """
        current = self._stacked_dict
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]

        # The path is valid as long as we have reached a valid key, regardless of its type
        return True

    def __repr__(self):
        """
        Returns a string representation of the DictPaths object.
        """
        return f"{self.__class__.__name__}({list(self)})"
