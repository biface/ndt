"""
This module provides the **core technical infrastructure** for manipulating nested dictionaries. While hidden from the
package's public API, it serves as the foundation for all nested dictionary operations.

The ``_StackedDict`` class is the **central engine** of the ``ndict_tools`` package. It implements all the fundamental
attributes, methods, and logic required to initialize, manage, and manipulate nested dictionaries. This class is
designed to:

- Orchestrate the **basic building blocks** of nested dictionary functionality
- Provide the **complete toolset** for dictionary nesting, key management, and hierarchical data operations
- Serve as a **versatile base** for current and future dictionary implementations

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


def compare_dict(d1, d2) -> bool:
    """
    Compare two (possibly nested) structures: dicts, lists, tuples, sets, or scalars.
    Returns True if they are the same type and have equal content recursively.
    """
    if type(d1) != type(d2):
        return False
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            return False
        for k in d1:
            if not compare_dict(d1[k], d2[k]):
                return False
        return True
    elif isinstance(d1, (list, tuple, set)):
        if len(d1) != len(d2):
            return False
        return all(compare_dict(x, y) for x, y in zip(d1, d2))
    else:
        return d1 == d2


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
    :param class_options: default settings to pass to class instance to be set up.
    :type class_options: dict
    :return: stacked dictionary or of subclasses of _StackedDict
    :rtype: _StackedDict
    :raise StackedKeyError: if attribute called is not an attribute of the class hierarchy.
    """

    if "default_setup" in class_options:
        dict_object = class_name(**class_options)
    else:
        raise StackedKeyError(
            f"The key 'default_setup' must be present in class options : {class_options}",
            key="default_setup",
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

    def __init__(self, *args, **kwargs):
        """
        At instantiation, it has two mandatory parameters for its creation:

            * **indent**, which is used to format the object's display.
            * **default_factory**, which initializes the ``default_factory`` attribute of its parent class ``defaultdict``.
            * these mandatory parameters are stored in ``default_setup`` attribute to be propagated.

        These parameters are passed using the kwargs dictionary.

        :param args:
        :type args: iterator
        :param kwargs:
        :type kwargs: dict
        """

        ind: int = 0
        default = None
        setup = set()

        # Initialize instance attributes

        self.indent: int = 0
        "indent is used to print the dictionary with json indentation"
        self._default_setup: set = set()
        "default_setup is ued to disseminate default parameters to stacked objects"

        # Manage init parameters
        settings = kwargs.pop("default_setup", None)

        if settings is None:
            if "indent" not in kwargs:
                raise StackedKeyError("Missing 'indent' arguments", key="indent")
            else:
                ind = kwargs.pop("indent")

            if "default" not in kwargs:
                default = None
            else:
                default = kwargs.pop("default")
            setup = {("indent", ind), ("default_factory", default)}
        else:
            if not "indent" in settings.keys():
                print("verifed")
                raise StackedKeyError(
                    "Missing 'indent' argument in default settings", key="indent"
                )
            if not "default_factory" in settings.keys():
                raise StackedKeyError(
                    "Missing 'default_factory' argument in default settings",
                    key="default_factory",
                )

            for key, value in settings.items():
                setup.add((key, value))

        # Initializing instance

        super().__init__()
        self._default_setup = setup
        for key, value in self._default_setup:
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                # You cannot initialize undefined attributes
                raise StackedAttributeError(
                    f"The key {key} is not an attribute of the {self.__class__} class.",
                    attribute=key,
                )

        # Update dictionary

        if len(args):
            for item in args:
                if isinstance(item, self.__class__):
                    nested = item.deepcopy()
                elif isinstance(item, dict):
                    nested = from_dict(
                        item, self.__class__, default_setup=dict(self.default_setup)
                    )
                else:
                    nested = from_dict(
                        dict(item),
                        self.__class__,
                        default_setup=dict(self.default_setup),
                    )
                self.update(nested)

        if kwargs:
            nested = from_dict(
                kwargs,
                self.__class__,
                default_setup=dict(self.default_setup),
            )
            self.update(nested)

    @property
    def default_setup(self) -> list:
        """Return a deterministic, list-based view of the internal setup set.
        Order: 'indent', 'default_factory', then other keys sorted alphabetically.
        """
        priority = ["indent", "default_factory"]
        # Convert internal set of tuples to dict to deduplicate and access by key
        d = {k: v for (k, v) in self._default_setup}
        ordered: list = []
        for p in priority:
            if p in d:
                ordered.append((p, d[p]))
        remaining = sorted(
            [(k, v) for (k, v) in self._default_setup if k not in priority],
            key=lambda kv: kv[0],
        )
        ordered.extend(remaining)
        return ordered

    @default_setup.setter
    def default_setup(self, value) -> None:
        """Accept dict, list[tuple], or set[tuple] and store internally as a set."""
        if isinstance(value, dict):
            items = value.items()
        else:
            items = value
        self._default_setup = set(items)

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

        new = self.__class__(default_setup=dict(self.default_setup))
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
            self.to_dict(), self.__class__, default_setup=dict(self.default_setup)
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
                    current[sub_key] = self.__class__(
                        default_setup=dict(self.default_setup)
                    )
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

        # if isinstance(key, str) and key in self.__dict__.keys():
        #    return self.__getattribute__(key)
        # else:
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
            super().__delitem__(key)

    def __eq__(self, other):
        """
        Override __eq__ to compare only structural content, not configuration.
        Two instances are equal if they are the same class and their expanded dict
        representations are equal, regardless of indent/default_factory settings.
        """
        if not isinstance(other, type(self)):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def __ne__(self, other):
        """
        Override __ne__ to handle hierarchical keys.
        """
        return not self.__eq__(other)

    def similar(self, other):
        """
        Override __eq__ to handle hierarchical keys.
        """
        if not isinstance(other, _StackedDict):
            return False
        if self._default_setup != other._default_setup:
            return False
        return compare_dict(self.to_dict(), other.to_dict())

    def isomorph(self, other):
        """
        Override __eq__ to handle hierarchical keys.
        """
        if not isinstance(other, (dict, _StackedDict)):
            return False
        return compare_dict(self.to_dict(), dict(other))

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
                        value, self.__class__, default_setup=dict(self.default_setup)
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
                    value, self.__class__, default_setup=dict(self.default_setup)
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

    def dict_paths(self) -> DictPaths:
        """
        Returns a view object for all hierarchical paths in the _StackedDict.
        """
        return DictPaths(self)

    def dict_search(self) -> DictSearch:
        """
        Returns a DictSearch object factorizing the hierarchical paths of the _StackedDict.
        """
        return DictSearch.from_dict_paths(self.dict_paths())

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

    def __eq__(self, other) -> bool:
        """
        Compares two DictPaths objects (or iterables of paths) for set-wise equality, ignoring order.

        Two DictPaths are considered equal if they contain exactly the same paths, regardless of ordering.
        A path is normalized to a tuple of elements to allow comparison and hashing.
        """
        # Allow comparing with another DictPaths or any iterable of paths
        try:
            self_set = {tuple(p) for p in self}
            other_iter = other if not isinstance(other, DictPaths) else iter(other)
            other_set = {tuple(p) for p in other_iter}
            return self_set == other_set
        except TypeError:
            return NotImplemented

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self):
        """
        Returns a string representation of the DictPaths object.
        """
        return f"{self.__class__.__name__}({list(self)})"

    def dict_search(self) -> DictSearch:
        """
        Returns a list of tuples representing the hierarchical paths in the _StackedDict.
        """
        return DictSearch.from_dict_paths(self)


class DictSearch:
    """
    A factorized representation of paths from a DictPaths object.

    DictSearch groups paths that share common prefixes into nested list structures,
    providing a compact representation of hierarchical dictionary traversal patterns.
    The relationship between DictPaths and DictSearch is bijective when constructed
    from a complete DictPaths, but DictSearch can also represent partial coverage
    for selective tree traversal operations.

    :param structure: List of factorized paths, or None for empty structure
    :type structure: list or None

    Examples:
        >>> # Complete coverage from DictPaths
        >>> paths = [['a'], ['b'], ['b', 'c'], ['b', 'd']]
        >>> search = DictSearch.from_dict_paths(MockDictPaths(paths))
        >>> # Results in: [['a'], ['b', ['c', 'd']]]

        >>> # Manual partial coverage
        >>> partial = DictSearch([['b', ['c']]])  # Only covers ['b'] and ['b', 'c']
    """

    def __init__(self, structure=None):
        """
        Initialize a DictSearch instance.

        :param structure: List of factorized paths, where each path can be either
                         a simple list of keys or a list ending with a sublist of children
        :type structure: list or None
        """
        self.structure = structure if structure is not None else []

    def __repr__(self):
        """
        Return string representation of the mock DictPaths.

        :return: String representation showing contained paths
        :rtype: str
        """
        """
        Return string representation of the DictSearch object.

        :return: String representation showing the factorized structure
        :rtype: str
        """
        return f"DictSearch({self.structure})"

    def __iter__(self):
        """
        Iterate over the stored paths.

        :return: Iterator over the paths
        :rtype: iterator
        """
        """
        Return iterator over the factorized structure elements.

        :return: Iterator over structure elements
        :rtype: iterator
        """
        return iter(self.structure)

    def __len__(self):
        """
        Return the number of top-level elements in the factorized structure.

        :return: Number of top-level structure elements
        :rtype: int
        """
        return len(self.structure)

    def __getitem__(self, index):
        """
        Get structure element at specified index.

        :param index: Index of the structure element to retrieve
        :type index: int
        :return: Structure element at the given index
        :rtype: list
        """
        return self.structure[index]

    @classmethod
    def from_dict_paths(cls, dict_paths):
        """
        Construct a DictSearch from a DictPaths object by factorizing common prefixes.

        This method analyzes all paths in the DictPaths and groups those sharing
        common prefixes into a nested structure. Paths with the same prefix are
        factorized with their child keys collected into sublists.

        :param dict_paths: DictPaths instance to factorize
        :type dict_paths: DictPaths
        :return: New DictSearch instance with factorized structure
        :rtype: DictSearch

        Examples:
            >>> paths = MockDictPaths([['a'], ['b'], ['b', 'c'], ['b', 'd']])
            >>> search = DictSearch.from_dict_paths(paths)
            >>> search.structure  # [['a'], ['b', ['c', 'd']]]
        """
        # Convertir l'itérateur en liste pour pouvoir manipuler les chemins
        paths_list = list(dict_paths)

        if not paths_list:
            return cls([])

        # Trier les chemins par longueur pour faciliter le regroupement
        sorted_paths = sorted(paths_list, key=len)

        # Structure pour construire l'arbre factorisé
        structure = []
        processed = set()

        for path in sorted_paths:
            path_tuple = tuple(path)
            if path_tuple in processed:
                continue

            # Chercher tous les chemins qui ont ce chemin comme préfixe
            children_keys = []

            for other_path in sorted_paths:
                other_tuple = tuple(other_path)
                if (
                    other_tuple != path_tuple
                    and len(other_path) > len(path)
                    and other_path[: len(path)] == path
                ):

                    # La prochaine clé après le préfixe actuel
                    next_key = other_path[len(path)]
                    if next_key not in children_keys:
                        children_keys.append(next_key)
                    processed.add(other_tuple)

            if children_keys:
                # Créer une structure avec la liste des enfants à la fin
                structure.append(path + [children_keys])
            else:
                # Chemin terminal sans enfants
                structure.append(path)

            processed.add(path_tuple)

        return cls(structure)

    def to_dict_paths_list(self):
        """
        Convert the factorized DictSearch back to a flat list of paths.

        This method expands the factorized structure back into individual paths,
        enabling bijective conversion between DictSearch and DictPaths representations.
        Each nested structure is recursively expanded to generate all possible paths.

        :return: List of expanded paths compatible with DictPaths
        :rtype: list[list]

        Examples:
            >>> search = DictSearch([['a'], ['b', ['c', 'd']]])
            >>> search.to_dict_paths_list()
            [['a'], ['b'], ['b', 'c'], ['b', 'd']]
        """
        paths = []

        def expand_structure(item, current_path=[]):
            if isinstance(item, list) and len(item) > 0:
                # Vérifier si le dernier élément est une liste (enfants)
                if isinstance(item[-1], list):
                    # C'est un nœud avec enfants
                    prefix = current_path + item[:-1]
                    paths.append(prefix[:])  # Ajouter le chemin du nœud parent

                    # Développer récursivement chaque enfant
                    for child_key in item[-1]:
                        child_path = prefix + [child_key]
                        expand_structure(child_key, child_path)
                else:
                    # C'est un chemin simple
                    paths.append(current_path + item)
            else:
                # Élément simple (cas des enfants individuels)
                if current_path:  # Ne pas ajouter si current_path est vide
                    paths.append(current_path)

        for item in self.structure:
            expand_structure(item)

        return paths

    def is_complete_coverage(self, dict_paths):
        """
        Check if this DictSearch provides complete coverage of the given DictPaths.

        Complete coverage means that expanding this DictSearch yields exactly the same
        set of paths as the original DictPaths, confirming bijective relationship.

        :param dict_paths: DictPaths instance to compare against
        :type dict_paths: DictPaths
        :return: True if coverage is complete (bijective), False otherwise
        :rtype: bool

        Examples:
            >>> paths = MockDictPaths([['a'], ['b']])
            >>> complete = DictSearch.from_dict_paths(paths)
            >>> complete.is_complete_coverage(paths)  # True
            >>> partial = DictSearch([['a']])
            >>> partial.is_complete_coverage(paths)   # False
        """
        expanded_paths = self.to_dict_paths_list()
        expanded_set = {tuple(path) for path in expanded_paths}
        original_set = {tuple(path) for path in dict_paths}

        return expanded_set == original_set

    def is_partial_coverage(self, dict_paths):
        """
        Check if this DictSearch represents partial coverage of the given DictPaths.

        Partial coverage means that this DictSearch covers a proper subset of the
        paths in the original DictPaths. This enables selective tree traversal
        operations on specific branches of the hierarchical structure.

        :param dict_paths: DictPaths instance to compare against
        :type dict_paths: DictPaths
        :return: True if coverage is partial (proper subset), False otherwise
        :rtype: bool

        Examples:
            >>> paths = MockDictPaths([['a'], ['b'], ['c']])
            >>> partial = DictSearch([['a'], ['b']])
            >>> partial.is_partial_coverage(paths)  # True
            >>> complete = DictSearch.from_dict_paths(paths)
            >>> complete.is_partial_coverage(paths)  # False
        """
        expanded_paths = self.to_dict_paths_list()
        expanded_set = {tuple(path) for path in expanded_paths}
        original_set = {tuple(path) for path in dict_paths}

        return expanded_set.issubset(original_set) and expanded_set != original_set

    def covers_path(self, path):
        """
        Check if a specific path is covered by this DictSearch.

        This method determines whether the given path would be included when
        expanding this DictSearch structure. Useful for testing membership
        without full expansion.

        :param path: List representing a hierarchical path to test
        :type path: list
        :return: True if the path is covered by this DictSearch, False otherwise
        :rtype: bool

        Examples:
            >>> search = DictSearch([['a'], ['b', ['c']]])
            >>> search.covers_path(['a'])      # True
            >>> search.covers_path(['b'])      # True
            >>> search.covers_path(['b', 'c']) # True
            >>> search.covers_path(['d'])      # False
        """
        path_tuple = tuple(path)
        expanded_paths = self.to_dict_paths_list()
        return path_tuple in {tuple(p) for p in expanded_paths}
