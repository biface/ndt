"""
This module provides tools and class for creating nested dictionaries, since standard python does not have nested
dictionaries.
"""

from __future__ import annotations

from .tools import _StackedDict

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

        indent = kwargs.pop("indent", 0)
        strict = kwargs.pop("strict", False)
        default_setup = kwargs.pop("default_setup", None)
        default_class = None if strict else NestedDictionary

        if not default_setup:
            default_setup = {"indent": indent, "default_factory": default_class}

        super().__init__(
            *args,
            **kwargs,
            default_setup=default_setup,
        )


class StrictNestedDictionary(NestedDictionary):
    """
    Strict nested dictionary class.

    This class is designed to implement a non-default answer to an unknown key.
    """

    def __init__(self, *args, **kwargs):
        """
        This function initializes a strict nested dictionary as it overwrites the default_factory attributes at None.
        """

        setup = kwargs.pop("default_setup", None)
        if setup:
            setup["indent"] = setup.pop("indent", 0)
            setup["default_factory"] = None

        else:
            setup = {"indent": 0, "default_factory": None}

        super().__init__(*args, **kwargs, default_setup=setup)


class SmoothNestedDictionary(NestedDictionary):
    """
    Smooth nested dictionary class.

    This class is designed to implement a default answer as an empty SmoothNestedDictionary to an unknown key.
    """

    def __init__(self, *args, **kwargs):
        """
        This function initializes a smooth nested dictionary as it overwrites the default_factory attributes at
        SmoothNestedDictionary.
        """

        setup = kwargs.pop("default_setup", None)
        if setup:
            setup["indent"] = setup.pop("indent", 0)
            setup["default_factory"] = SmoothNestedDictionary

        else:
            setup = {"indent": 0, "default_factory": SmoothNestedDictionary}

        super().__init__(*args, **kwargs, default_setup=setup)
