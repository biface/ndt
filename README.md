![Python](https://img.shields.io/badge/Language-python-green.svg)
![PyPI - Status](https://img.shields.io/pypi/status/ndict-tools)
![PyPI - License](https://img.shields.io/pypi/l/ndict-tools)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ndict-tools)
![Read the Docs](https://img.shields.io/readthedocs/ndict-tools)
![Test](https://github.com/biface/ndt/actions/workflows/python-ci-tests.yaml/badge.svg?branch=master)
![Codecov](https://img.shields.io/codecov/c/github/biface/ndt)
![GitHub Release](https://img.shields.io/github/v/release/biface/ndt)
![PyPI - Version](https://img.shields.io/pypi/v/ndict-tools)

**[Version française disponible](README.fr.md)**

---

# ndict-tools

In standard Python, dictionaries within dictionaries are possible, creating nested data
structures. However, while this functionality exists, Python does not offer native
features to easily search and manage keys and values within complex nested dictionaries.

My research and testing of libraries dedicated to managing nested dictionaries led me to
several solutions, but none fully met my expectations. The module I found that came
closest was one from 2015,
[available on PyPI](https://pypi.org/project/nested_dict/), but it does not provide a
complete architecture for managing "nested dictionary objects" in a smooth and robust
way.

This motivated me to redevelop such a module, offering a more complete and intuitive way
to handle nested dictionaries. This new module makes it easier to manipulate, search,
and manage keys and values in complex data structures by providing tools dedicated to
this specific task.

## What is a Nested Dictionary?

A nested dictionary is simply a dictionary where the values themselves are dictionaries.
This allows for the creation of richer, hierarchical data structures where each "node"
in the structure can hold additional information in the form of dictionaries, making it
easier to model complex data in an organized and accessible way.

## Using Nested Keys and Managing Hierarchies in Dictionaries

### Keys of Different Types and Using Lists for Hierarchical Keys

Just like with standard dictionaries in Python, the keys in a nested dictionary must be
**hashable**. This means you can use types such as **numbers**, **strings**, or even
**tuples** as keys. However, **lists** are not hashable and cannot be used directly as
keys.

### Accessing Nested Values

Nested dictionaries allow you to structure your data over multiple levels. For example,
to access a value in a nested dictionary, you can use a sequence of keys that represents
each level of the hierarchy.

Using simple, non-nested lists is the way used to represent nested keys.

#### Nested Access Example

The following two expressions are **equivalent** for accessing a value in a nested
dictionary:

```python,ignore
dictionary[[1, "a", (2, 3)]]   # is equivalent to
dictionary[1]["a"][(2, 3)]
```

## Installation

```bash
pip install ndict-tools
```

## Documentation

Full documentation is available at
[ndict-tools.readthedocs.io](https://ndict-tools.readthedocs.io/en/latest/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and contribution
guidelines.

## License

This project is licensed under the
[CeCILL-C Free Software License](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html).
