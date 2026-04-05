# Changelog — ndict-tools

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-04-05

### Added

- `serialize.py` — new private module providing the serialization
  infrastructure (not part of the public API):
  - `_encode_key` / `_decode_key`: JSON key encoding for non-string keys
    per DD-021 (supports `int`, `float`, `bool`, flat `tuple`,
    flat `frozenset`).
  - `NestedDictionaryEncoder`: `json.JSONEncoder` subclass used by
    `to_json`.
  - `_make_decoder_hook`: factory for the `object_pairs_hook` used by
    `from_json`.
  - `_pickle_dump` / `_pickle_load`: pickle helpers with SHA-256 sidecar
    for integrity verification.
- `_StackedDict.from_dict(cls, dictionary, **class_options)`: new
  `@classmethod` alternative constructor. Inherited by all three public
  variants — `NestedDictionary.from_dict(...)`,
  `StrictNestedDictionary.from_dict(...)`,
  `SmoothNestedDictionary.from_dict(...)` are all available without any
  wrapper in `core.py`. Closes #42.
- `_StackedDict.to_json(path, indent=None)`: serialize to a JSON file.
  Non-string keys are encoded per DD-021. Closes #43.
- `_StackedDict.from_json(cls, path, **class_options)`: reconstruct from a
  JSON file. `@classmethod`, returns an instance of the calling class.
  Closes #43.
- `_StackedDict.to_pickle(path, protocol=None)`: serialize to a pickle
  file alongside a SHA-256 sidecar. Always emits `UserWarning` about
  pickle safety. Closes #44.
- `_StackedDict.from_pickle(cls, path, verify=True)`: reconstruct from a
  pickle file. `verify=True` (default) checks the SHA-256 sidecar and
  raises `StackedValueError` on mismatch. Closes #44.
- `_StackedDict.__reduce__`: native pickle support via the module-level
  `_reconstruct` helper, preserving `default_setup` and `default_factory`
  across the pickle round-trip.
- CI matrix: Python 3.13 added as stable target; Python 3.14 added as
  best-effort (`continue-on-error: true`). Closes #41.

### Changed

- **Python 3.9 support dropped.** Minimum Python version is now 3.10.
  `python_requires >= "3.10"` in `pyproject.toml`. Closes #41.
- `from __future__ import annotations` removed from `tools.py`, `core.py`,
  and `exception.py`. All forward references are now explicitly quoted.
  Closes #38.
- Legacy `typing` generics replaced with built-in equivalents throughout
  `tools.py` and `exception.py`: `Dict→dict`, `List→list`, `Set→set`,
  `Tuple→tuple`. `Union` removed from `exception.py` (unused). Closes #39.
- `_type_name` compatibility helper removed from `exception.py`.
  `type.__name__` is used directly in `StackedTypeError`. Closes #40.
- `black` configured with `--target-version py310` in `tox.ini` to ensure
  consistent formatting across all supported Python versions.

### Deprecated

- **`from_dict` free function** (`from ndict_tools.tools import from_dict`)
  is deprecated since 1.1.0 and will be **removed in 1.5.0**
  (issue #81, milestone v1.5.0).

  Calling the free function now emits a `DeprecationWarning` with an
  explicit removal notice. Migrate to the classmethod:

  ```python
  # Before (deprecated since 1.1.0, removed in 1.5.0)
  from ndict_tools.tools import from_dict
  nd = from_dict(data, NestedDictionary, default_setup={...})

  # After
  nd = NestedDictionary.from_dict(data, default_setup={...})
  ```

  Closes #47.

---

## [1.0.0] — 2026-01-31

First stable release.

### Added

- `_StackedDict` base class extending `collections.defaultdict`.
- `_HKey` tree node with `__slots__`, immutable tuple children, DFS/BFS
  traversal.
- `_Paths` lazy view over all hierarchical paths.
- `_CPaths` compact/factorized view with coverage analysis (`is_covering`,
  `coverage`, `missing_paths`, `uncovered_paths`).
- Public API: `NestedDictionary`, `StrictNestedDictionary`,
  `SmoothNestedDictionary`, `PathsView`, `CompactPathsView`.
- Custom exception hierarchy: `StackedDictionaryError` and six
  specialisations (`StackedKeyError`, `StackedTypeError`,
  `StackedValueError`, `StackedAttributeError`, `StackedIndexError`,
  `NestedDictionaryException`).
- Published on PyPI. CI on GitHub Actions and GitLab CI. Coverage reported
  to codecov.io.

### Deprecated

- `DictPaths` deprecated at instantiation with `DeprecationWarning`.
  Scheduled for removal in 1.2.0. Use `CompactPathsView` or
  `nd.compact_paths()` instead.

---

## [0.8.0] — EoL

End-of-life milestone. Removed deprecated `indent=` and `strict=`
attributes from `NestedDictionary.__init__`. Use `default_setup=` instead.

---

*For the full list of changes, see the
[GitHub issue tracker](https://github.com/biface/ndt/issues).*
