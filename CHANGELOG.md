# Changelog — ndict-tools

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.3.0] — Unreleased

### Added

- **Inline type information (PEP 561).** The package ships a `py.typed`
  marker, so type checkers use its inline annotations instead of treating
  it as untyped. Downstream stubs for `ndict_tools` are no longer needed.
  Closes #129 ([DD-027](https://github.com/biface/ndt/issues/125)).
- **`verifytypes` tox environment.** Runs `basedpyright --verifytypes` on the
  installed package; the type completeness of the exported API must stay at
  100%. Included in `pre-push` (hence `local`), `check` and `ci-quality`.
- **`typing` tox environment.** Runs `basedpyright tests/typing`, where
  `assert_type` calls check the types inferred for the public API. Unlike the
  `basedpyright src` step, it fails on any error. Included in `basedpyright`,
  `pre-push` (hence `local`), `check` and `ci-quality`. Added with #75.
- **Concepts: "The Forest of Keys".** A new page defines the forest of keys of
  a nested dictionary (vertices, edges, roots, depth, leaves, height and
  levels), matches `len()`, `size()`, `leaves()`, `height()` and
  `occurrences()` with measures of the forest, explains why the `_HKey` tree
  operations apply, defines complete, perfect and full trees for an arity n,
  and states the limits of the model (shared references, cycles). Its
  diagrams are Mermaid flowcharts, rendered by `sphinxcontrib-mermaid` 2.1.1,
  added to the `docs` extra and `docs/source/requirements.txt`; the extension
  pins the Mermaid version it loads in the browser (11.12.1). The EN/FR
  table of tree shapes becomes the translation glossary of `CONTRIBUTING.md`
  and `CONTRIBUTING.fr.md`. Closes #146.

### Changed

- **Python 3.11 is the minimum version.** Python 3.10 support is dropped
  (`requires-python = ">=3.11"`). Closes #74.
- **CI matrix:** Python 3.11, 3.12, 3.13 and 3.14 are required; 3.14 is added
  to the classifiers. Python 3.15 (pre-release) and free-threaded 3.14t run as
  best effort; the 3.14t job only checks that the package runs without the
  GIL and makes no thread-safety claim.
- **Development baseline:** `.python-version` pins 3.11 (`uv python pin 3.11`);
  `pyrightconfig.json` checks against Python 3.11; black targets py311.
- **pytest configuration:** `DeprecationWarning` is raised as an error, so
  deprecated calls surface before a later Python version removes them.
- **`tox.ini`:** `py310` environment removed; `py315` and `py314t` added;
  black commands no longer pass `--target-version` and read it from
  `[tool.black]` in `pyproject.toml`.
- **`@override` (PEP 698)** on the 24 methods that override a base class
  method, through a private `_compat.py` shim: `typing.override` on Python
  3.12+, a local decorator that sets `__override__` on 3.11. No runtime
  dependency is added; `typing_extensions` is used by the type checker only.
  The shim is removed in 1.4.0. Closes #103.
- **Configuration propagation (`default_setup`).** `update()` passes its
  configuration to inserted `_StackedDict` values through the
  `default_setup` setter, so every nested level of the inserted value is
  updated, not only its top level. An inserted `StrictNestedDictionary` or
  `SmoothNestedDictionary` keeps its own `default_factory`. Closes #137.
- **`_StackedDict.__init__`** takes `default_setup` as an explicit
  keyword-only parameter instead of reading it from `**kwargs`. Runtime
  behaviour is unchanged. `NestedDictionary`, `StrictNestedDictionary` and
  `SmoothNestedDictionary` no longer define `__init__`: their defaults and
  forced values go through the `_normalize_setup` class hook, which custom
  subclasses can override.
- **`from_dict()` and `from_json()` without `default_setup`.** The
  alternate constructors resolve the configuration like the constructor,
  through `cls._normalize_setup`: `NestedDictionary.from_dict(d)` and
  `NestedDictionary.from_json(path)` use the default of the class, and the
  Strict and Smooth variants their own. A given `default_setup` is validated
  or forced as before. Only the base `_StackedDict`, which has no default,
  still raises `StackedKeyError`, now with the message of `_normalize_setup`.
  The deprecated `from_dict()` free function follows the same rule.
  Closes #151.
- **Type annotations completed on the exported API** (part of #129, [DD-027](https://github.com/biface/ndt/issues/125)).
  These signatures are now part of the public contract:
  - constructors: `*args: Mapping[Any, Any] | Iterable[tuple[Any, Any]]`,
    `**kwargs: Any`;
  - `__eq__`, `__ne__`, `equal`, `similar`, `isomorph`: `other: object`,
    returning `bool`;
  - `__getitem__`, `__setitem__`, `__delitem__`: `key: Any`;
  - `pop(key, default: Any = None)`, `popitem() -> tuple[list[Any], Any]`,
    `ancestors(value: Any) -> list[Any]`, `update(..., **kwargs: Any)`;
  - `dfs(node: Mapping[Any, Any] | None, path: list[Any] | None)`;
  - `__str__(padding: int = 0)`;
  - `**class_options: Any` on `from_dict`, `from_json`, `from_pickle` and the
    deprecated `from_dict` free function; `compare_dict(d1: Any, d2: Any)`;
  - `CompactPathsView.is_covering`, `coverage`, `missing_paths`,
    `uncovered_paths`: `stacked_dict: _StackedDict`;
  - `NestedDictionaryEncoder.iterencode(...) -> Iterator[str]`.
- **`update()` first argument** is annotated like `dict.update`:
  `SupportsKeysAndGetItem[Any, Any] | Iterable[tuple[Any, Any]] | None`,
  positional-only (`m, /` instead of `__m`). Objects with `keys()` and
  `__getitem__` were already accepted at runtime; the override of
  `MutableMapping.update` is now compatible and its suppression comment is
  removed. Closes #105.
- **`reportPropertyTypeMismatch`** (basedpyright): the `default_setup` and
  `structure` properties keep their asymmetric types on purpose (the setter
  accepts several sources, the getter returns the normalized form). Each
  setter carries a justified, rule-specific suppression, so the rule stays
  active for other properties. Closes #106.
- **`typing.Self` (PEP 673)** as the return type of the methods that build
  an instance of the calling class: `from_dict`, `copy`, `deepcopy`,
  `__copy__` and `__deepcopy__` on `_StackedDict`, and `_HKey.build_forest`.
  `StrictNestedDictionary().copy()` is now typed `StrictNestedDictionary`
  instead of `_StackedDict`. No runtime change for these methods.
- **`from_json` and `from_pickle` check the type of what they load** and
  are also annotated `-> Self`. Their loaders return whatever the file
  holds, so both methods now raise `StackedTypeError` (a `TypeError`) when
  the result is not an instance of the calling class: a pickle file holding
  another variant or a non-dictionary object, or a JSON document whose root
  is not an object. Before, the object was returned as is. Instances of a
  subclass are accepted: `NestedDictionary.from_pickle` still returns a
  pickled `StrictNestedDictionary`. Closes #75 ([DD-022](https://github.com/biface/ndt/issues/94) amendment).
- **`_HKey` tree predicates take an arity `n`, binary by default.**
  `is_complete_tree(n=2)`, `is_perfect_tree(n=2)` and `is_full_tree(n=2)`.
  `is_complete_tree` no longer hard-codes arity 2, and `is_perfect_tree` and
  `is_full_tree` no longer infer it from the first internal node: a ternary
  tree needs `n=3`. `is_full_tree(n=None)` is no longer accepted. An `n` below
  the minimum (2 for complete and perfect, 1 for full) raises
  `StackedValueError`; a node with more than `n` children makes the predicate
  return `False`. Nodes created with `is_root=True` are still not checked.
  Terminology follows English usage (French *complet* is English *perfect*).
- **Test suite hygiene.** The flake8 `per-file-ignores` on `tests/` are
  removed, and what they hid is fixed: three shadowed tests renamed
  (`test_get_depth_by_path`, `test_ne_empty`, `test_dfs_path`; 11 cases that
  never ran now run), unused results inside `pytest.raises` blocks turned
  into bare expressions or used in an assertion, unused imports removed,
  lambdas assigned to names turned into functions, `== None` / `== True` /
  `== False` comparisons replaced. Closes #136.
- **Docstrings** of `NestedDictionary`, `StrictNestedDictionary` and
  `SmoothNestedDictionary` describe the current parameters. The `indent` and
  `strict` keyword settings were removed in 1.2.0 (#114); keyword arguments
  are data.
- **Documentation toolchain** resolved for Python 3.11, with exact versions in
  `docs/source/requirements.txt` and the `docs` extra: Sphinx 9.0.4,
  myst-parser 5.1.0, furo 2025.12.19; `sphinx-intl` 2.4.0 added for the
  translation catalogs. `sphinx-multiversion-contrib` is removed, with its
  extension, its `smv_*` settings and the `versioning.html` sidebar template;
  versioned builds move to a per-tag archive ([DD-028](https://github.com/biface/ndt/issues/126)). The build runs without
  warnings. Closes #130.
- **Read the Docs configuration** moves from `docs/conf/.readthedocs.yaml` to
  `.readthedocs.yaml` at the repository root, the location Read the Docs reads
  by default, and builds with Python 3.11. Read the Docs serves `stable` and
  `latest` only ([DD-028](https://github.com/biface/ndt/issues/126)). Part of #132.
- **GitHub Pages archive.** `docs-ghpages.yml` no longer runs
  sphinx-multiversion on every push to `master`. It runs when a final
  release tag `vX.Y.Z` is pushed, like the package build and publication
  (or by hand through `workflow_dispatch` with `tag` and `python-version`),
  builds that tag once from its own sources and documentation requirements,
  and publishes it under `/vX.Y.Z/` without touching the other directories.
  Release candidates are not archived. The Python
  version comes from the tag's `.readthedocs.yaml`. Translations listed in
  `docs/source/locales/LANGUAGES` go under `/<lang>/vX.Y.Z/`. The landing
  page lists the archived versions and links to Read the Docs `stable` and
  `latest`, replacing the hard-coded redirect to a `v1.2.0/` build that was
  never produced ([DD-028](https://github.com/biface/ndt/issues/126)). Closes #131.
- **French translation of the documentation.** Sphinx gettext catalogs,
  one per source page, in `docs/source/locales/<lang>/LC_MESSAGES/`
  (`locale_dirs`, `gettext_compact = False`, `gettext_location = False`).
  The French catalogs cover the Concepts, the Guide, the API reference and
  the Changelog axis; the full change log stays in English. Untranslated
  strings fall back to English. `docs/source/locales/LANGUAGES` lists `fr`,
  so the GitHub Pages archive publishes it under `/fr/vX.Y.Z/`. On Read the
  Docs the French documentation is a separate project linked as a
  translation; `conf.py` takes the language from `READTHEDOCS_LANGUAGE`.
  The extraction, update, check and build steps are documented in
  `CONTRIBUTING.md` and `CONTRIBUTING.fr.md`
  ([DD-028](https://github.com/biface/ndt/issues/126)). Closes #133.
- **Package metadata (PEP 639).** The licence is declared as an SPDX
  expression (`license = "CECILL-C"`) with its file (`license-files`),
  which replaces the licence classifier; the build requires
  `hatchling>=1.27`. The `Typing :: Typed` classifier and `keywords` are
  added, and `dependencies = []` states that the package has no runtime
  dependency. Part of #130.
- **Comparisons (breaking).** `==` is now strict, like `equal()`: same
  class, same `default_setup` and same content. A nested dictionary is no
  longer equal to a plain `dict` with the same content, in either order;
  `!=` follows. `similar()` and `isomorph()` exchange their behaviour:
  `isomorph()` compares the content of two dictionaries of the
  `_StackedDict` family, whatever their class and configuration, and is
  never true for a plain `dict`; `similar()` compares the content alone and
  accepts a plain `dict`. `equal()` implies `isomorph()`, which implies
  `similar()`. To compare a nested dictionary with a plain `dict`, use
  `similar()` or `to_dict() ==` ([DD-031](https://github.com/biface/ndt/issues/156)). Closes #157.

### Removed

- **Root `requirements.txt`, `requirements.dev.txt` and
  `requirements.test.txt`.** No tool or workflow read them; development
  dependencies are in `.tox-config/requirements/` ([DD-025](https://github.com/biface/ndt/issues/116)). Part of #130.

### Fixed

- **`copy.deepcopy()`** raised `TypeError` on every nested dictionary:
  `__deepcopy__` did not accept the `memo` argument passed by the `copy`
  module. It now implements the protocol, so shared sub-structures stay
  shared in the copy and a dictionary that contains itself no longer
  raises `RecursionError`. Closes #127.
- **`deepcopy()`** shared mutable leaf values (lists, sets, custom objects)
  with the original, because the copy was rebuilt through
  `to_dict()` / `from_dict()`. Every value is now deep-copied.
- **`default_setup` setter** stored the new configuration without applying
  it: `indent` and `default_factory` kept their old values. It now validates
  the configuration like the constructor, applies it and propagates it to
  every nested level; shared sub-structures and self-references are handled.
  On a strict or smooth dictionary, `default_factory` stays forced.
- **`_HKey.is_complete_tree()`** returned `True` for incomplete trees, including
  the example of its own docstring: after the first node with missing children,
  it checked the grandchildren instead of rejecting any later node that has
  children. Closes #135.
- **`CompactPathsView.structure` setter** raised `StackedKeyError` when given
  a plain `dict`, although the docstring lists it as accepted: the dict was
  wrapped without a `default_setup`. It is now wrapped with
  `{'indent': 0, 'default_factory': None}`. Found while working on #106.
- **`StrictNestedDictionary` / `SmoothNestedDictionary`** modified the
  `default_setup` dict passed by the caller. They now work on a copy.
- **`str(CompactPathsView)`** showed the private class name
  (`_CPaths(3 paths): ...`): the prefix was hard-coded. It now uses the name
  of the actual class, like `repr()`. The same defect elsewhere in the public
  API is corrected: the error messages of `popitem()` on an empty dictionary
  and of a nested list in a key name the class of the instance
  (`popitem(): NestedDictionary is empty`), so they follow every subclass;
  the error of the `structure` setter lists what it accepts in public terms;
  and `CompactPathsView.to_compact()` returns a `CompactPathsView`, as
  `PathsView.to_compact()` does, instead of the private class. Closes #138.
- **`PathsView.get_subtree_paths()`** returned wrong paths for any non-empty
  prefix: the key of the prefix node appeared twice
  (`[['a'], ['a', 'a'], ['a', 'a', 'b'], ...]`). The cause was
  `_HKey.get_all_paths()`, which added the key of a non-root node twice. It
  now returns the full path of the node followed by those of its
  descendants, and `get_subtree_paths()` returns that list as is. The
  expected values of the existing tests contained the duplicated key and are
  corrected. Closes #140.
- **`_HKey.is_valid_tree()`** printed debugging traces on standard output
  for every non-root node; the `print` calls are removed. The same method
  and `check_parent_consistency()` tested the parent of a node by its truth
  value, which for `_HKey` is its number of children: a leaf parent counted
  as missing. A node whose parent is a leaf that does not list it was not
  reported, and the message named its parent `None`. Both now compare with
  `None`. Closes #142.
- **`key_list()`** was annotated `-> list[list[Any]]` but returns tuples, as
  its docstring shows. The annotation is now `list[tuple[Any, ...]]`, like the
  paths yielded by `unpacked_keys()`. No runtime change; a type checker now
  sees the actual type. Checked by `tests/typing/check_key_list.py`.
  Closes #143.
- **`CompactPathsView.structure` setter** accepted a structure whose keys are
  not hashable, such as a list in key position (`[[['a']]]`) or a set as a
  leaf; `expand()` then returned paths that no nested dictionary can have.
  Every key of the structure is now checked with `hash()`; an unhashable key
  raises `StackedTypeError`, with the key in the message, its type in
  `actual_type` and the path of its parent in `path`. Closes #141.
- **Concepts pages and compact-path docstrings** stated behaviours the code
  does not have: a compact structure shown as `[['a', 'b', 'c'], ['d']]`
  instead of `[['a', 'b', 'c'], 'd']`, a coverage of `1.25` (coverage is the
  share of the dictionary's paths found in the structure, always between 0
  and 1), an `uncovered_paths()` result of `[]` where it is
  `[['settings', 'lang']]`, and a bijective compact format (only the
  canonical form is unique). They now state that `is_covering()` tests the
  equality of the two sets of paths, and that `['a', 'b', 'c']` and
  `['a', ['b', 'c']]` describe different trees. The examples of the Concepts
  pages are `.. doctest::` blocks, checked by `sphinx-build -b doctest`.
  Closes #145.
- **`size()`** counted the leaf paths of the dictionary instead of its keys,
  although its docstring describes the number of keys at every level, which
  is the number of nodes of the forest of keys. It now counts every key:
  `{'a': {'b': {'c': 1}}, 'd': 2, 'e': {}}` gives 5 instead of 3. The result
  changes for any dictionary with a nested level. Part of #144.
- **`leaves()`** dropped the value of a key whose value is an empty nested
  dictionary, although such a key has no children and is a leaf, as the
  docstring states and as `paths()`, `unpacked_values()` and `height()`
  already treat it. The empty dictionary is now in the result. Part of #144.
- **Docstrings of the traversal and measure methods** described other
  results than the code returns. `bfs()` yields the terminal values only,
  in breadth-first order, not every node. `height()`, `PathsView.get_depth()`
  and `_HKey.get_max_depth()` called on the forest root return the number of
  levels, which is the greatest depth of a key plus 1 (top-level keys have
  depth 0). `_HKey.get_statistics()` counts the forest root in
  `total_nodes`; the examples of `get_statistics()` and `prune()` are
  corrected to the actual output. No runtime change. Closes #144.
- **`_HKey.is_binary_tree()`** also checked the root built by
  `build_forest()`, whose children are the top-level keys: a dictionary with
  more than two top-level keys was never binary, even when no key had more
  than two children. The root is now skipped, as in `is_complete_tree()`,
  `is_perfect_tree()` and `is_full_tree()`, so the number of top-level keys
  is free. Found while writing the Concepts page of #146.
- **`pop()`** used `None` both as its default value and as the mark of a
  missing default. A missing flat key without a default returned `None`
  instead of raising, unlike `dict.pop()` and the docstring, and a missing
  path with an explicit `default=None` raised instead of returning `None`. A
  private sentinel now marks the missing default: without a default, a
  missing key or path raises `StackedKeyError` (a `KeyError`); an explicit
  `None` is returned. The test named after the flat case tested a path; it is
  renamed and the flat case is tested. The sentinel has a readable `repr`, so
  the documented signature shows `default=<no default>` instead of an object
  address. Closes #155.
- **`equal()`** accepted an instance of a subclass in one direction only:
  for a subclass `Inventory` of `NestedDictionary` with the same
  configuration and content, `NestedDictionary(d).equal(Inventory(d))` was
  `True` and the reverse `False`. It tested `isinstance(other, type(self))`;
  it now requires the same type, as its docstring states. Part of #157.
- **Docstrings of the JSON key encoding** (`to_json()`, `from_json()` and the
  `serialize` module) described a `__type__:value` prefix (`"__int__:42"`)
  and a collision with string keys of the same form. Keys are written in
  square brackets (`"[42]"`), and a string key that starts with `[` is
  escaped with a backslash, so that collision does not occur. No runtime
  change. Part of #148.
- **Exception messages** of `StackedKeyError`, `StackedAttributeError`,
  `StackedTypeError`, `StackedValueError` and `StackedIndexError` did not
  show the path given to them, although `StackedDictionaryError` appends
  `" (at path: k1 | k2)"` to its message. Each class called the
  initialiser of its standard exception a second time, which reset the
  message without the path. That call is removed: every exception of the
  family shows the path, for instance the `StackedKeyError` raised by
  `pop()` on a missing path. Closes #158.

---

## [1.2.0] — Persistence (Serialize) — 2026-05-06

### Added

- `.tox-config/` — fragmented requirements by role (`base`, `format`, `linter`,
  `security`, `type-check`, `full`) with dedicated scripts `test.sh` and
  `coverage.sh` for sequential multi-version execution ([DD-025](https://github.com/biface/ndt/issues/116)).
- `pyproject.toml` — `[project.optional-dependencies]` groups `dev` and `docs`
  enabling `uv sync --extra dev --extra docs` for environment bootstrap.
- `PUBLISHING.md` — bilingual (FR/EN) publication procedure covering the full
  RC → TestPyPI → PyPI chain.
- `CONTRIBUTING.md` / `CONTRIBUTING.fr.md` — bilingual contributor guide:
  uv setup, PyCharm configuration, tox environments, CI chain, branch strategy.
- `README.fr.md` — French README split from the original bilingual file.
- `CODE_OF_CONDUCT.md` / `CODE_DE_CONDUITE.md` — aligned to ndict-tools.

### Changed

- **Build tooling:** virtualenv + pip replaced by
  [uv](https://docs.astral.sh/uv/) for virtual environment management,
  dependency installation, and CI. `tox-uv` adopted as tox provisioner
  ([DD-024](https://github.com/biface/ndt/issues/115)). Closes #109.
- **Type checking:** mypy replaced by basedpyright as the sole type checker.
  `pyrightconfig.json` is the single source of configuration ([DD-026](https://github.com/biface/ndt/issues/117)).
- **`tox.ini`** rewritten: new environments `ci-quality`, `ci-tests`,
  `pre-push`, `basedpyright`, `black-check`, `isort-check`, `format`,
  `check`, `coverage`. `local` kept as alias for `pre-push`. `gh-ci`
  and `lint` environments removed.
- **GitHub Actions** — all workflows updated:
  - `actions/checkout@v4` → `actions/checkout@v6` across all workflows.
  - `python-ci-quality.yaml` introduced as the new quality gate (replaces
    the quality section of the former `gh-ci` tox env).
  - `python-ci-tests.yaml` now triggers via `workflow_run` after Quality;
    Python 3.14 runs with `continue-on-error: true`.
  - `python-ci-coverage.yaml` updated to use `tox -e coverage`.
  - Publication chain remains tag-based (`v*.*.*` and `v*.*.*rc*`).
- **Dependencies** updated: pytest 9.0.3 validated (unpinned). Closes #110,
  Closes #121.
- `[tool.isort]` in `pyproject.toml`: `known_first_party` corrected from
  `src` to `ndict_tools`.
- `[tool.tox.*]` sections removed from `pyproject.toml` (now solely in
  `tox.ini`).

### Fixed

- `tools.py:119` — E721: type comparison replaced by identity check
  (`type(d1) != type(d2)` → `type(d1) is not type(d2)`).
- `tools.py:1954–1955` — F841: unused local variables `ind` and `default`
  removed.

### Deprecated

- Flake8 violations in the test suite (`E711`, `E712`, `E731`, `F401`,
  `F541`, `F811`, `F841`) are suppressed via `per-file-ignores` and
  scheduled for cleanup in v1.3.0.

## [1.1.0] — JSON Bridge (Encoder) — 2026-04-05

### Added

- `serialize.py` — new private module providing the serialization
  infrastructure (not part of the public API):
  - `_encode_key` / `_decode_key`: JSON key encoding for non-string keys
    per [DD-021](https://github.com/biface/ndt/issues/87) (supports `int`, `float`, `bool`, flat `tuple`,
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
  Non-string keys are encoded per [DD-021](https://github.com/biface/ndt/issues/87). Closes #43.
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
  explicit removal notice. Migrate to the `classmethod`:

  ```python
  # Before (deprecated since 1.1.0, removed in 1.5.0)
  from ndict_tools.tools import from_dict
  nd = from_dict(data, NestedDictionary, default_setup={...})

  # After
  nd = NestedDictionary.from_dict(data, default_setup={...})
  ```

  Closes #47.

---

## [1.0.0] — Path Manager (Compass) — 2026-01-31

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
  specializations (`StackedKeyError`, `StackedTypeError`,
  `StackedValueError`, `StackedAttributeError`, `StackedIndexError`,
  `NestedDictionaryException`).
- Published on PyPI. CI on GitHub Actions and GitLab CI. Coverage reported
  to codecov.io.

### Deprecated

- `DictPaths` deprecated at instantiation with `DeprecationWarning`.
  Scheduled for removal in 1.2.0. Use `CompactPathsView` or
  `nd.compact_paths()` instead.

---

## [0.9.0] — 2025-11-04

First version kept in the documentation archive; it gathers the work of the
earlier 0.x releases, which are no longer documented.

- **`default_setup` generalised.** The configuration is given through
  `default_setup=`; the specific `indent=` and `strict=` attributes of
  `NestedDictionary.__init__` are removed. Started in 0.8.0, stabilised in
  0.9.0.
- **Python versions earlier than 3.9 are no longer supported.**

---

*For the full list of changes, see the
[GitHub issue tracker](https://github.com/biface/ndt/issues).*
