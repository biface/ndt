# Contributing to ndict-tools

**[Version française disponible](CONTRIBUTING.fr.md)**

Thank you for your interest in contributing to the ndict-tools project!

---

## Becoming a Contributor

To become an official contributor:

1. **Open an issue** with the label `Applying`
2. **Include the following information:**
   - First name and last name
   - GitHub username (@username)
   - Email address
   - What motivates you to contribute to this project

Maintainers will review your application and contact you to discuss next steps.

---

## Prerequisites

- Python 3.11 (baseline version)
- [uv](https://docs.astral.sh/uv/) installed system-wide
- Git

---

## Setting up the development environment

### 1. Clone the repository

```bash
git clone https://github.com/biface/ndt.git
cd ndt
```

### 2. Create the virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install the package and tox

```bash
uv pip install -e . tox tox-uv
```

The development tools (pytest, basedpyright, black, Sphinx…) are not
installed in `.venv/`: tox installs them in its own environments, from the
files of `.tox-config/requirements/`. To use one of them in your IDE,
install it in `.venv/` by hand.

### 4. Verify the setup

```bash
tox --version
python -c "import sys; print(sys.version, sys.prefix)"
```

---

## PyCharm setup

After creating `.venv/` with uv, PyCharm must be pointed to the new interpreter:

`Settings` → `Project: ndt` → `Python Interpreter`
→ `Add Interpreter` → `Add Local Interpreter` → `Existing`
→ select `.venv/bin/python`

> **Note:** if you previously used `venv/` (the old pip-based environment),
> PyCharm may still reference it. Always verify the interpreter path after
> recreating the environment.

---

## Branch strategy

| Branch type | Pattern | Purpose | Example |
|---|---|---|---|
| Production | `master` | Stable versions published to PyPI | `master` |
| Version development | `update/X.Y.Z` | Development for a specific version | `update/1.2.0` |
| Pre-production | `staging/X.Y.Z` | Testing before publication | `staging/1.2.0` |
| Feature | `feature/*` | New features | `feature/add-validation` |

```
feature/*  ──PR──▶  update/X.Y.Z  ──PR──▶  staging/X.Y.Z  ──PR──▶  master
```

- Work is done on `update/X.Y.Z` branches.
- `staging/X.Y.Z` is created from `master` at release time.
- Direct commits to `master` are not allowed.

---

## Tox environments

The environments are defined in `pyproject.toml` (`[tool.tox]`), like the
settings of the other tools. Their dependencies come from
`.tox-config/requirements/<role>.txt`.

### Local development

| Command | Purpose |
|---|---|
| `tox -e format` | Auto-format code (black + isort) |
| `tox -e check` | Quick verification (no auto-fix) |
| `tox -e basedpyright` | Type checking only |
| `tox -e flake8` | Linting only |
| `tox -e bandit` | Security analysis only |
| `tox -e py311` | Run tests on Python 3.11 |
| `tox -e coverage` | Generate coverage report |
| `tox -e docs` | Build the documentation (English, French) and run its doctests |
| `tox -e pre-push` | Full workflow before push |
| `tox -e local` | Alias for `pre-push` |

### CI environments (GitHub Actions only — do not run locally)

| Environment | Purpose |
|---|---|
| `ci-quality` | Quality gate (format + lint + type + security) |

The test workflow runs `py311` to `py314` (`py315` and `py314t` best
effort), then `coverage`.

> **Important:** `ci-quality` is designed for GitHub Actions.
> Use `tox -e pre-push` or `tox -e check` for local verification.

---

## CI chain overview

| Event | Workflow triggered | Outcome |
|---|---|---|
| Push to any branch | Python CI - Quality | Quality checks |
| Quality succeeded | Python CI - Tests | Multi-version tests (3.11–3.14; 3.15 and 3.14t best effort) |
| Tests succeeded (staging/**, master) | Python CI - Coverage | Codecov upload |
| Push tag `vX.Y.Zrc1` | Python CI - Build → Publish TestPyPI | RC on TestPyPI |
| Push tag `vX.Y.Z` | Python CI - Build → Publish PyPI | Final release on PyPI |

> The full `workflow_run` chain (Quality → Tests → Coverage) only works once
> the workflow files are present on `master`.

---

## Workflow before opening a PR

Always run the full local workflow before pushing:

```bash
tox -e pre-push
```

This runs in sequence:

1. Auto-formatting (black + isort)
2. Type checking (basedpyright)
3. Linting (flake8)
4. Security analysis (bandit)
5. Sequential multi-version tests (`.tox-config/scripts/test.sh`)
6. Coverage report (`.tox-config/scripts/coverage.sh`)

---

## Documentation

The documentation is built with Sphinx from `docs/source/`. Its dependencies
are pinned in `docs/source/requirements.txt`, which Read the Docs reads, and
installed by tox in the `docs` environment. The builds must pass without
warnings, in English and in French, and the doctests must pass in both
languages:

```bash
tox -e docs
```

The commands below run in that environment through `tox exec -e docs --`.

### Translations

The documentation is written in English. Translations use Sphinx gettext
catalogs, one per source page, in `docs/source/locales/<lang>/LC_MESSAGES/`.
Untranslated strings fall back to English.

1. Extract the messages. The `.pot` files are build output and are not
   committed:

   ```bash
   tox exec -e docs -- sphinx-build -b gettext docs/source docs/build/gettext
   ```

2. Create or update the catalogs of a language (here French):

   ```bash
   tox exec -e docs -- sphinx-intl update -p docs/build/gettext -l fr -d docs/source/locales
   ```

3. Fill in the `msgstr` entries of the `.po` files and commit them. The `.mo`
   files are compiled by Sphinx at build time and are not committed.

   Code and doctest blocks are extracted too. Translate only their comments
   and the annotations of text diagrams; copy the code and its output
   unchanged, since the translated build does not run the doctests.
   `sphinx.po` holds the strings of the theme and of Sphinx itself: it has
   no template and is kept by hand.

4. Check what remains to translate. `-d` is required when the command runs
   from the repository root: without it, sphinx-intl looks for `conf.py` in
   the current directory and fails with a `TypeError`.

   ```bash
   tox exec -e docs -- sphinx-intl stat -d docs/source/locales -l fr
   ```

5. Build the translated documentation:

   ```bash
   tox exec -e docs -- sphinx-build -W -b html -D language=fr docs/source docs/build/html-fr
   ```

The full change log (`changelog/history.po`) stays in English: only the
title and introduction of the page are translated, so `sphinx-intl stat`
always reports untranslated strings for this catalog, and exits with status 1.

A translation is published in the GitHub Pages archive, under
`/<lang>/vX.Y.Z/`, once its language code is listed in
`docs/source/locales/LANGUAGES`. Add the code when the catalog is translated,
not before.

### Translation glossary

The documentation and the code follow English usage for tree shapes. The
French terms do not map word for word: a French *arbre complet* is an English
*perfect tree*. Translations use this table; the definitions are on the
Concepts page "The Forest of Keys".

| English | French | Meaning |
|---|---|---|
| complete tree | arbre quasi complet (*tassé à gauche*) | every level filled except possibly the last, filled left to right |
| perfect tree | arbre complet | every level filled |
| full tree | arbre localement complet (*strict*) | every internal node has exactly n children |

In French, a class name that stands for an object is masculine (*un
`PathsView`*). Exceptions take the gender of *une exception* (*lève une
`KeyError`*) and warnings that of *un avertissement* (*émet un
`UserWarning`*).

---

## Commit conventions

- Language: **English**
- Style: imperative verb, lowercase (`fix`, `add`, `remove`, `update`)
- Format: `<type>: <short description>`
- Close issues with `Closes #N` in the commit body
- Group related changes into a single atomic commit

**Types:** `feat`, `fix`, `chore`, `docs`, `test`, `ci`, `refactor`

---

## Design decisions

Any non-trivial architectural choice must be documented in `DESIGN_DECISIONS.md`
**before** implementation begins. Use the DD-NNN identifier format.

---

## Coverage target

80–90% line coverage (enforced by `.codecov.yml`).
