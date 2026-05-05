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

- Python 3.10 (baseline version)
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
uv venv --python 3.10
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install development dependencies

```bash
uv sync --extra dev --extra docs
```

### 4. Install tox and tox-uv

```bash
uv pip install tox tox-uv
```

### 5. Verify the setup

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

### Local development

| Command | Purpose |
|---|---|
| `tox -e format` | Auto-format code (black + isort) |
| `tox -e check` | Quick verification (no auto-fix) |
| `tox -e basedpyright` | Type checking only |
| `tox -e flake8` | Linting only |
| `tox -e bandit` | Security analysis only |
| `tox -e py310` | Run tests on Python 3.10 |
| `tox -e coverage` | Generate coverage report |
| `tox -e pre-push` | Full workflow before push |
| `tox -e local` | Alias for `pre-push` |

### CI environments (GitHub Actions only — do not run locally)

| Environment | Purpose |
|---|---|
| `ci-quality` | Quality gate (format + lint + type + security) |
| `ci-tests` | Test matrix runner (Python 3.10–3.14) |

> **Important:** `ci-quality` and `ci-tests` are designed for GitHub Actions.
> Use `tox -e pre-push` or `tox -e check` for local verification.

---

## CI chain overview

| Event | Workflow triggered | Outcome |
|---|---|---|
| Push to any branch | Python CI - Quality | Quality checks |
| Quality succeeded | Python CI - Tests | Multi-version tests (3.10–3.14) |
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
