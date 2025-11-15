# Contributing to NDT

**[Version française disponible](CONTRIBUTING.fr.md)**

Thank you for your interest in contributing to the NDT project!

## Becoming a Contributor

To become an official contributor:

1. **Open an issue** with the label `Applying`
2. **Include the following information:**
   - First name and last name
   - GitHub username (@username)
   - Email address
   - What motivates you to contribute to this project

Maintainers will review your application and contact you to discuss next steps.

## Development Process

This project follows a **controlled software delivery methodology** with automated workflows. The complete methodology is documented in detail here:

**📖 [Controlled Delivery Software - Full Documentation](https://gitlab.com/biface/biface/-/wikis/controlled-delivery-software)**

### Branch Structure Overview

| Branch Type | Pattern | Purpose | Example |
|-------------|---------|---------|---------|
| **Production** | `main` | Stable versions published to PyPI | `main` |
| **Version Development** | `updates/X.Y.0` | Development for specific version | `updates/1.0.0` |
| **Pre-production** | `staging/X.Y.x` | Testing before publication | `staging/1.0.x` |
| **Feature** | `feature/*` | New features | `feature/add-validation` |
| **Hotfix** | `hotfix/*` | Urgent fixes | `hotfix/security-fix` |

### Versioning Strategy

We use an **even/odd minor version system**:

- **Odd versions** (1.1.x, 1.3.x): Experimental, published to TestPyPI only
- **Even versions** (1.0.x, 1.2.x): Stable, published to official PyPI

**Example flow:**
```
Feature development → updates/1.1.0 → staging/1.1.x → TestPyPI (experimental)
                                                      → Validation
Stabilization → updates/1.2.0 → staging/1.2.x → TestPyPI → main → PyPI (stable)
```
## Automated Workflows

This project uses 6 automated GitHub Actions workflows. Full technical documentation is available here:

**📖 [Automation Pipelines Documentation](https://github.com/biface/biface/blob/main/automation/pipelines.md)**

### Workflow Overview

| Workflow | Trigger | Branches | Action |
|----------|---------|----------|--------|
| **1. Tests** | Push, PR | All branches | Run test suite on Python 3.9-3.12 |
| **2. Coverage** | After Tests | `updates/*`, `staging/*`, `main` | Calculate code coverage |
| **3. Build** | After Coverage | `staging/*`, `main` | Build package (.whl, .tar.gz) |
| **4. TestPyPI** | After Build | `staging/*`, `main` | Publish to test.pypi.org |
| **5. PyPI** | After TestPyPI | `main` only | Publish to pypi.org (production) |
| **6. Release** | After PyPI | `main` only | Create Git tag and GitHub Release |

**Workflow execution by branch:**

| Branch Type | Tests | Coverage | Build | TestPyPI | PyPI | Release |
|-------------|-------|----------|-------|----------|------|---------|
| `feature/*` | ✅ | - | - | - | - | - |
| `updates/*` | ✅ | ✅ | - | - | - | - |
| `staging/*` | ✅ | ✅ | ✅ | ✅ | - | - |
| `main` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |