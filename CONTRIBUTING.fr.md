# Contribuer à NDT

**[English version available](CONTRIBUTING.md)**

Merci de votre intérêt pour contribuer au projet NDT !

## Devenir Contributeur

Pour devenir contributeur officiel :

1. **Ouvrez une issue** avec le label `Applying`
2. **Indiquez les informations suivantes :**
   - Nom et prénom
   - Pseudo GitHub (@pseudo)
   - Adresse email
   - Ce qui vous motive à contribuer à ce projet

Les mainteneurs examineront votre candidature et vous contacteront pour discuter des prochaines étapes.

## Processus de Développement

Ce projet suit une **méthodologie de livraison contrôlée de logiciels** avec des workflows automatisés. La méthodologie complète est documentée en détail ici :

**📖 [Controlled Delivery Software - Documentation Complète](https://gitlab.com/biface/biface/-/wikis/controlled-delivery-software)** *(en anglais)*

### Vue d'Ensemble de la Structure des Branches

| Type de Branche | Pattern | Objectif | Exemple |
|-----------------|---------|----------|---------|
| **Production** | `main` | Versions stables publiées sur PyPI | `main` |
| **Développement Version** | `updates/X.Y.0` | Développement par version | `updates/1.0.0` |
| **Pré-production** | `staging/X.Y.x` | Tests avant publication | `staging/1.0.x` |
| **Feature** | `feature/*` | Nouvelles fonctionnalités | `feature/add-validation` |
| **Hotfix** | `hotfix/*` | Corrections urgentes | `hotfix/security-fix` |

### Stratégie de Versionnage

Nous utilisons un **système de versions mineures pair/impair** :

- **Versions impaires** (1.1.x, 1.3.x) : Expérimentales, publiées sur TestPyPI uniquement
- **Versions paires** (1.0.x, 1.2.x) : Stables, publiées sur PyPI officiel

**Exemple de flux :**
```
Développement feature → updates/1.1.0 → staging/1.1.x → TestPyPI (expérimental)
                                                       → Validation
Stabilisation → updates/1.2.0 → staging/1.2.x → TestPyPI → main → PyPI (stable)
```

## Workflows Automatisés

Ce projet utilise 6 workflows GitHub Actions automatisés. La documentation technique complète est disponible ici :

**📖 [Documentation des Pipelines d'Automation](https://github.com/biface/biface/blob/main/automation/pipelines.md)**

### Vue d'Ensemble des Workflows

| Workflow | Déclencheur | Branches | Action |
|----------|-------------|----------|--------|
| **1. Tests** | Push, PR | Toutes les branches | Exécute les tests sur Python 3.9-3.12 |
| **2. Coverage** | Après Tests | `updates/*`, `staging/*`, `main` | Calcule la couverture de code |
| **3. Build** | Après Coverage | `staging/*`, `main` | Compile le package (.whl, .tar.gz) |
| **4. TestPyPI** | Après Build | `staging/*`, `main` | Publie sur test.pypi.org |
| **5. PyPI** | Après TestPyPI | `main` uniquement | Publie sur pypi.org (production) |
| **6. Release** | Après PyPI | `main` uniquement | Crée le tag Git et la GitHub Release |

**Exécution des workflows par branche :**

| Type de Branche | Tests | Coverage | Build | TestPyPI | PyPI | Release |
|-----------------|-------|----------|-------|----------|------|---------|
| `feature/*` | ✅ | - | - | - | - | - |
| `updates/*` | ✅ | ✅ | - | - | - | - |
| `staging/*` | ✅ | ✅ | ✅ | ✅ | - | - |
| `main` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |