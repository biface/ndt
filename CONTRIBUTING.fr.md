# Contribuer à ndict-tools

**[English version available](CONTRIBUTING.md)**

Merci de votre intérêt pour contribuer au projet ndict-tools !

---

## Devenir contributeur

Pour devenir contributeur officiel :

1. **Ouvrez une issue** avec le label `Applying`
2. **Indiquez les informations suivantes :**
   - Nom et prénom
   - Pseudo GitHub (@pseudo)
   - Adresse email
   - Ce qui vous motive à contribuer à ce projet

Les mainteneurs examineront votre candidature et vous contacteront pour discuter des
prochaines étapes.

---

## Prérequis

- Python 3.11 (version de référence)
- [uv](https://docs.astral.sh/uv/) installé sur le système
- Git

---

## Mise en place de l'environnement de développement

### 1. Cloner le dépôt

```bash
git clone https://github.com/biface/ndt.git
cd ndt
```

### 2. Créer l'environnement virtuel

```bash
uv venv --python 3.11
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 3. Installer les dépendances de développement

```bash
uv sync --extra dev --extra docs
```

### 4. Installer tox et tox-uv

```bash
uv pip install tox tox-uv
```

### 5. Vérifier l'installation

```bash
tox --version
python -c "import sys; print(sys.version, sys.prefix)"
```

---

## Configuration de PyCharm

Après la création de `.venv/` avec uv, PyCharm doit être pointé vers le nouvel
interpréteur :

`Settings` → `Project: ndt` → `Python Interpreter`
→ `Add Interpreter` → `Add Local Interpreter` → `Existing`
→ sélectionner `.venv/bin/python`

> **Note :** si vous utilisiez précédemment `venv/` (l'ancien environnement pip),
> PyCharm peut encore le référencer. Vérifiez toujours le chemin de l'interpréteur
> après avoir recréé l'environnement.

---

## Stratégie de branches

| Type de branche | Pattern | Objectif | Exemple |
|---|---|---|---|
| Production | `master` | Versions stables publiées sur PyPI | `master` |
| Développement de version | `update/X.Y.Z` | Développement pour une version | `update/1.2.0` |
| Pré-production | `staging/X.Y.Z` | Tests avant publication | `staging/1.2.0` |
| Fonctionnalité | `feature/*` | Nouvelles fonctionnalités | `feature/add-validation` |

```
feature/*  ──PR──▶  update/X.Y.Z  ──PR──▶  staging/X.Y.Z  ──PR──▶  master
```

- Le travail s'effectue sur les branches `update/X.Y.Z`.
- `staging/X.Y.Z` est créée depuis `master` au moment de la release.
- Les commits directs sur `master` ne sont pas autorisés.

---

## Environnements tox

### Développement local

| Commande | Usage |
|---|---|
| `tox -e format` | Formatage automatique (black + isort) |
| `tox -e check` | Vérification rapide (sans correction) |
| `tox -e basedpyright` | Vérification de types uniquement |
| `tox -e flake8` | Analyse statique uniquement |
| `tox -e bandit` | Analyse de sécurité uniquement |
| `tox -e py311` | Tests sur Python 3.11 |
| `tox -e coverage` | Génération du rapport de couverture |
| `tox -e pre-push` | Workflow complet avant push |
| `tox -e local` | Alias de `pre-push` |

### Environnements CI (GitHub Actions uniquement — ne pas exécuter en local)

| Environnement | Usage |
|---|---|
| `ci-quality` | Contrôle qualité (format + lint + types + sécurité) |
| `ci-tests` | Exécution de la matrice de tests (Python 3.11–3.14 ; 3.15 et 3.14t sans garantie) |

> **Important :** `ci-quality` et `ci-tests` sont conçus pour GitHub Actions.
> Utilisez `tox -e pre-push` ou `tox -e check` pour les vérifications locales.

---

## Vue d'ensemble de la chaîne CI

| Événement | Workflow déclenché | Résultat |
|---|---|---|
| Push sur n'importe quelle branche | Python CI - Quality | Contrôles qualité |
| Quality réussie | Python CI - Tests | Tests multi-versions (3.11–3.14 ; 3.15 et 3.14t sans garantie) |
| Tests réussis (staging/**, master) | Python CI - Coverage | Upload Codecov |
| Push tag `vX.Y.Zrc1` | Python CI - Build → Publish TestPyPI | RC sur TestPyPI |
| Push tag `vX.Y.Z` | Python CI - Build → Publish PyPI | Release finale sur PyPI |

> La chaîne `workflow_run` complète (Quality → Tests → Coverage) ne fonctionne
> qu'une fois les fichiers de workflow présents sur `master`.

---

## Workflow avant l'ouverture d'une PR

Toujours exécuter le workflow local complet avant de pousser :

```bash
tox -e pre-push
```

Cela exécute dans l'ordre :

1. Formatage automatique (black + isort)
2. Vérification de types (basedpyright)
3. Analyse statique (flake8)
4. Analyse de sécurité (bandit)
5. Tests multi-versions séquentiels (`.tox-config/scripts/test.sh`)
6. Rapport de couverture (`.tox-config/scripts/coverage.sh`)

---

## Documentation

La documentation est construite avec Sphinx à partir de `docs/source/` ; ses
dépendances sont dans l'extra `docs` (`uv sync --extra docs`). Le build doit
passer sans avertissement :

```bash
sphinx-build -W -b html docs/source docs/build/html
```

### Traductions

La documentation est rédigée en anglais. Les traductions passent par les
catalogues gettext de Sphinx, un par page source, dans
`docs/source/locales/<langue>/LC_MESSAGES/`. Les chaînes non traduites
s'affichent en anglais.

1. Extraire les messages. Les fichiers `.pot` sont des produits de build et ne
   sont pas commités :

   ```bash
   sphinx-build -b gettext docs/source docs/build/gettext
   ```

2. Créer ou mettre à jour les catalogues d'une langue (ici le français) :

   ```bash
   sphinx-intl update -p docs/build/gettext -l fr -d docs/source/locales
   ```

3. Renseigner les entrées `msgstr` des fichiers `.po` et les commiter. Les
   fichiers `.mo` sont compilés par Sphinx au moment du build et ne sont pas
   commités.

4. Vérifier ce qu'il reste à traduire. L'option `-d` est obligatoire depuis la
   racine du dépôt : sans elle, sphinx-intl cherche `conf.py` dans le
   répertoire courant et échoue avec une `TypeError`.

   ```bash
   sphinx-intl stat -d docs/source/locales -l fr
   ```

5. Construire la documentation traduite :

   ```bash
   sphinx-build -W -b html -D language=fr docs/source docs/build/html-fr
   ```

Une traduction est publiée dans l'archive GitHub Pages, sous
`/<langue>/vX.Y.Z/`, dès que son code de langue figure dans
`docs/source/locales/LANGUAGES`. N'y ajouter le code qu'une fois le catalogue
traduit.

### Glossaire de traduction

La documentation et le code suivent l'usage anglais pour les formes d'arbre.
Les termes français ne se traduisent pas mot à mot : un *arbre complet* en
français est un *perfect tree* en anglais. Les traductions suivent ce tableau ;
les définitions sont sur la page Concepts « The Forest of Keys ».

| Anglais | Français | Sens |
|---|---|---|
| complete tree | arbre quasi complet (*tassé à gauche*) | tous les niveaux remplis sauf peut-être le dernier, rempli de gauche à droite |
| perfect tree | arbre complet | tous les niveaux remplis |
| full tree | arbre localement complet (*strict*) | chaque nœud interne a exactement n enfants |

---

## Conventions de commit

- Langue : **anglais**
- Style : verbe impératif, minuscules (`fix`, `add`, `remove`, `update`)
- Format : `<type>: <description courte>`
- Fermer les issues avec `Closes #N` dans le corps du commit
- Regrouper les modifications liées dans un seul commit atomique

**Types :** `feat`, `fix`, `chore`, `docs`, `test`, `ci`, `refactor`

---

## Décisions de conception

Tout choix architectural non trivial doit être documenté dans `DESIGN_DECISIONS.md`
**avant** le début de l'implémentation. Utiliser le format d'identifiant DD-NNN.

---

## Objectif de couverture

80–90 % de couverture de lignes (imposé par `.codecov.yml`).
