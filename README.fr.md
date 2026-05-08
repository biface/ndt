![Python](https://img.shields.io/badge/Language-python-green.svg)
![PyPI - Status](https://img.shields.io/pypi/status/ndict-tools)
![PyPI - License](https://img.shields.io/pypi/l/ndict-tools)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ndict-tools)
![Read the Docs](https://img.shields.io/readthedocs/ndict-tools)
![Test](https://github.com/biface/ndt/actions/workflows/python-ci-tests.yaml/badge.svg?branch=master)
![Codecov](https://img.shields.io/codecov/c/github/biface/ndt)
![GitHub Release](https://img.shields.io/github/v/release/biface/ndt)
![PyPI - Version](https://img.shields.io/pypi/v/ndict-tools)

**[English version available](README.md)**

---

# ndict-tools

En Python standard, il est possible d'avoir des dictionnaires à l'intérieur d'autres
dictionnaires, créant ainsi des structures de données imbriquées. Cependant, bien que
cette fonctionnalité existe, Python ne propose pas de moyens natifs pour rechercher
facilement ou gérer les clés et valeurs dans des dictionnaires imbriqués complexes.

Mes recherches et tests sur des bibliothèques dédiées à la gestion des dictionnaires
imbriqués m'ont conduit à plusieurs solutions, mais aucune n'a pleinement répondu à mes
attentes. Le module qui s'en rapproche le plus est celui datant de 2015,
[disponible sur PyPI](https://pypi.org/project/nested_dict/), mais il n'offre pas une
architecture complète pour gérer les « objets de dictionnaires imbriqués » de manière
fluide et robuste.

Cela m'a donc poussé à redévelopper un tel module, offrant une gestion plus complète et
intuitive des dictionnaires imbriqués. Ce module facilite la manipulation, la recherche,
et la gestion des clés et valeurs dans des structures de données plus complexes, en
offrant des outils dédiés à cette tâche spécifique.

## Qu'est-ce qu'un dictionnaire imbriqué ?

Un dictionnaire imbriqué est simplement un dictionnaire dont les valeurs peuvent
elles-mêmes être des dictionnaires. Cela permet de créer des structures de données plus
riches et hiérarchiques, où chaque « nœud » de la structure peut contenir des
informations supplémentaires sous forme de dictionnaires, permettant ainsi de modéliser
des données complexes de manière organisée et accessible.

## Utilisation des clés imbriquées et gestion des hiérarchies

### Clés de différents types et utilisation des listes pour gérer la hiérarchie

Comme pour les dictionnaires classiques en Python, les clés dans un dictionnaire
imbriqué doivent être **hashables**. Cela signifie que vous pouvez utiliser des types
comme **nombres**, **chaînes de caractères** ou **tuples** comme clés. Cependant, les
**listes** ne sont pas hashables et ne peuvent pas être utilisées directement comme
clés.

### Accès aux valeurs imbriquées

Les dictionnaires imbriqués vous permettent de structurer vos données en plusieurs
niveaux. Pour accéder à une valeur dans un dictionnaire imbriqué, vous pouvez utiliser
une séquence de clés qui représente chaque niveau de la hiérarchie.

Par le biais des listes simples et non imbriquées, nous représentons cette hiérarchie
d'imbrication.

#### Exemple d'accès imbriqué

Les deux expressions suivantes sont **équivalentes** pour accéder à une valeur dans un
dictionnaire imbriqué :

```python,ignore
dictionnaire[[1, "a", (2, 3)]]   # est équivalent à
dictionnaire[1]["a"][(2, 3)]
```

## Installation

```bash
pip install ndict-tools
```

## Documentation

La documentation complète est disponible sur
[ndict-tools.readthedocs.io](https://ndict-tools.readthedocs.io/en/latest/).

## Contribuer

Voir [CONTRIBUTING.fr.md](CONTRIBUTING.fr.md) pour les instructions de mise en place et
les directives de contribution.

## Licence

Ce projet est distribué sous la
[licence CeCILL-C](https://cecill.info/licences/Licence_CeCILL-C_V1-fr.html).
