# Lecteur français

En Python standard, il est possible d'avoir des dictionnaires à l'intérieur d'autres dictionnaires, créant ainsi des
structures de données imbriquées. Cependant, bien que cette fonctionnalité existe, Python ne propose pas de moyens
natifs pour rechercher facilement ou gérer les clés et valeurs dans des dictionnaires imbriqués complexes.

Mes recherches et tests sur des bibliothèques dédiées à la gestion des dictionnaires imbriqués m'ont conduit à
plusieurs solutions, mais aucune n'a pleinement répondu à mes attentes. Le module qui s'en rapproche le plus est celui 
datant de 2015, [disponible sur PyPI](https://pypi.org/project/nested_dict/), mais il n'offre pas une architecture 
complète pour gérer les "objets de dictionnaires imbriqués" de manière fluide et robuste.

Cela m'a donc poussé à redévelopper un tel module, offrant une gestion plus complète et intuitive des dictionnaires 
imbriqués. Ce module facilite la manipulation, la recherche, et la gestion des clés et valeurs dans des structures 
de données plus complexes, en offrant des outils dédiés à cette tâche spécifique.

## Qu'est-ce qu'un dictionnaire imbriqué ?

Un dictionnaire imbriqué est simplement un dictionnaire dont les valeurs peuvent elles-mêmes être des dictionnaires.
Cela permet de créer des structures de données plus riches et hiérarchiques, où chaque "nœud" de la structure peut 
contenir des informations supplémentaires sous forme de dictionnaires, permettant ainsi de modéliser des données 
complexes de manière organisée et accessible.

## Utilisation des clés imbriquées et gestion des hiérarchies dans les dictionnaires

### Clés de différents types et utilisation des listes pour gérer la hiérarchie

Comme pour les dictionnaires classiques en Python, les clés dans un dictionnaire imbriqué doivent être **hashables**.
Cela signifie que vous pouvez utiliser des types comme **nombres**, **chaînes de caractères**, ou **tuples** comme clés.
Cependant, les **listes** ne sont pas hashables et ne peuvent pas être utilisées directement comme clés.

### Accès aux valeurs imbriquées

Les dictionnaires imbriqués vous permettent de structurer vos données en plusieurs niveaux. Par exemple, pour accéder à
une valeur dans un dictionnaire imbriqué, vous pouvez utiliser une séquence de clés qui représentera chaque niveau de
la hiérarchie.

Par le biais des listes simples et non imbriquées, nous représentons cette hiérarchie d'imbrication. 

#### Exemple d'accès imbriqué

Les deux expressions suivantes sont **équivalentes** pour accéder à une valeur dans un dictionnaire imbriqué :

```dictionnaire[[1, "a", (2, 3)]]``` est equivalent à ```dictionnaire[1]["a"][(2, 3)]```

# English reader and ROW

In standard Python, dictionaries within dictionaries are possible, creating nested data structures. However, while this
functionality exists, Python does not offer native features to easily search and manage keys and values within complex
nested dictionaries.

My research and testing of libraries dedicated to managing nested dictionaries led me to several solutions, but none
fully met my expectations. The module I found that came closest was one from 2015, [available on
PyPI](https://pypi.org/project/nested_dict/), but it does not provide a complete architecture for managing
"nested dictionary objects" in a smooth and robust way. This motivated me to redevelop such a module, offering a more
complete and intuitive way to handle nested dictionaries. This new module makes it easier to manipulate, search, and 
manage keys and values in complex data structures by providing tools dedicated to this specific task.

## What is a Nested Dictionary?

A nested dictionary is simply a dictionary where the values themselves are dictionaries. This allows for the creation
of richer, hierarchical data structures where each "node" in the structure can hold additional information in the form
of dictionaries, making it easier to model complex data in an organized and accessible way.

## Using Nested Keys and Managing Hierarchies in Dictionaries

### Keys of Different Types and Using Lists for Hierarchical Keys

Just like with standard dictionaries in Python, the keys in a nested dictionary must be **hashable**. This means you 
can use types such as **numbers**, **strings**, or even **tuples** as keys. However, **lists** are not hashable and
cannot be used directly as keys.

### Accessing Nested Values

Nested dictionaries allow you to structure your data over multiple levels. For example, to access a value in a nested
dictionary, you can use a sequence of keys that represents each level of the hierarchy.

Using simple non nested lists is the way used to represents nested keys.

#### Nested Access Example

The following two expressions are **equivalent** for accessing a value in a nested dictionary:

```dict[[1, "a", (2, 3)]]``` is equivalent to ```dict[1]["a"][(2, 3)]```

