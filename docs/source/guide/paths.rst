Part 2 — Working with Paths
============================

This part shows how to enumerate, navigate, filter, and analyse the paths
of a :class:`~ndict_tools.NestedDictionary`. For the underlying concepts,
see :doc:`/concepts/paths` and :doc:`/concepts/compact_paths`.


Listing all paths
------------------

Call :meth:`~ndict_tools.NestedDictionary.paths` to obtain a
:class:`~ndict_tools.PathsView` — a lazy view that enumerates every node
in the tree, from root to leaves:

.. code-block:: python

   from ndict_tools import NestedDictionary

   nd = NestedDictionary({
       "config": {"host": "localhost", "port": 5432},
       "debug": True,
   })

   pv = nd.paths()

   for path in pv:
       print(path)
   # ['config']
   # ['config', 'host']
   # ['config', 'port']
   # ['debug']

   len(pv)            # 4
   ['debug'] in pv    # True
   ['config', 'db'] in pv  # False

The view is computed once on first access and cached — subsequent
iterations and membership tests reuse the same internal tree.


Navigating parent–child relationships
---------------------------------------

.. code-block:: python

   pv.get_children(['config'])        # ['host', 'port']
   pv.get_children(['config', 'host']) # []  — leaf node
   pv.has_children(['config'])        # True
   pv.has_children(['debug'])         # False

   # All paths rooted at a prefix
   pv.get_subtree_paths(['config'])
   # [['config'], ['config', 'host'], ['config', 'port']]

   # Leaf paths only (nodes with no children)
   pv.get_leaf_paths()
   # [['config', 'host'], ['config', 'port'], ['debug']]

   # Maximum nesting depth
   pv.get_depth()     # 2


Filtering paths
----------------

Pass a predicate to :meth:`~ndict_tools.PathsView.filter_paths` to
select a subset:

.. code-block:: python

   # Only paths deeper than one level
   pv.filter_paths(lambda p: len(p) > 1)
   # [['config', 'host'], ['config', 'port']]

   # Paths that contain the key 'host'
   pv.filter_paths(lambda p: 'host' in p)
   # [['config', 'host']]

   # Leaf paths under 'config'
   pv.filter_paths(lambda p: p[0] == 'config' and not pv.has_children(p))
   # [['config', 'host'], ['config', 'port']]


Searching by key name
----------------------

When you don't know the exact path but know the key name, use the search
methods directly on the dictionary:

.. code-block:: python

   nd = NestedDictionary({
       "prod":    {"db": {"host": "prod.db", "port": 5432}},
       "staging": {"db": {"host": "stg.db",  "port": 5432}},
   })

   nd.is_key("host")       # True  — exists somewhere
   nd.occurrences("host")  # 2     — found at two paths
   nd.key_list("host")
   # [('prod', 'db', 'host'), ('staging', 'db', 'host')]
   nd.items_list("host")
   # ['prod.db', 'stg.db']


Working with compact paths
---------------------------

A :class:`~ndict_tools.CompactPathsView` groups siblings under their
shared parent. It is useful for inspecting structure at a glance and for
coverage analysis.

.. code-block:: python

   cpv = nd.compact_paths()
   cpv.structure
   # [['prod', ['db', 'host', 'port']], ['staging', ['db', 'host', 'port']]]

Convert between the two views freely:

.. code-block:: python

   # PathsView → CompactPathsView
   cpv = nd.paths().to_compact()

   # CompactPathsView → PathsView
   pv2 = cpv.to_paths()


Checking coverage
------------------

Use a :class:`~ndict_tools.CompactPathsView` to measure how much of a
nested dictionary a given path set describes.

A typical use case is validating that an incoming payload covers all keys
required by a template:

.. code-block:: python

   from ndict_tools import NestedDictionary

   # Required structure — values are irrelevant, keys matter
   template = NestedDictionary({
       "user": {"name": None, "email": None},
       "settings": {"theme": None},
   })
   spec = template.compact_paths()

   # Incoming payload
   payload = NestedDictionary({
       "user": {"name": "Alice", "email": "alice@example.com"},
       "settings": {"theme": "dark"},
   })

   spec.is_covering(payload)      # True  — all required paths present
   spec.coverage(payload)         # 1.0
   spec.uncovered_paths(payload)  # []
   spec.missing_paths(payload)    # []

If the payload is incomplete:

.. code-block:: python

   incomplete = NestedDictionary({
       "user": {"name": "Bob"},   # missing 'email'
       "settings": {"theme": "light"},
   })

   spec.is_covering(incomplete)      # False
   spec.coverage(incomplete)         # 0.8  — 4 of 5 paths covered
   spec.uncovered_paths(incomplete)  # [['user', 'email']]
