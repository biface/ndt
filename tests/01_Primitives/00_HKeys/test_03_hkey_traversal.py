"""
Tests for tree traversal algorithms in _HKey class.

These tests use a key tree (key_tree) built from a complex stacked dictionary
(_StackedDict) representing a multi-environment configuration.
"""

from collections import defaultdict, deque
from typing import Any

import pytest

# ============================================================================
# Tests for tree traversal algorithms (DFS and BFS)
# ============================================================================


class TestTreeTraversal:
    """
    Tests for the three main tree traversal algorithms.

    Covers:
    - DFS pre-order (node before children)
    - DFS post-order (children before node)
    - BFS (level-by-level traversal)
    """

    # Shared attributes for collecting data during traversals
    keys_list = []
    depth_map = {}
    path_map = {}

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset attributes before each test to avoid side effects."""
        self.keys_list = []
        self.depth_map = {}
        self.path_map = {}

    # ========================================================================
    # HELPER METHODS - Collection
    # ========================================================================

    def make_collector(self, collection_type="key"):
        """
        Factory method to create different types of collectors.

        Parameters
        ----------
        collection_type : str
            Type of data to collect:
            - 'key': collect node keys
            - 'node': collect node objects
            - 'depth': collect nodes grouped by depth
            - 'path': collect node paths

        Returns
        -------
        callable
            Collector function compatible with traversal visit parameter
        """
        if collection_type == "key":

            def collector(node):
                if not node.is_root:
                    self.keys_list.append(node.key)

            return collector

        elif collection_type == "node":
            nodes_list = []

            def collector(node):
                if not node.is_root:
                    nodes_list.append(node)

            collector.get_collected = lambda: nodes_list
            return collector

        elif collection_type == "depth":

            def collector(node):
                if not node.is_root:
                    depth = node.get_depth()
                    if depth not in self.depth_map:
                        self.depth_map[depth] = []
                    self.depth_map[depth].append(node.key)

            return collector

        elif collection_type == "path":

            def collector(node):
                if not node.is_root:
                    self.path_map[node.key] = tuple(node.get_path())

            return collector

        else:
            raise ValueError(f"Unknown collection_type: {collection_type}")

    def collect_nodes_with_traversal(self, tree, method="bfs", filter_func=None):
        """
        Generic node collection with any traversal method and optional filtering.

        Parameters
        ----------
        tree : _HKey
            Tree to traverse
        method : str
            Traversal method: 'bfs', 'dfs_preorder', 'dfs_postorder'
        filter_func : callable, optional
            Filter function applied to each node. If None, collect all nodes.

        Returns
        -------
        list
            List of collected nodes
        """
        collected = []

        def collector(node):
            if not node.is_root:
                if filter_func is None or filter_func(node):
                    collected.append(node)

        traversal_method = getattr(tree, method)
        list(traversal_method(collector))

        return collected

    # ========================================================================
    # HELPER METHODS - BFS Analysis
    # ========================================================================

    def build_position_depth_tuples(self, nodes, verbose=False):
        """
        Build (position, depth) tuples from collected nodes.

        Parameters
        ----------
        nodes : list
            List of _HKey nodes
        verbose : bool, optional
            If True, print the tuples

        Returns
        -------
        list
            List of (position, depth) tuples
        """
        tuples = [(i, node.get_depth()) for i, node in enumerate(nodes)]

        if verbose:
            print("\nPOSITION-DEPTH TUPLES (first 15):")
            print("-" * 60)
            for pos, depth in tuples[:15]:
                key_str = str(nodes[pos].key)[:35]
                print(f"  Position {pos:3d}: depth={depth}  key={key_str}")
            if len(tuples) > 15:
                print(f"  ... ({len(tuples) - 15} more tuples)")

        return tuples

    def build_level_ranges(self, position_depth_tuples, verbose=False):
        """
        Build [depth, first, last] ranges from position-depth tuples.

        Algorithm:
        - Iterate through (position, depth) tuples
        - Track current_depth, first_index, last_index
        - When depth changes, save range and start new level

        Parameters
        ----------
        position_depth_tuples : list
            List of (position, depth) tuples
        verbose : bool, optional
            If True, print algorithm steps

        Returns
        -------
        list
            List of [depth, first_index, last_index] for each level
        """
        level_ranges = []
        current_depth = None
        first_index = None
        last_index = None

        for position, depth in position_depth_tuples:
            if current_depth is None:
                # First iteration
                current_depth = depth
                first_index = position
                last_index = position
                if verbose:
                    print(f"  Start level {depth} at position {position}")

            elif depth == current_depth:
                # Same level continues
                last_index = position

            else:
                # Level changed: save previous level and start new one
                level_ranges.append([current_depth, first_index, last_index])
                if verbose:
                    print(
                        f"  Finish level {current_depth}: [{first_index}, {last_index}]"
                    )
                    print(f"  Start level {depth} at position {position}")

                current_depth = depth
                first_index = position
                last_index = position

        # Don't forget the last level
        if current_depth is not None:
            level_ranges.append([current_depth, first_index, last_index])
            if verbose:
                print(f"  Finish level {current_depth}: [{first_index}, {last_index}]")

        return level_ranges

    def verify_bfs_continuity(self, level_ranges, verbose=False):
        """
        Verify BFS property: last(depth N) + 1 == first(depth N+1).

        Two rules:
        1. Depths are sequential (no gaps: 0,1,2... not 0,2,4...)
        2. Levels are contiguous (no overlap or gap in positions)

        Parameters
        ----------
        level_ranges : list
            List of [depth, first, last] for each level
        verbose : bool, optional
            If True, print verification details

        Raises
        ------
        AssertionError
            If BFS continuity is violated
        """
        if verbose:
            print("\nBFS PROPERTY VERIFICATION:")
            print("-" * 80)

        # Rule 1: Depths should be sequential (no gaps)
        depths_in_order = [depth for depth, _, _ in level_ranges]
        for i in range(len(depths_in_order) - 1):
            expected_next_depth = depths_in_order[i] + 1
            actual_next_depth = depths_in_order[i + 1]
            assert (
                actual_next_depth == expected_next_depth
            ), f"Depth gap: depth {depths_in_order[i]} followed by {actual_next_depth}"

        if verbose:
            print("✓ Rule 1: Depths are sequential (no gaps)")

        # Rule 2: No overlap between levels (last + 1 == next first)
        for i in range(len(level_ranges) - 1):
            current_depth, _, current_last = level_ranges[i]
            next_depth, next_first, _ = level_ranges[i + 1]

            expected_next_first = current_last + 1

            if verbose:
                print(
                    f"  Depth {current_depth} ends at {current_last}, "
                    f"Depth {next_depth} starts at {next_first} "
                    f"(expected {expected_next_first})",
                    end="",
                )

            assert next_first == expected_next_first, (
                f"BFS continuity violated: "
                f"depth {current_depth} ends at {current_last}, "
                f"but depth {next_depth} starts at {next_first} "
                f"(expected {expected_next_first})"
            )

            if verbose:
                print(" ✓")

        if verbose:
            print("✓ Rule 2: Levels are contiguous (perfect continuity)")
            print("✓ All BFS properties verified")

    # ========================================================================
    # HELPER METHODS - Display
    # ========================================================================

    def print_level_ranges_table(self, level_ranges, nodes, verbose=True):
        """
        Print formatted table of level ranges with sample keys.

        Parameters
        ----------
        level_ranges : list
            List of [depth, first, last] for each level
        nodes : list
            List of collected nodes
        verbose : bool, optional
            If True, print the table
        """
        if not verbose:
            return

        print("\nLEVEL RANGES:")
        print("-" * 80)
        print(
            f"{'Depth':<10} {'First':<10} {'Last':<10} {'Count':<10} {'Sample Keys':<40}"
        )
        print("-" * 80)

        for depth, first, last in level_ranges:
            count = last - first + 1
            # Show sample keys from this level
            sample_keys = [
                str(nodes[i].key)[:15] for i in range(first, min(first + 3, last + 1))
            ]
            samples = ", ".join(sample_keys)
            if count > 3:
                samples += ", ..."
            print(f"{depth:<10} {first:<10} {last:<10} {count:<10} {samples:<40}")

        print("-" * 80)

    def print_test_header(self, title, verbose=True):
        """Print formatted test header."""
        if verbose:
            print("\n" + "=" * 80)
            print(f"TEST: {title}")
            print("=" * 80)

    def print_test_footer(self, verbose=True):
        """Print formatted test footer."""
        if verbose:
            print("=" * 80 + "\n")

    # ========================================================================
    # HELPER METHODS - Verification
    # ========================================================================

    def verify_parent_child_order(self, parent: Any, child: Any, message: str = None):
        """
        Helper to verify parent appears before child in keys_list.

        Parameters
        ----------
        parent : Any
            Parent key to check
        child : Any
            Child key to check
        message : str, optional
            Custom assertion message
        """
        assert parent in self.keys_list, f"Parent '{parent}' not found in traversal"
        assert child in self.keys_list, f"Child '{child}' not found in traversal"

        parent_idx = self.keys_list.index(parent)
        child_idx = self.keys_list.index(child)

        default_message = f"Parent {parent} should appear before child {child}"
        assert parent_idx < child_idx, message or default_message

    # ========================================================================
    # DFS PRE-ORDER TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "parent, child",
        [
            ("database", "host"),
            ("database", "port"),
            (frozenset(["cache", "redis"]), "config"),
            ("monitoring", ("metrics", "cpu")),
            ("global_settings", ("security", "encryption")),
        ],
    )
    def test_dfs_preorder_parent_before_child(self, key_tree, parent, child):
        """
        Verify that pre-order traversal visits parent before its children.

        In pre-order traversal:
        1. Visit current node
        2. Visit children left to right
        3. Recursively visit their descendants

        Example: for tree A -> [B, C], B -> [D]
        Expected order: A, B, D, C
        """
        collector = self.make_collector("key")
        list(key_tree.dfs_preorder(collector))
        self.verify_parent_child_order(parent, child)

    @pytest.mark.parametrize(
        "key_type, expected_keys",
        [
            (str, ["database", "api", "monitoring"]),
            (int, [1, 2, 42, 54, 12, 34]),
            (tuple, [("env", "production"), ("env", "dev")]),
        ],
    )
    def test_dfs_preorder_collect_by_type(self, key_tree, key_type, expected_keys):
        """
        Filter keys by type during pre-order traversal.

        Demonstrates using traversal to filter by key type.
        Useful for extracting only certain types (str, int, tuple, etc.)
        """
        filter_func = lambda node: isinstance(node.key, key_type)
        filtered_nodes = self.collect_nodes_with_traversal(
            key_tree, method="dfs_preorder", filter_func=filter_func
        )

        filtered_keys = [n.key for n in filtered_nodes]

        assert len(filtered_keys) > 0, f"Should find some {key_type.__name__} keys"

        # Check for at least some expected keys
        found_keys = [k for k in expected_keys if k in filtered_keys]
        assert (
            len(found_keys) > 0
        ), f"Should find some expected {key_type.__name__} keys"

    @pytest.mark.parametrize("min_levels", [2, 3])
    def test_dfs_preorder_depth_distribution(self, key_tree, min_levels):
        """
        Analyze node distribution by depth with DFS pre-order.

        Helps understand tree structure:
        - How many nodes at each level
        - Maximum depth
        - Tree balance
        """
        collector = self.make_collector("depth")
        list(key_tree.dfs_preorder(collector))

        assert (
            len(self.depth_map) >= min_levels
        ), f"Tree should have at least {min_levels} levels, got {len(self.depth_map)}"
        assert 0 in self.depth_map, "Should have nodes at level 0"
        assert len(self.depth_map[0]) > 0, "Level 0 should not be empty"

        # Verify depths are sequential (no gaps)
        max_depth = max(self.depth_map.keys())
        for d in range(max_depth + 1):
            assert (
                d in self.depth_map
            ), f"Level {d} should exist (no gaps in tree depth)"

    @pytest.mark.parametrize(
        "parent_key, child_key",
        [
            ("database", "host"),
            ("replicas", "region"),
        ],
    )
    def test_dfs_preorder_path_consistency(self, key_tree, parent_key, child_key):
        """
        Verify path consistency: children have longer paths than parents.

        Creates an index to quickly find complete access path
        to any key in the original dictionary.
        """
        collector = self.make_collector("path")
        list(key_tree.dfs_preorder(collector))

        assert parent_key in self.path_map, f"Parent '{parent_key}' should be in tree"
        assert child_key in self.path_map, f"Child '{child_key}' should be in tree"

        parent_path = self.path_map[parent_key]
        child_path = self.path_map[child_key]

        # If child is descendant of parent, its path should be longer
        if parent_key in child_path:
            assert len(child_path) > len(
                parent_path
            ), f"Child '{child_key}' should have longer path than parent '{parent_key}'"

    def test_dfs_preorder_negative_nonexistent_key(self, key_tree):
        """
        Negative test: verify behavior when searching for non-existent key.
        """
        collector = self.make_collector("key")
        list(key_tree.dfs_preorder(collector))

        non_existent_keys = ["nonexistent", "fake_key", "does_not_exist"]
        for key in non_existent_keys:
            assert (
                key not in self.keys_list
            ), f"Non-existent key '{key}' should not be in traversal"

    # ========================================================================
    # DFS POST-ORDER TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "child, parent",
        [
            ("host", "database"),
            ("port", "database"),
            ("config", frozenset(["cache", "redis"])),
            ("ttl", "config"),
        ],
    )
    def test_dfs_postorder_child_before_parent(self, key_tree, child, parent):
        """
        Verify that post-order traversal visits children before parent.

        Post-order is crucial for:
        - Node deletion (delete children before parent)
        - Bottom-up calculations (aggregations, sums, etc.)
        - Resource cleanup

        Example: for A -> [B, C], B -> [D]
        Expected order: D, B, C, A
        """
        collector = self.make_collector("key")
        list(key_tree.dfs_postorder(collector))

        assert child in self.keys_list, f"Child '{child}' not found in traversal"
        assert parent in self.keys_list, f"Parent '{parent}' not found in traversal"

        child_idx = self.keys_list.index(child)
        parent_idx = self.keys_list.index(parent)

        assert (
            child_idx < parent_idx
        ), f"Child {child} should appear before parent {parent}"

    @pytest.mark.parametrize(
        "key_with_children",
        [
            "database",
            "replicas",
            "global_settings",
        ],
    )
    def test_dfs_postorder_subtree_size(self, key_tree, key_with_children):
        """
        Calculate subtree size in post-order (bottom-up).

        Post-order is ideal for this calculation as we process leaves first,
        then accumulate sizes upward.

        Node size = 1 (itself) + sum of children sizes
        """
        subtree_sizes = {}

        def calculate_size(node):
            if not node.is_root:
                size = 1 + sum(
                    subtree_sizes.get(child.key, 0) for child in node.children
                )
                subtree_sizes[node.key] = size

        list(key_tree.dfs_postorder(calculate_size))

        assert (
            key_with_children in subtree_sizes
        ), f"Key '{key_with_children}' should be in tree"

        # Node with children should have size > 1
        assert (
            subtree_sizes[key_with_children] > 1
        ), f"'{key_with_children}' should have subtree size > 1, got {subtree_sizes[key_with_children]}"

    def test_dfs_postorder_leaf_counting(self, key_tree):
        """
        Count leaves in each subtree (bottom-up aggregation).

        For each node, calculate how many leaves exist in its subtree.
        Classic example of post-order aggregation.
        """
        leaf_counts = {}

        for node in key_tree.dfs_postorder():
            if not node.is_root:
                if node.is_leaf():
                    leaf_counts[node.key] = 1
                else:
                    leaf_counts[node.key] = sum(
                        leaf_counts.get(child.key, 0) for child in node.children
                    )

        assert len(leaf_counts) > 0, "Should count some leaves"

        # Some internal nodes should have multiple leaves
        multi_leaf_nodes = [k for k, v in leaf_counts.items() if v > 1]
        assert (
            len(multi_leaf_nodes) > 0
        ), "Should have nodes with multiple leaves in subtree"

    # ========================================================================
    # BFS TESTS - Basic Traversal
    # ========================================================================

    @pytest.mark.parametrize("depth", [0, 1, 2])
    def test_bfs_level_order_traversal(self, key_tree, depth):
        """
        Verify that BFS traverses the tree level by level.

        BFS explores all nodes at depth N before exploring
        nodes at depth N+1.

        Useful for:
        - Finding shortest path
        - Exploring by proximity
        - Analyzing structure by levels
        """
        collector = self.make_collector("depth")
        list(key_tree.bfs(collector))

        assert len(self.depth_map) >= 2, "Should have at least 2 levels"

        # Check specific depth level if it exists
        if depth in self.depth_map:
            assert len(self.depth_map[depth]) > 0, f"Level {depth} should not be empty"

    @pytest.mark.parametrize("target_key", ["status", "replicas", "instances"])
    def test_bfs_finds_shallowest_node(self, key_tree, target_key):
        """
        BFS guarantees finding the node closest to root first.

        If multiple nodes have the same key, BFS finds the one
        with smallest depth.
        """
        first_occurrence = None

        for node in key_tree.bfs():
            if node.key == target_key:
                first_occurrence = node
                break

        if first_occurrence:
            depth = first_occurrence.get_depth()

            # Find all occurrences
            all_occurrences = [
                n for n in key_tree.dfs_preorder() if n.key == target_key
            ]

            if len(all_occurrences) > 1:
                # First found by BFS should be shallowest
                min_depth = min(n.get_depth() for n in all_occurrences)
                assert (
                    depth == min_depth
                ), f"BFS should find shallowest '{target_key}' at depth {min_depth}, got {depth}"

    @pytest.mark.parametrize(
        "depth, expected_properties",
        [
            (0, {"min_path_length": 1, "has_children": True}),
            (1, {"min_path_length": 2}),
        ],
    )
    def test_bfs_level_properties(self, key_tree, depth, expected_properties):
        """
        Verify properties of nodes at specific depth levels.

        Creates detailed map of each level for structure analysis.
        """
        level_nodes = []

        for node in key_tree.bfs():
            if not node.is_root and node.get_depth() == depth:
                level_nodes.append(
                    {
                        "key": node.key,
                        "path": node.get_path(),
                        "children_count": len(node.children),
                        "is_leaf": node.is_leaf(),
                    }
                )

        if level_nodes:
            # Verify path length
            if "min_path_length" in expected_properties:
                min_expected = expected_properties["min_path_length"]
                for node_info in level_nodes:
                    assert (
                        len(node_info["path"]) >= min_expected
                    ), f"Path length should be >= {min_expected} at depth {depth}"

            # Verify children existence
            if expected_properties.get("has_children", False):
                has_children_nodes = [n for n in level_nodes if n["children_count"] > 0]
                assert (
                    len(has_children_nodes) > 0
                ), f"Some nodes at depth {depth} should have children"

    def test_bfs_negative_excessive_depth(self, key_tree):
        """
        Negative test: verify no nodes exist at unreasonably deep levels.
        """
        excessive_depth = 100

        for node in key_tree.bfs():
            if not node.is_root:
                assert (
                    node.get_depth() < excessive_depth
                ), f"Tree should not have nodes at depth >= {excessive_depth}"

    # ========================================================================
    # BFS TESTS - With Callbacks
    # ========================================================================

    def test_bfs_with_callback_collection(self, key_tree):
        """
        Verify BFS traversal with callback function using rigorous algorithm.

        Algorithm:
        1. Collect all nodes during BFS traversal
        2. Build (position, depth) tuples
        3. Build [depth, first, last] ranges
        4. Verify continuity: last(N) + 1 == first(N+1)
        """
        self.print_test_header("BFS WITH CALLBACK COLLECTION")

        # Step 1: Collect all nodes
        collected_nodes = self.collect_nodes_with_traversal(key_tree, method="bfs")
        assert len(collected_nodes) > 0, "Should collect nodes via BFS callback"

        # Step 2: Build position-depth tuples
        pos_depth_tuples = self.build_position_depth_tuples(
            collected_nodes, verbose=True
        )

        # Step 3: Build level ranges
        level_ranges = self.build_level_ranges(pos_depth_tuples, verbose=False)
        self.print_level_ranges_table(level_ranges, collected_nodes, verbose=True)

        # Step 4: Verify BFS properties
        self.verify_bfs_continuity(level_ranges, verbose=True)

        self.print_test_footer()

    @pytest.mark.parametrize("target_depth", [0, 1, 2])
    def test_bfs_callback_depth_filtering(self, key_tree, target_depth):
        """
        Use BFS callback to collect only nodes at specific depth.

        Demonstrates conditional collection during BFS traversal.
        This is useful for extracting a specific "layer" of the tree.
        """
        self.print_test_header(f"BFS DEPTH FILTERING (target_depth={target_depth})")

        # Collect only nodes at target depth
        filter_func = lambda node: node.get_depth() == target_depth
        collected_nodes = self.collect_nodes_with_traversal(
            key_tree, method="bfs", filter_func=filter_func
        )

        # Should find nodes at common depths
        if target_depth <= 3:
            assert (
                len(collected_nodes) > 0
            ), f"Should find nodes at depth {target_depth}"

        print(f"\nCollected {len(collected_nodes)} nodes at depth {target_depth}")

        # Verify all collected nodes are at target depth
        for node in collected_nodes:
            assert (
                node.get_depth() == target_depth
            ), f"Node '{node.key}' should be at depth {target_depth}, got {node.get_depth()}"

        # Show sample keys
        if collected_nodes:
            print(f"\nSample keys at depth {target_depth}:")
            for i, node in enumerate(collected_nodes[:10]):
                print(f"  {i}: {node.key}")
            if len(collected_nodes) > 10:
                print(f"  ... ({len(collected_nodes) - 10} more)")

        self.print_test_footer()

    def test_bfs_callback_statistics_collection(self, key_tree):
        """
        Use BFS callback to collect statistics during traversal.

        Demonstrates using callback for real-time analysis during traversal,
        without needing to iterate over nodes multiple times.
        """
        self.print_test_header("BFS STATISTICS COLLECTION")

        stats = {
            "total_nodes": 0,
            "leaves": 0,
            "internal": 0,
            "max_children": 0,
            "depth_distribution": {},
        }

        def collect_stats(node):
            if not node.is_root:
                stats["total_nodes"] += 1

                if node.is_leaf():
                    stats["leaves"] += 1
                else:
                    stats["internal"] += 1

                children_count = len(node.children)
                if children_count > stats["max_children"]:
                    stats["max_children"] = children_count

                depth = node.get_depth()
                stats["depth_distribution"][depth] = (
                    stats["depth_distribution"].get(depth, 0) + 1
                )

        list(key_tree.bfs(collect_stats))

        # Print collected statistics
        print("\nCOLLECTED STATISTICS:")
        print("-" * 60)
        print(f"  Total nodes:       {stats['total_nodes']}")
        print(f"  Leaf nodes:        {stats['leaves']}")
        print(f"  Internal nodes:    {stats['internal']}")
        print(f"  Max children:      {stats['max_children']}")
        print(f"  Depth levels:      {len(stats['depth_distribution'])}")
        print("\n  Depth distribution:")
        for depth in sorted(stats["depth_distribution"].keys()):
            count = stats["depth_distribution"][depth]
            print(f"    Depth {depth}: {count} nodes")

        # Verify collected statistics are meaningful
        assert stats["total_nodes"] > 0, "Should count some nodes"
        assert stats["leaves"] > 0, "Should have some leaf nodes"
        assert stats["internal"] > 0, "Should have some internal nodes"
        assert stats["max_children"] >= 0, "Should track max children count"
        assert (
            len(stats["depth_distribution"]) >= 2
        ), "Should have multiple depth levels"

        # Verify consistency: total = leaves + internal
        assert stats["total_nodes"] == stats["leaves"] + stats["internal"], (
            f"Total ({stats['total_nodes']}) should equal leaves + internal "
            f"({stats['leaves']} + {stats['internal']})"
        )

        # Verify sum of depth distribution equals total nodes
        total_from_depths = sum(stats["depth_distribution"].values())
        assert (
            total_from_depths == stats["total_nodes"]
        ), f"Sum of nodes per depth ({total_from_depths}) should equal total ({stats['total_nodes']})"

        print("\n✓ All statistics verified for consistency")
        self.print_test_footer()

    @pytest.mark.parametrize("max_depth", [1, 2, 3])
    def test_bfs_callback_early_termination_simulation(self, key_tree, max_depth):
        """
        Simulate early termination by stopping collection at a certain depth.

        While BFS itself doesn't stop, we can use the callback to only
        collect nodes up to a certain depth, simulating a bounded search.
        """
        self.print_test_header(f"BFS EARLY TERMINATION (max_depth={max_depth})")

        # Collect only nodes up to max_depth
        filter_func = lambda node: node.get_depth() <= max_depth
        collected_nodes = self.collect_nodes_with_traversal(
            key_tree, method="bfs", filter_func=filter_func
        )

        assert len(collected_nodes) > 0, f"Should collect nodes up to depth {max_depth}"

        # Build level ranges to verify structure
        pos_depth_tuples = self.build_position_depth_tuples(
            collected_nodes, verbose=False
        )
        level_ranges = self.build_level_ranges(pos_depth_tuples, verbose=False)

        print(f"\nCollected {len(collected_nodes)} nodes up to depth {max_depth}")
        self.print_level_ranges_table(level_ranges, collected_nodes, verbose=True)

        # Verify no collected node exceeds max_depth
        for node in collected_nodes:
            assert (
                node.get_depth() <= max_depth
            ), f"Node '{node.key}' should be at depth <= {max_depth}, got {node.get_depth()}"

        # Verify BFS properties still hold for collected subset
        print("\nVerifying BFS properties on collected subset:")
        print("-" * 60)
        self.verify_bfs_continuity(level_ranges, verbose=True)

        self.print_test_footer()

    def test_bfs_callback_vs_iteration_equivalence(self, key_tree):
        """
        Verify that using callback gives same results as manual iteration.

        Both approaches should visit nodes in the same order.
        """
        self.print_test_header("BFS CALLBACK VS ITERATION EQUIVALENCE")

        # Method 1: Using callback
        nodes_via_callback = self.collect_nodes_with_traversal(key_tree, method="bfs")

        # Method 2: Manual iteration
        nodes_via_iteration = []
        for node in key_tree.bfs():
            if not node.is_root:
                nodes_via_iteration.append(node)

        # Both methods should give identical results
        assert len(nodes_via_callback) == len(nodes_via_iteration), (
            f"Callback ({len(nodes_via_callback)}) and iteration ({len(nodes_via_iteration)}) "
            f"should visit same number of nodes"
        )

        print(f"\nBoth methods collected {len(nodes_via_callback)} nodes")

        # Verify same nodes in same order (compare by identity)
        for i, (n1, n2) in enumerate(zip(nodes_via_callback, nodes_via_iteration)):
            assert n1 is n2, (
                f"Node mismatch at position {i}: "
                f"callback has {n1.key}, iteration has {n2.key}"
            )

        # Build and compare level ranges
        tuples_callback = self.build_position_depth_tuples(
            nodes_via_callback, verbose=False
        )
        tuples_iteration = self.build_position_depth_tuples(
            nodes_via_iteration, verbose=False
        )

        ranges_callback = self.build_level_ranges(tuples_callback, verbose=False)
        ranges_iteration = self.build_level_ranges(tuples_iteration, verbose=False)

        assert (
            ranges_callback == ranges_iteration
        ), "Level ranges should be identical for both methods"

        print("\n✓ Callback and iteration produce identical results")
        print("✓ Same nodes in same order")
        print("✓ Same level structure")

        self.print_test_footer()

    # ========================================================================
    # COMPARATIVE TESTS
    # ========================================================================

    def test_all_traversals_visit_same_nodes(self, key_tree):
        """
        Verify that all three traversals visit exactly the same nodes.

        Only the visit order changes, not the set of visited nodes.
        """
        preorder_nodes = set(id(n) for n in key_tree.dfs_preorder())
        postorder_nodes = set(id(n) for n in key_tree.dfs_postorder())
        bfs_nodes = set(id(n) for n in key_tree.bfs())

        assert (
            preorder_nodes == postorder_nodes
        ), "Pre-order and post-order should visit same nodes"
        assert preorder_nodes == bfs_nodes, "DFS and BFS should visit same nodes"

    @pytest.mark.parametrize("min_count", [10, 20])
    def test_traversal_node_counts(self, key_tree, min_count):
        """
        Compare total number of nodes visited by each algorithm.
        """
        preorder_count = len(list(key_tree.dfs_preorder()))
        postorder_count = len(list(key_tree.dfs_postorder()))
        bfs_count = len(list(key_tree.bfs()))

        assert (
            preorder_count == postorder_count == bfs_count
        ), f"All traversals should visit same count: pre={preorder_count}, post={postorder_count}, bfs={bfs_count}"
        assert (
            preorder_count >= min_count
        ), f"Tree should have at least {min_count} nodes, got {preorder_count}"


# ============================================================================
# Tests for search algorithms
# ============================================================================


class TestTreeSearch:
    """
    Tests for tree search algorithms.

    Covers:
    - dfs_find: depth-first search (first match)
    - bfs_find: breadth-first search (closest match)
    - find_all: finds all matching nodes
    """

    # Shared attribute for collecting search results
    found_nodes = []

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset attributes before each test to avoid side effects."""
        self.found_nodes = []

    def verify_shallowest_node(self, node, key_tree, key: Any):
        """
        Helper to verify that a node is the shallowest occurrence of a key.

        Parameters
        ----------
        node : _HKey
            Node to verify
        key_tree : _HKey
            Tree root for searching
        key : Any
            Key to search for
        """
        all_matches = key_tree.find_all(lambda n: n.key == key)

        if len(all_matches) > 1:
            depths = [n.get_depth() for n in all_matches]
            min_depth = min(depths)
            assert (
                node.get_depth() == min_depth
            ), f"Node should be at shallowest depth {min_depth}, got {node.get_depth()}"

    # ------------------------------------------------------------------------
    # DFS Find Tests
    # ------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "key, expected_in_path",
        [
            ("port", "database"),
            ("host", "database"),
            ("timeout", "api"),
            ("ttl", "config"),
        ],
    )
    def test_dfs_find_simple_key(self, key_tree, key, expected_in_path):
        """
        DFS search for a simple key.

        DFS finds first matching node by exploring depth-first
        (goes completely down one branch before exploring the next).
        """
        result = key_tree.dfs_find(lambda n: n.key == key)

        assert result is not None, f"Should find key '{key}'"
        assert result.key == key, f"Found node should have key '{key}'"

        path = result.get_path()
        assert len(path) > 0, "Node should have a non-empty path"
        assert (
            expected_in_path in path
        ), f"'{expected_in_path}' should be in path of '{key}': {path}"

    @pytest.mark.parametrize(
        "min_depth, should_exist",
        [
            (2, True),
            (3, True),
            (10, False),
        ],
    )
    def test_dfs_find_by_depth_condition(self, key_tree, min_depth, should_exist):
        """
        Find node according to depth criterion.

        Example: find first node at depth >= min_depth.
        Useful for analyzing deep levels of the tree.
        """
        result = key_tree.dfs_find(
            lambda n: not n.is_root and n.get_depth() >= min_depth
        )

        if should_exist:
            assert result is not None, f"Should find node at depth >= {min_depth}"
            assert (
                result.get_depth() >= min_depth
            ), f"Found node depth should be >= {min_depth}, got {result.get_depth()}"
        else:
            assert (
                result is None
            ), f"Should not find node at excessive depth {min_depth}"

    @pytest.mark.parametrize("parent_key", ["api", "config", "replicas"])
    def test_dfs_find_leaf_with_specific_parent(self, key_tree, parent_key):
        """
        Combined search: leaf having a specific parent.

        Demonstrates using complex predicates combining
        multiple conditions (node type + parent relationship).
        """
        result = key_tree.dfs_find(
            lambda n: (
                not n.is_root
                and n.is_leaf()
                and n.parent is not None
                and n.parent.key == parent_key
            )
        )

        if result:
            assert result.is_leaf(), f"Found node should be a leaf"
            assert result.parent is not None, "Leaf should have a parent"
            assert (
                result.parent.key == parent_key
            ), f"Parent should be '{parent_key}', got '{result.parent.key}'"

    @pytest.mark.parametrize(
        "key_type, should_contain",
        [
            (tuple, "env"),
            (frozenset, "cache"),
        ],
    )
    def test_dfs_find_complex_key_types(self, key_tree, key_type, should_contain):
        """
        Search for complex key types (tuple, frozenset).

        Stacked dictionaries often use tuples or frozensets as keys.
        """
        result = key_tree.dfs_find(
            lambda n: isinstance(n.key, key_type) and should_contain in str(n.key)
        )

        if result:
            assert isinstance(
                result.key, key_type
            ), f"Key should be {key_type.__name__}, got {type(result.key).__name__}"
            assert should_contain in str(
                result.key
            ), f"Key should contain '{should_contain}': {result.key}"

    @pytest.mark.parametrize("min_children", [2, 3, 4, 5])
    def test_dfs_find_by_children_count(self, key_tree, min_children):
        """
        Find node having specific number of children.

        Useful for identifying "hub" nodes or important branches.
        """
        result = key_tree.dfs_find(
            lambda n: not n.is_root and len(n.children) >= min_children
        )

        if result:
            assert (
                len(result.children) >= min_children
            ), f"Found node should have >= {min_children} children, got {len(result.children)}"

    def test_dfs_find_negative_impossible_condition(self, key_tree):
        """
        Negative test: search with impossible condition returns None.
        """
        # Search for node that is both leaf and has children (impossible)
        result = key_tree.dfs_find(
            lambda n: not n.is_root and n.is_leaf() and len(n.children) > 0
        )

        assert result is None, "Should not find node that is both leaf and has children"

    # ------------------------------------------------------------------------
    # BFS Find Tests
    # ------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "target_key", ["replicas", "instances", "database", "monitoring"]
    )
    def test_bfs_find_shallowest_match(self, key_tree, target_key):
        """
        BFS guarantees finding match closest to root.

        If multiple nodes match the criterion, BFS finds
        the one with smallest depth.
        """
        result = key_tree.bfs_find(lambda n: n.key == target_key)

        if result:
            assert result.key == target_key, f"Found key should be '{target_key}'"
            self.verify_shallowest_node(result, key_tree, target_key)

    @pytest.mark.parametrize("threshold", [3, 4, 5, 6])
    def test_bfs_find_first_hub_node(self, key_tree, threshold):
        """
        BFS to find first "hub" node (many children).

        Finds node closest to root with many branches.
        Useful for identifying divergence points.
        """
        result = key_tree.bfs_find(
            lambda n: not n.is_root and len(n.children) >= threshold
        )

        if result:
            assert (
                len(result.children) >= threshold
            ), f"Hub node should have >= {threshold} children"

            # Verify it's the closest to root with this property
            all_hubs = key_tree.find_all(
                lambda n: not n.is_root and len(n.children) >= threshold
            )
            if len(all_hubs) > 1:
                min_depth = min(n.get_depth() for n in all_hubs)
                assert (
                    result.get_depth() == min_depth
                ), f"Should find hub closest to root at depth {min_depth}"

    @pytest.mark.parametrize(
        "key_type, max_expected_depth",
        [
            (frozenset, 2),
            (tuple, 3),
        ],
    )
    def test_bfs_find_by_key_type(self, key_tree, key_type, max_expected_depth):
        """
        BFS search for specific key types.

        Frozensets and tuples are used as composite keys
        in stacked dictionaries.
        """
        result = key_tree.bfs_find(lambda n: isinstance(n.key, key_type))

        if result:
            assert isinstance(
                result.key, key_type
            ), f"Key should be {key_type.__name__}"
            assert (
                result.get_depth() <= max_expected_depth
            ), f"{key_type.__name__} keys should be at depth <= {max_expected_depth}, got {result.get_depth()}"

    @pytest.mark.parametrize("target_key", ["database", "monitoring", "api", "config"])
    def test_bfs_vs_dfs_find_comparison(self, key_tree, target_key):
        """
        Compare BFS and DFS results for same search.

        If only one node matches, both should find it.
        If multiple match, BFS finds closest to root.
        """
        bfs_result = key_tree.bfs_find(lambda n: n.key == target_key)
        dfs_result = key_tree.dfs_find(lambda n: n.key == target_key)

        if bfs_result and dfs_result:
            assert bfs_result.key == dfs_result.key == target_key

            # BFS should find closest (or equal)
            assert (
                bfs_result.get_depth() <= dfs_result.get_depth()
            ), f"BFS depth ({bfs_result.get_depth()}) should be <= DFS depth ({dfs_result.get_depth()})"

    def test_bfs_find_negative_nonexistent_pattern(self, key_tree):
        """
        Negative test: BFS should return None for non-existent pattern.
        """
        result = key_tree.bfs_find(
            lambda n: isinstance(n.key, str) and n.key.startswith("nonexistent_")
        )

        assert result is None, "Should not find node with non-existent pattern"

    # ------------------------------------------------------------------------
    # Find All Tests
    # ------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "key_type, min_expected",
        [
            (int, 4),
            (str, 10),
            (tuple, 2),
        ],
    )
    def test_find_all_by_key_type(self, key_tree, key_type, min_expected):
        """
        Find all nodes with specific key type.

        In the dictionary, different types serve different purposes:
        - int: IDs and indices
        - str: standard keys
        - tuple: composite keys
        """
        self.found_nodes = key_tree.find_all(
            lambda n: not n.is_root and isinstance(n.key, key_type)
        )

        assert (
            len(self.found_nodes) >= min_expected
        ), f"Should find at least {min_expected} {key_type.__name__} keys, got {len(self.found_nodes)}"
        assert all(
            isinstance(n.key, key_type) for n in self.found_nodes
        ), f"All found keys should be {key_type.__name__}"

    def test_find_all_leaves(self, key_tree):
        """
        Find all tree leaves.

        Leaves are terminal nodes (no children).
        Represent "final values" in the dictionary.
        """
        self.found_nodes = key_tree.find_all(lambda n: not n.is_root and n.is_leaf())

        assert len(self.found_nodes) > 0, "Tree should have leaves"
        assert all(
            n.is_leaf() for n in self.found_nodes
        ), "All found nodes should be leaves"
        assert all(
            len(n.children) == 0 for n in self.found_nodes
        ), "Leaves should have no children"

    @pytest.mark.parametrize(
        "target_depth, min_expected",
        [
            (0, 3),
            (1, 5),
            (2, 10),
        ],
    )
    def test_find_all_at_specific_depth(self, key_tree, target_depth, min_expected):
        """
        Find all nodes at given depth (horizontal slice).

        Useful for analyzing a "slice" of tree at given level.
        """
        self.found_nodes = key_tree.find_all(
            lambda n: not n.is_root and n.get_depth() == target_depth
        )

        if self.found_nodes:
            assert all(
                n.get_depth() == target_depth for n in self.found_nodes
            ), f"All nodes should be at depth {target_depth}"
            assert (
                len(self.found_nodes) >= min_expected
            ), f"Should find at least {min_expected} nodes at depth {target_depth}, got {len(self.found_nodes)}"

    @pytest.mark.parametrize("pattern", ["env", "cache", "metrics", "security"])
    def test_find_all_keys_with_pattern(self, key_tree, pattern):
        """
        Find all keys (any type) containing specific pattern.

        Works with tuples, strings, and other key types.
        """
        self.found_nodes = key_tree.find_all(
            lambda n: not n.is_root and pattern in str(n.key)
        )

        if self.found_nodes:
            assert all(
                pattern in str(n.key) for n in self.found_nodes
            ), f"All found keys should contain '{pattern}'"

    @pytest.mark.parametrize("child_count", [1, 2, 3])
    def test_find_all_by_children_count(self, key_tree, child_count):
        """
        Find all nodes with specific number of children.

        Useful for analyzing tree structure and branching patterns.
        """
        self.found_nodes = key_tree.find_all(
            lambda n: not n.is_root and len(n.children) == child_count
        )

        if self.found_nodes:
            assert all(len(n.children) == child_count for n in self.found_nodes)

    def test_find_all_internal_nodes(self, key_tree):
        """
        Find all internal nodes (neither root nor leaf).

        Internal nodes represent groupings/categories
        in dictionary structure.
        """
        self.found_nodes = key_tree.find_all(
            lambda n: not n.is_root and not n.is_leaf()
        )

        assert len(self.found_nodes) > 0, "Should have internal nodes"
        assert all(len(n.children) > 0 for n in self.found_nodes)

    def test_find_all_count_consistency(self, key_tree):
        """
        Verify counting consistency: total = leaves + internal.
        """
        all_nodes = key_tree.find_all(lambda n: not n.is_root)
        leaves = key_tree.find_all(lambda n: not n.is_root and n.is_leaf())
        internal = key_tree.find_all(lambda n: not n.is_root and not n.is_leaf())

        assert len(all_nodes) == len(leaves) + len(
            internal
        ), "Total nodes should equal leaves + internal nodes"


# ============================================================================
# Tests for transformation functions
# ============================================================================


class TestTreeTransformation:
    """
    Tests for tree transformation functions.

    Covers:
    - map_nodes: applies function to each node
    - filter_paths: filters paths according to criterion
    """

    # Shared attribute for collecting transformed data
    mapped_data = []
    filtered_paths = []

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Reset attributes before each test to avoid side effects."""
        self.mapped_data = []
        self.filtered_paths = []

    # ------------------------------------------------------------------------
    # Map Nodes Tests
    # ------------------------------------------------------------------------

    def test_map_nodes_extract_keys(self, key_tree):
        """
        Simple extraction: retrieve all keys.

        Most basic map function: transform each node into its key.
        """
        self.mapped_data = key_tree.map_nodes(
            lambda n: n.key if not n.is_root else None
        )

        keys = [k for k in self.mapped_data if k is not None]
        assert len(keys) > 0, "Should extract some keys"

        # Check for string keys
        string_keys = [k for k in keys if isinstance(k, str)]
        assert len(string_keys) > 0, "Should have string keys"

    @pytest.mark.parametrize("include_depth", [True, False])
    def test_map_nodes_create_pairs(self, key_tree, include_depth):
        """
        Create (key, depth) or (key, path) pairs for analysis.

        Transform each node into tuple containing metadata.
        """
        if include_depth:
            mapper = lambda n: (n.key, n.get_depth()) if not n.is_root else None
        else:
            mapper = lambda n: (n.key, tuple(n.get_path())) if not n.is_root else None

        self.mapped_data = key_tree.map_nodes(mapper)
        valid_pairs = [p for p in self.mapped_data if p is not None]

        assert len(valid_pairs) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in valid_pairs)

    @pytest.mark.parametrize("metric_key", ["path_length", "children_count", "is_leaf"])
    def test_map_nodes_calculate_metrics(self, key_tree, metric_key):
        """
        Calculate specific metrics for each node.

        Creates dictionary of statistics per node.
        """
        self.mapped_data = key_tree.map_nodes(
            lambda n: (
                {
                    "key": n.key,
                    "depth": n.get_depth(),
                    "is_leaf": n.is_leaf(),
                    "children_count": len(n.children),
                    "path_length": len(n.get_path()),
                }
                if not n.is_root
                else None
            )
        )

        metrics = [m for m in self.mapped_data if m is not None]

        assert len(metrics) > 0
        assert all(metric_key in m for m in metrics)

        # Verify consistency for specific metrics
        for metric in metrics:
            if metric_key == "path_length":
                assert metric["path_length"] == metric["depth"] + 1
            elif metric_key == "is_leaf":
                if metric["is_leaf"]:
                    assert metric["children_count"] == 0

    @pytest.mark.parametrize("format_style", ["arrows", "dots", "slashes"])
    def test_map_nodes_format_paths(self, key_tree, format_style):
        """
        Format paths of all nodes as strings with different separators.

        Useful for display or logging.
        """
        separators = {"arrows": " → ", "dots": ".", "slashes": "/"}
        separator = separators[format_style]

        self.mapped_data = key_tree.map_nodes(
            lambda n: (
                separator.join(str(k) for k in n.get_path()) if not n.is_root else None
            )
        )

        paths = [p for p in self.mapped_data if p is not None]

        assert len(paths) > 0
        # Verify formatted strings contain separator
        multi_level_paths = [p for p in paths if separator in p]
        assert len(multi_level_paths) > 0

    @pytest.mark.parametrize(
        "transformation",
        [
            lambda n: f"LEAF: {n.key}" if n.is_leaf() else f"BRANCH: {n.key}",
            lambda n: n.key if isinstance(n.key, str) else str(n.key),
            lambda n: (n.key, type(n.key).__name__),
        ],
    )
    def test_map_nodes_conditional_transformation(self, key_tree, transformation):
        """
        Apply different transformations based on node type.

        Example: special treatment for leaves vs internal nodes.
        """
        results = key_tree.map_nodes(
            lambda n: transformation(n) if not n.is_root else None
        )
        results = [r for r in results if r is not None]

        assert len(results) > 0

    # ------------------------------------------------------------------------
    # Filter Paths Tests
    # ------------------------------------------------------------------------

    @pytest.mark.parametrize("min_length", [2, 3, 4])
    def test_filter_paths_by_length(self, key_tree, min_length):
        """
        Filter paths based on their length.

        Find all paths longer than specified threshold.
        Useful for finding deeply nested structures.
        """
        long_paths = key_tree.filter_paths(lambda p: len(p) >= min_length)

        if long_paths:
            assert all(len(p) >= min_length for p in long_paths)

    @pytest.mark.parametrize("required_key", ["database", "api", "monitoring"])
    def test_filter_paths_containing_key(self, key_tree, required_key):
        """
        Find paths containing specific key.

        Useful for tracking all paths leading through a particular node.
        """
        paths_with_key = key_tree.filter_paths(lambda p: required_key in p)

        if paths_with_key:
            assert all(required_key in path for path in paths_with_key)

    def test_filter_paths_by_key_type(self, key_tree):
        """
        Filter paths containing specific key types.

        Example: find all paths with tuple keys.
        """
        tuple_paths = key_tree.filter_paths(
            lambda p: any(isinstance(k, tuple) for k in p)
        )

        if tuple_paths:
            for path in tuple_paths:
                assert any(isinstance(k, tuple) for k in path)

    @pytest.mark.parametrize(
        "start_key", ["database", "monitoring", frozenset(["cache", "redis"])]
    )
    def test_filter_paths_starting_with(self, key_tree, start_key):
        """
        Filter paths starting with specific key.

        Useful for extracting all paths under a particular subtree.
        """
        filtered_paths = key_tree.filter_paths(
            lambda p: p[0] == start_key if p else False
        )

        if filtered_paths:
            assert all(path[0] == start_key for path in filtered_paths)

    def test_filter_paths_leaves_only(self, key_tree):
        """
        Filter to get only paths leading to leaves.

        Combines filter_paths with leaf detection to find
        all terminal paths in the tree.
        """
        all_paths = key_tree.get_all_paths()
        leaf_paths = key_tree.filter_paths(
            lambda p: (
                key_tree.find_by_path(p).is_leaf()
                if key_tree.find_by_path(p)
                else False
            )
        )

        if leaf_paths:
            # Verify all filtered paths lead to leaves
            for path in leaf_paths:
                node = key_tree.find_by_path(path)
                assert node is not None, f"Path {path} should exist"
                assert node.is_leaf(), f"Path {path} should lead to leaf"
