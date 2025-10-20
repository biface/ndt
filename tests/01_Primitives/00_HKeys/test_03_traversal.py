"""
Tests for tree traversal algorithms in _HKey class.

These tests use a key tree (key_tree) built from a complex stacked dictionary
(_StackedDict) representing a multi-environment configuration.
"""

from typing import Any, List, Optional

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

    def collect_key(self, node):
        """
        Callback to collect node keys during traversal.

        Used as visit function for traversal methods.
        """
        if not node.is_root:
            self.keys_list.append(node.key)

    def collect_depth(self, node):
        """
        Callback to collect nodes by depth level.
        """
        if not node.is_root:
            depth = node.get_depth()
            if depth not in self.depth_map:
                self.depth_map[depth] = []
            self.depth_map[depth].append(node.key)

    def collect_path(self, node):
        """
        Callback to collect complete paths for each node.
        """
        if not node.is_root:
            self.path_map[node.key] = tuple(node.get_path())

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

    # ------------------------------------------------------------------------
    # DFS Pre-order Tests
    # ------------------------------------------------------------------------

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
        list(key_tree.dfs_preorder(self.collect_key))
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

        def collect_by_type(node):
            if not node.is_root and isinstance(node.key, key_type):
                self.keys_list.append(node.key)

        list(key_tree.dfs_preorder(collect_by_type))

        assert len(self.keys_list) > 0, f"Should find some {key_type.__name__} keys"

        # Check for at least some expected keys
        found_keys = [k for k in expected_keys if k in self.keys_list]
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
        list(key_tree.dfs_preorder(self.collect_depth))

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
        list(key_tree.dfs_preorder(self.collect_path))

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
        list(key_tree.dfs_preorder(self.collect_key))

        non_existent_keys = ["nonexistent", "fake_key", "does_not_exist"]
        for key in non_existent_keys:
            assert (
                key not in self.keys_list
            ), f"Non-existent key '{key}' should not be in traversal"

    # ------------------------------------------------------------------------
    # DFS Post-order Tests
    # ------------------------------------------------------------------------

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
        list(key_tree.dfs_postorder(self.collect_key))

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

    # ------------------------------------------------------------------------
    # BFS (Breadth-First Search) Tests
    # ------------------------------------------------------------------------

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
        for node in key_tree.bfs():
            if not node.is_root:
                node_depth = node.get_depth()
                if node_depth not in self.depth_map:
                    self.depth_map[node_depth] = []
                self.depth_map[node_depth].append(node.key)

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

    def test_bfs_with_callback_collection(self, key_tree):
        """
        Verify BFS traversal with callback function.

        BFS must visit all nodes at depth N before any node at depth N+1.

        Algorithm:
        1. Collect (position, depth) tuples during traversal
        2. Build [depth, first_index, last_index] for each level
        3. Verify: last_index(depth) + 1 == first_index(depth + 1)
        """
        # Collect nodes to preserve full identity
        collected_nodes = []

        def collect_node(node):
            if not node.is_root:
                collected_nodes.append(node)

        list(key_tree.bfs(collect_node))

        assert len(collected_nodes) > 0, "Should collect nodes via BFS callback"

        # Step 1: Build list of (position, depth) tuples
        position_depth_tuples = [
            (i, node.get_depth()) for i, node in enumerate(collected_nodes)
        ]

        print("\n" + "=" * 80)
        print("POSITION-DEPTH TUPLES (first 20):")
        print("=" * 80)
        for pos, depth in position_depth_tuples[:20]:
            key_str = str(collected_nodes[pos].key)[:30]
            print(f"Position {pos:3d}: depth={depth}  key={key_str}")
        if len(position_depth_tuples) > 20:
            print(f"... ({len(position_depth_tuples) - 20} more tuples)")
        print("=" * 80 + "\n")

        # Step 2: Build [depth, first, last] for each level
        level_ranges = []  # Will contain [depth, first, last]

        current_depth = None
        first_index = None
        last_index = None

        for position, depth in position_depth_tuples:
            if current_depth is None:
                # First iteration
                current_depth = depth
                first_index = position
                last_index = position
            elif depth == current_depth:
                # Same level continues
                last_index = position
            else:
                # Level changed: save previous level and start new one
                level_ranges.append([current_depth, first_index, last_index])
                current_depth = depth
                first_index = position
                last_index = position

        # Don't forget the last level
        if current_depth is not None:
            level_ranges.append([current_depth, first_index, last_index])

        # Display level ranges
        print("LEVEL RANGES:")
        print("-" * 80)
        print(
            f"{'Depth':<10} {'First':<10} {'Last':<10} {'Count':<10} {'Sample Keys':<40}"
        )
        print("-" * 80)
        for depth, first, last in level_ranges:
            count = last - first + 1
            # Show sample keys from this level
            sample_keys = [
                str(collected_nodes[i].key)[:15]
                for i in range(first, min(first + 3, last + 1))
            ]
            samples = ", ".join(sample_keys)
            if count > 3:
                samples += ", ..."
            print(f"{depth:<10} {first:<10} {last:<10} {count:<10} {samples:<40}")
        print("-" * 80 + "\n")

        # Step 3: Verify BFS properties
        print("BFS PROPERTY VERIFICATION:")
        print("-" * 80)

        # Rule 1: Depths should be sequential (no gaps)
        depths_in_order = [depth for depth, _, _ in level_ranges]
        for i in range(len(depths_in_order) - 1):
            expected_next_depth = depths_in_order[i] + 1
            actual_next_depth = depths_in_order[i + 1]
            assert actual_next_depth == expected_next_depth, (
                f"Depth gap detected: depth {depths_in_order[i]} followed by {actual_next_depth} "
                f"(expected {expected_next_depth})"
            )
        print("✓ Depths are sequential (no gaps)")

        # Rule 2: No overlap between levels (last + 1 == next first)
        for i in range(len(level_ranges) - 1):
            current_depth, _, current_last = level_ranges[i]
            next_depth, next_first, _ = level_ranges[i + 1]

            expected_next_first = current_last + 1

            print(
                f"  Depth {current_depth} ends at {current_last}, "
                f"Depth {next_depth} starts at {next_first} "
                f"(expected {expected_next_first})",
                end="",
            )

            assert next_first == expected_next_first, (
                f"\nBFS level continuity violated: "
                f"depth {current_depth} ends at index {current_last}, "
                f"but depth {next_depth} starts at index {next_first} "
                f"(expected {expected_next_first}). "
                f"Gap of {next_first - expected_next_first} positions!"
            )

            print(" ✓")

        print("\n✓ All BFS properties verified: perfect level-by-level traversal")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------------
    # Comparative tests between algorithms
    # ------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "min_count", [10, 50, 100, 113]
    )  # with key_tree actual count is 113
    def test_all_traversals_visit_same_nodes(self, key_tree, min_count):
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
        assert (
            len(bfs_nodes) >= min_count
        ), f"Tree should have at least {min_count} nodes, got {len(bfs_nodes)}"

    @pytest.mark.parametrize(
        "min_count", [10, 50, 100, 113]
    )  # with key_tree actual count is 113
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
