"""
This test files evaluates the deprecated functionalities of paths with class DictPaths.
It will be removed in version 1.0.0
"""

import re
import warnings

import pytest

from ndict_tools import CompactPathsView, DictPaths, NestedDictionary


class TestDictPathsDeprecation:

    def test_dict_paths_raise_deprecated_warning(self, strict_c_nd):
        with pytest.warns(
            DeprecationWarning,
            match=re.escape(
                "DictPaths is deprecated since version 0.9.0 and will be removed in version 1.0.0. "
                "Use CompactPathsView or NestedDictionary.compact_paths() instead."
            ),
        ) as record:
            d_paths = DictPaths(strict_c_nd)

        assert len(record) == 1

    def test_dict_paths_warning_pattern(self, smooth_c_nd):
        """Test warning message matches expected pattern."""
        pattern = re.compile(
            r"DictPaths is deprecated since version \d+\.\d+\.\d+ "
            r"and will be removed in version \d+\.\d+\.\d+\. "
            r"Use CompactPathsView or NestedDictionary\.compact_paths\(\) instead\."
        )

        with pytest.warns(DeprecationWarning) as record:
            d_paths = DictPaths(smooth_c_nd)

        message = str(record[0].message)
        assert pattern.match(message), f"Message doesn't match pattern: {message}"

    def test_dict_paths_warning_contains_versions(self, strict_c_nd):
        """Test that warning contains specific version numbers."""
        version_pattern = re.compile(r"\d+\.\d+\.\d+")

        with pytest.warns(DeprecationWarning) as record:
            dpaths = DictPaths(strict_c_nd)

        message = str(record[0].message)
        versions = version_pattern.findall(message)

        # Should find exactly 2 versions
        assert len(versions) == 2
        assert versions[0] == "0.9.0"  # Deprecation version
        assert versions[1] == "1.0.0"  # Removal version

    def test_dict_paths_still_works(self, smooth_c_nd):
        """Test that DictPaths still functions correctly despite being deprecated."""
        with warnings.catch_warnings():
            # Suppress warnings for this test
            warnings.simplefilter("ignore", DeprecationWarning)

            # Create DictPaths instance
            dpaths = DictPaths(smooth_c_nd)

            # Verify it's an instance of the expected classes
            assert isinstance(dpaths, DictPaths)
            assert not isinstance(dpaths, CompactPathsView)

            # Verify it has the expected structure
            structure = dpaths.structure
            assert isinstance(structure, list)
            assert len(structure) > 0

            # Verify expand works
            expanded = dpaths.expand()
            assert isinstance(expanded, list)
            assert len(expanded) > 0

    def test_compact_paths_view_no_warning(self, strict_c_nd):
        """Test that CompactPathsView does not raise a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # If a warning is raised, it will be converted to an error
            c_paths = CompactPathsView(strict_c_nd)
            assert isinstance(c_paths, CompactPathsView)
            assert not isinstance(c_paths, DictPaths)

    def test_compact_paths_method_no_warning(self, smooth_c_nd):
        """Test that NestedDictionary.compact_paths() does not raise a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # If a warning is raised, it will be converted to an error
            c_paths = smooth_c_nd.compact_paths()
            assert isinstance(c_paths, CompactPathsView)
            assert not isinstance(c_paths, DictPaths)


class TestDeprecatedMigrationPaths:

    def test_equivalent_behavior(self, strict_c_nd):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            d_paths = DictPaths(strict_c_nd)
            c_paths = CompactPathsView(strict_c_nd)

            assert len(d_paths) == len(c_paths)
            assert d_paths.structure == c_paths.structure
            assert d_paths.expand() == c_paths.expand()
            assert list(d_paths) == list(c_paths)
