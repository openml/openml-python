# License: BSD 3-Clause
from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
import pytest

from openml.study import OpenMLBenchmarkSuite
from openml.testing import TestBase


class TestBenchmarkSuiteMetadata(TestBase):
    """Test suite for OpenMLBenchmarkSuite.metadata property."""

    def setUp(self):
        """Create a test suite instance."""
        super().setUp()
        self.suite = OpenMLBenchmarkSuite(
            suite_id=99,
            alias="test-suite",
            name="Test Suite",
            description="A test suite",
            status="active",
            creation_date="2022-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=[1, 2, 3],
        )

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_basic_structure(self, mock_list_tasks, mock_list_datasets):
        """Test that metadata returns a DataFrame with expected structure."""
        # Mock task response (with tid as index)
        task_data = {
            1: {"tid": 1, "did": 10, "name": "Task1", "NumberOfInstances": 100},
            2: {"tid": 2, "did": 11, "name": "Task2", "NumberOfInstances": 200},
            3: {"tid": 3, "did": 10, "name": "Task3", "NumberOfInstances": 150},
        }
        task_df = pd.DataFrame.from_dict(task_data, orient="index")
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df

        # Mock dataset response
        dataset_df = pd.DataFrame(
            {
                "did": [10, 11],
                "version": [1, 1],
                "uploader": [5, 5],
                "name": ["Dataset1", "Dataset2"],
            }
        )
        mock_list_datasets.return_value = dataset_df

        # Access property
        metadata = self.suite.metadata

        # Assertions
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) == 3  # One row per task
        assert "tid" in metadata.columns
        assert "did" in metadata.columns
        assert "version" in metadata.columns
        assert "NumberOfInstances" in metadata.columns

        # Verify API calls
        mock_list_tasks.assert_called_once()
        mock_list_datasets.assert_called_once()

    @patch("openml.study.study._list_tasks")
    def test_metadata_caching(self, mock_list_tasks):
        """Test that metadata is cached after first access."""
        task_df = pd.DataFrame(
            {
                "tid": [1],
                "did": [10],
                "name": ["Task1"],
            }
        )
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df

        # First access
        meta1 = self.suite.metadata
        # Second access
        meta2 = self.suite.metadata

        # Should be same object (cached)
        assert meta1 is meta2
        # Should only call API once
        assert mock_list_tasks.call_count == 1

    def test_metadata_empty_suite(self):
        """Test metadata for suite with no tasks."""
        empty_suite = OpenMLBenchmarkSuite(
            suite_id=1,
            alias=None,
            name="Empty Suite",
            description="",
            status="active",
            creation_date="2022-01-01",
            creator=1,
            tags=None,
            data=None,
            tasks=[],  # Empty tasks
        )

        metadata = empty_suite.metadata
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) == 0

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_merge_behavior(self, mock_list_tasks, mock_list_datasets):
        """Test that merge preserves task structure (left join)."""
        # Task with dataset that doesn't exist in dataset_df
        task_df = pd.DataFrame(
            {
                "tid": [1, 2],
                "did": [10, 99],  # did=99 doesn't exist in dataset_df
                "name": ["Task1", "Task2"],
            }
        )
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df

        dataset_df = pd.DataFrame({"did": [10], "version": [1]})
        mock_list_datasets.return_value = dataset_df

        metadata = self.suite.metadata

        # Should have 2 rows (one per task)
        assert len(metadata) == 2
        # Task 1 should have version
        assert metadata.loc[metadata["tid"] == 1, "version"].iloc[0] == 1
        # Task 2 should have NaN for version (missing dataset)
        assert pd.isna(metadata.loc[metadata["tid"] == 2, "version"].iloc[0])

    @patch("openml.study.study._list_tasks")
    def test_metadata_error_handling(self, mock_list_tasks):
        """Test error handling when API calls fail."""
        from openml.exceptions import OpenMLServerException

        mock_list_tasks.side_effect = OpenMLServerException("Server error", code=500)

        with pytest.raises(RuntimeError, match="Failed to retrieve task metadata"):
            _ = self.suite.metadata

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_index_reset(self, mock_list_tasks, mock_list_datasets):
        """Test that index is properly reset when tid is index."""
        # Create DataFrame with tid as index
        task_df = pd.DataFrame(
            {
                "did": [10, 11],
                "name": ["Task1", "Task2"],
                "NumberOfInstances": [100, 200],
            },
            index=[1, 2],
        )
        task_df.index.name = "tid"
        mock_list_tasks.return_value = task_df

        dataset_df = pd.DataFrame({"did": [10, 11], "version": [1, 1]})
        mock_list_datasets.return_value = dataset_df

        metadata = self.suite.metadata

        # After reset_index, tid should be a column, not the index
        assert "tid" in metadata.columns
        assert metadata.index.name is None or isinstance(metadata.index, pd.RangeIndex)

    @patch("openml.study.study.list_datasets")
    @patch("openml.study.study._list_tasks")
    def test_metadata_no_did_column(self, mock_list_tasks, mock_list_datasets):
        """Test fallback when did column is missing."""
        # Task DataFrame without 'did' column (unlikely but test it)
        task_df = pd.DataFrame(
            {
                "tid": [1, 2],
                "name": ["Task1", "Task2"],
            }
        )
        mock_list_tasks.return_value = task_df

        metadata = self.suite.metadata

        # Should return task_df without merging
        assert len(metadata) == 2
        assert "did" not in metadata.columns
        # Should not call list_datasets
        mock_list_datasets.assert_not_called()

