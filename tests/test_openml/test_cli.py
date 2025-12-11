# License: BSD 3-Clause
from __future__ import annotations

import argparse
from unittest import mock

import pandas as pd
import pytest

from openml import cli


class TestTasksCLI:
    """Test suite for tasks CLI commands."""

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_list_basic(self, mock_list):
        """Test basic task listing."""
        mock_df = pd.DataFrame({
            "tid": [1, 2, 3],
            "task_type": ["Supervised Classification", "Supervised Regression", "Clustering"],
            "did": [61, 62, 63],
        })
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            offset=None,
            size=10,
            tag=None,
            task_type=None,
            status=None,
            data_name=None,
            format="table",
            verbose=False,
        )

        cli.tasks_list(args)
        mock_list.assert_called_once_with(offset=None, size=10)

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_list_with_filters(self, mock_list):
        """Test task listing with filters."""
        mock_df = pd.DataFrame({
            "tid": [1],
            "task_type": ["Supervised Classification"],
        })
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            offset=0,
            size=5,
            tag="study_14",
            task_type="Supervised Classification",
            status="active",
            data_name=None,
            format="table",
            verbose=False,
        )

        cli.tasks_list(args)
        mock_list.assert_called_once_with(
            offset=0,
            size=5,
            tag="study_14",
            task_type="Supervised Classification",
            status="active",
        )

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_list_empty_results(self, mock_list):
        """Test handling of empty results."""
        mock_list.return_value = pd.DataFrame()

        args = argparse.Namespace(
            offset=None,
            size=None,
            tag=None,
            task_type=None,
            status=None,
            data_name=None,
            format="table",
            verbose=False,
        )

        cli.tasks_list(args)
        mock_list.assert_called_once()

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_list_json_format(self, mock_list):
        """Test JSON output format."""
        mock_df = pd.DataFrame({"tid": [1], "task_type": ["Supervised Classification"]})
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            offset=None,
            size=10,
            tag=None,
            task_type=None,
            status=None,
            data_name=None,
            format="json",
            verbose=False,
        )

        cli.tasks_list(args)
        mock_list.assert_called_once()

    @mock.patch("openml.tasks.get_task")
    def test_tasks_info(self, mock_get):
        """Test task info display."""
        mock_task = mock.Mock()
        mock_task.task_id = 1
        mock_task.task_type = "Supervised Classification"
        mock_task.dataset_id = 61
        mock_task.estimation_procedure = {"type": "crossvalidation"}
        mock_task.evaluation_measure = "predictive_accuracy"
        mock_task.target_name = "class"
        mock_task.class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        mock_get.return_value = mock_task

        args = argparse.Namespace(task_id="1")

        cli.tasks_info(args)
        mock_get.assert_called_once_with(1)

    @mock.patch("openml.tasks.get_task")
    def test_tasks_info_error(self, mock_get):
        """Test task info with invalid ID."""
        mock_get.side_effect = Exception("Task not found")

        args = argparse.Namespace(task_id="99999")

        with pytest.raises(SystemExit):
            cli.tasks_info(args)

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_search_found(self, mock_list):
        """Test search with results."""
        mock_df = pd.DataFrame({
            "tid": [1],
            "data_name": ["iris"],
        })
        mock_list.return_value = mock_df

        args = argparse.Namespace(
            query="iris",
            size=20,
            format="table",
            verbose=False,
        )

        cli.tasks_search(args)
        mock_list.assert_called()

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_search_not_found(self, mock_list):
        """Test search with no results."""
        mock_list.return_value = pd.DataFrame()

        args = argparse.Namespace(
            query="nonexistent",
            size=20,
            format="table",
            verbose=False,
        )

        cli.tasks_search(args)
        assert mock_list.call_count >= 1

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_search_case_insensitive(self, mock_list):
        """Test case-insensitive search."""
        # First call returns empty (no exact match)
        # Second call returns all tasks for client-side filtering
        mock_list.side_effect = [
            pd.DataFrame(),  # No exact match
            pd.DataFrame({
                "tid": [1, 2],
                "data_name": ["Iris", "IRIS-versicolor"],
            }),
        ]

        args = argparse.Namespace(
            query="iris",
            size=20,
            format="table",
            verbose=False,
        )

        cli.tasks_search(args)
        assert mock_list.call_count == 2

    def test_tasks_handler_no_action(self):
        """Test tasks handler with no action specified."""
        args = argparse.Namespace(tasks_action=None)

        with pytest.raises(SystemExit):
            cli.tasks_handler(args)

    @mock.patch("openml.tasks.list_tasks")
    def test_tasks_handler_list_action(self, mock_list):
        """Test tasks handler routes list action correctly."""
        mock_list.return_value = pd.DataFrame({"tid": [1], "task_type": ["test"]})

        args = argparse.Namespace(
            tasks_action="list",
            offset=None,
            size=5,
            tag=None,
            task_type=None,
            status=None,
            data_name=None,
            format="table",
            verbose=False,
        )

        cli.tasks_handler(args)
        mock_list.assert_called_once()
