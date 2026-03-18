# License: BSD 3-Clause
from __future__ import annotations

from time import time
from unittest.mock import MagicMock, patch

import pytest

import openml
from openml.testing import TestBase


# Common methods between tasks
class OpenMLTaskMethodsTest(TestBase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @patch("openml.tasks.list_tasks")
    @patch("openml.tasks.get_task")
    def test_tagging(self, mock_get_task, mock_list_tasks):
        task_id = 1
        mock_task = MagicMock()
        mock_task.tid = task_id
        mock_get_task.return_value = mock_task

        # Initial state: no tasks with the tag
        mock_list_tasks.return_value = {"tid": []}

        task = openml.tasks.get_task(task_id)
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_OpenMLTaskMethodsTest_{unique_indicator}"

        tasks = openml.tasks.list_tasks(tag=tag)
        assert len(tasks["tid"]) == 0

        # After push_tag
        task.push_tag(tag)
        mock_list_tasks.return_value = {"tid": [task_id]}

        tasks = openml.tasks.list_tasks(tag=tag)
        assert len(tasks["tid"]) == 1
        assert task_id in tasks["tid"]

        # After remove_tag
        task.remove_tag(tag)
        mock_list_tasks.return_value = {"tid": []}

        tasks = openml.tasks.list_tasks(tag=tag)
        assert len(tasks["tid"]) == 0

        # Verify interactions
        mock_get_task.assert_called_with(task_id)
        mock_task.push_tag.assert_called_with(tag)
        mock_task.remove_tag.assert_called_with(tag)

    @patch("openml.tasks.get_task")
    def test_get_train_and_test_split_indices(self, mock_get_task):
        task_id = 1882
        mock_task = MagicMock()
        mock_task.tid = task_id
        # Define expected indices for the mock
        expected_train_00 = [16, 395]
        expected_test_00 = [412, 364]
        expected_train_22 = [237, 681]
        expected_test_22 = [583, 24]

        def side_effect_indices(fold, repeat, sample=0):
            if repeat == 0 and fold == 0:
                return (expected_train_00, expected_test_00)
            if repeat == 2 and fold == 2:
                return (expected_train_22, expected_test_22)
            if repeat != 0 and repeat != 2:
                raise ValueError(f"Repeat {repeat} not known")
            if fold != 0 and fold != 2:
                raise ValueError(f"Fold {fold} not known")
            raise ValueError(f"Split not found for fold={fold}, repeat={repeat}")

        mock_task.get_train_test_split_indices.side_effect = side_effect_indices
        mock_get_task.return_value = mock_task

        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(task_id)

        train_indices, test_indices = task.get_train_test_split_indices(0, 0)
        assert train_indices[0] == expected_train_00[0]
        assert train_indices[-1] == expected_train_00[-1]
        assert test_indices[0] == expected_test_00[0]
        assert test_indices[-1] == expected_test_00[-1]

        train_indices, test_indices = task.get_train_test_split_indices(2, 2)
        assert train_indices[0] == expected_train_22[0]
        assert train_indices[-1] == expected_train_22[-1]
        assert test_indices[0] == expected_test_22[0]
        assert test_indices[-1] == expected_test_22[-1]

        self.assertRaisesRegex(
            ValueError,
            "Fold 10 not known",
            task.get_train_test_split_indices,
            10,
            0,
        )
        self.assertRaisesRegex(
            ValueError,
            "Repeat 10 not known",
            task.get_train_test_split_indices,
            0,
            10,
        )
