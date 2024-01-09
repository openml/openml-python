# License: BSD 3-Clause
from __future__ import annotations

from time import time

import openml
from openml.testing import TestBase


# Common methods between tasks
class OpenMLTaskMethodsTest(TestBase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_tagging(self):
        task = openml.tasks.get_task(1)  # anneal; crossvalidation
        # tags can be at most 64 alphanumeric (+ underscore) chars
        unique_indicator = str(time()).replace(".", "")
        tag = f"test_tag_OpenMLTaskMethodsTest_{unique_indicator}"
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        assert len(tasks) == 0
        task.push_tag(tag)
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        assert len(tasks) == 1
        assert 1 in tasks["tid"]
        task.remove_tag(tag)
        tasks = openml.tasks.list_tasks(tag=tag, output_format="dataframe")
        assert len(tasks) == 0

    def test_get_train_and_test_split_indices(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882)
        train_indices, test_indices = task.get_train_test_split_indices(0, 0)
        assert train_indices[0] == 16
        assert train_indices[-1] == 395
        assert test_indices[0] == 412
        assert test_indices[-1] == 364
        train_indices, test_indices = task.get_train_test_split_indices(2, 2)
        assert train_indices[0] == 237
        assert train_indices[-1] == 681
        assert test_indices[0] == 583
        assert test_indices[-1] == 24
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
