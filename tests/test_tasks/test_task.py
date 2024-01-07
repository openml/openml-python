# License: BSD 3-Clause
from __future__ import annotations

import unittest
from random import randint, shuffle

from openml.datasets import (
    get_dataset,
    list_datasets,
)
from openml.exceptions import OpenMLServerException
from openml.tasks import TaskType, create_task, get_task
from openml.testing import TestBase


class OpenMLTaskTest(TestBase):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLTaskTest:
            raise unittest.SkipTest("Skip OpenMLTaskTest tests," " it's a base class")
        super().setUpClass()

    def setUp(self, n_levels: int = 1):
        super().setUp()

    def test_download_task(self):
        return get_task(self.task_id)

    def test_upload_task(self):
        # We don't know if the task in question already exists, so we try a few times. Checking
        # beforehand would not be an option because a concurrent unit test could potentially
        # create the same task and make this unit test fail (i.e. getting a dataset and creating
        # a task for it is not atomic).
        compatible_datasets = self._get_compatible_rand_dataset()
        for i in range(100):
            try:
                dataset_id = compatible_datasets[i % len(compatible_datasets)]
                # TODO consider implementing on the diff task types.
                task = create_task(
                    task_type=self.task_type,
                    dataset_id=dataset_id,
                    target_name=self._get_random_feature(dataset_id),
                    estimation_procedure_id=self.estimation_procedure,
                )

                task.publish()
                TestBase._mark_entity_for_removal("task", task.id)
                TestBase.logger.info(
                    "collected from {}: {}".format(__file__.split("/")[-1], task.id),
                )
                # success
                break
            except OpenMLServerException as e:
                # Error code for 'task already exists'
                # Should be 533 according to the docs
                # (# https://www.openml.org/api_docs#!/task/post_task)
                if e.code == 614:
                    continue
                else:
                    raise e
        else:
            raise ValueError(
                f"Could not create a valid task for task type ID {self.task_type}",
            )

    def _get_compatible_rand_dataset(self) -> list:
        active_datasets = list_datasets(status="active", output_format="dataframe")

        # depending on the task type, find either datasets
        # with only symbolic features or datasets with only
        # numerical features.
        if self.task_type == TaskType.SUPERVISED_REGRESSION:
            compatible_datasets = active_datasets[active_datasets["NumberOfSymbolicFeatures"] == 0]
        elif self.task_type == TaskType.CLUSTERING:
            compatible_datasets = active_datasets
        else:
            compatible_datasets = active_datasets[active_datasets["NumberOfNumericFeatures"] == 0]

        compatible_datasets = list(compatible_datasets["did"])
        # in-place shuffling
        shuffle(compatible_datasets)
        return compatible_datasets

        # random_dataset_pos = randint(0, len(compatible_datasets) - 1)
        #
        # return compatible_datasets[random_dataset_pos]

    def _get_random_feature(self, dataset_id: int) -> str:
        random_dataset = get_dataset(dataset_id)
        # necessary loop to overcome string and date type
        # features.
        while True:
            random_feature_index = randint(0, len(random_dataset.features) - 1)
            random_feature = random_dataset.features[random_feature_index]
            if self.task_type == TaskType.SUPERVISED_REGRESSION:
                if random_feature.data_type == "numeric":
                    break
            else:
                if random_feature.data_type == "nominal":
                    break
        return random_feature.name
