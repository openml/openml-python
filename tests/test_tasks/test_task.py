import unittest
from time import time

from openml.testing import TestBase
from openml.datasets import (
    get_dataset,
    OpenMLDataset,
    check_datasets_active,
)
from openml.tasks import (
    get_task,
    OpenMLTask,
)


class OpenMLTaskTest(TestBase):
    """
    A helper class. The methods of the test case
    are only executed in subclasses of the test case.
    """
    def setUp(self):
        super(OpenMLTaskTest, self).setUp()
        # task_id and estimation_procedure
        # act as placeholder variables.
        # They are set from the extending classes.
        self.task_id = 11
        self.estimation_procedure = 23

    @classmethod
    def setUpClass(cls):
        # placed here to avoid a circular import
        from .test_supervised_task import OpenMLSupervisedTaskTest
        if cls is OpenMLTaskTest or cls is OpenMLSupervisedTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLTaskTest tests,"
                " it's a base class"
            )
        super(OpenMLTaskTest, cls).setUpClass()

    def test_download_task(self) -> OpenMLTask:

        task = get_task(self.task_id)
        return task

    def test_upload_task(self):

        task = get_task(self.task_id)
        dataset = get_dataset(task.dataset_id)
        new_dataset_id = self._upload_dataset(dataset)
        OpenMLTaskTest._wait_dataset_activation(new_dataset_id, 240)
        task.dataset_id = new_dataset_id
        task.estimation_procedure_id = self.estimation_procedure
        task.publish()

    def _upload_dataset(self, dataset: OpenMLDataset) -> int:
        """Reupload the dataset.

        Add a sentinel to the dataset name to achieve a
        successful upload every time without creating a
        new dataset.

        Parameters
        ----------
        dataset: OpenMLDataset
            The dataset from OpenML that will be
            reuploaded.

        Returns
        -------
        int
            Dataset id. If the reupload is successful,
            the new id. Otherwise, the old id of the
            dataset.
        """
        dataset.name = '%s%s' % (self._get_sentinel(), dataset.name)
        # Providing both dataset file and url
        # raises an error when uploading.
        dataset.url = None
        return dataset.publish()

    @staticmethod
    def _wait_dataset_activation(
            dataset_id: int,
            max_wait_time: int
    ):
        """Wait until the dataset status is changed
        to activated, given a max wait time.

        Parameters
        ----------
        dataset_id: int
            The id of the dataset whose status
            activation will be observed.
        max_wait_time: int
            Maximal amount of time to wait in
            seconds.
        """
        start_time = time()
        # Check while the status of the dataset is not activated
        while not check_datasets_active([dataset_id]).get(dataset_id):
            # break if the time so far exceeds max wait time
            if time() - start_time > max_wait_time:
                break
