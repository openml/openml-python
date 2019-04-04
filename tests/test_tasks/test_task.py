import unittest

import openml
from openml.testing import TestBase
from openml.datasets import OpenMLDataset
from openml.tasks import OpenMLTask
from openml.exceptions import OpenMLServerException


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

        task = openml.tasks.get_task(self.task_id)
        return task

    def test_upload_task(self):

        task = openml.tasks.get_task(self.task_id)
        task_dataset = openml.datasets.get_dataset(task.dataset_id)
        task.dataset_id = self._upload_dataset(task_dataset)
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
        dataset.status = 'active'
        try:
            return dataset.publish()
        except openml.exceptions.OpenMLServerException:
            # Something went wrong.
            # Test dataset was not
            # published. Return old id.
            return dataset.dataset_id
