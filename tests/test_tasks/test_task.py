import unittest

import openml
from openml.testing import TestBase
from openml.datasets import OpenMLDataset
from openml.tasks import OpenMLTask
from openml.exceptions import OpenMLServerException
from .test_supervised_task import OpenMLSupervisedTaskTest

# Helper class
# The test methods in this class
# are not supposed to be executed.
class OpenMLTaskTest(TestBase):
    # task id, dataset_id,
    # estimation_procedure
    # will be set from the
    # extending classes

    def setUp(self):

        super(OpenMLTaskTest, self).setUp()
        self.task_id = 11
        self.estimation_procedure = 23

    @classmethod
    def setUpClass(cls):

        if cls is OpenMLTaskTest or OpenMLSupervisedTaskTest:
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
        # adding sentinel so we can have a new dataset
        # hence a "new task" to upload
        task_dataset = openml.datasets.get_dataset(task.dataset_id)
        task.dataset_id = self._upload_dataset(task_dataset)
        task.estimation_procedure_id = self.estimation_procedure
        try:
            task.publish()
        except OpenMLServerException as e:
            # 614 is the error code
            # when the task already
            # exists
            if e.code != 614:
                raise e

    def _upload_dataset(self, dataset: OpenMLDataset) -> int:

        dataset.name = '%s%s' % (self._get_sentinel(), dataset.name)
        try:
            return dataset.publish()
        except openml.exceptions.OpenMLServerException:
            # something went wrong
            # test dataset was not
            # published. Return old id.
            return dataset.dataset_id
