import unittest

import openml
from openml.testing import TestBase
from openml.exceptions import OpenMLServerException


class OpenMLTaskTest(TestBase):
    # task id will be set from the
    # extending classes

    def setUp(self):

        super(OpenMLTaskTest, self).setUp()
        self.task_id = 11
        self.estimation_procedure = 23

    @classmethod
    def setUpClass(cls):
        if cls is OpenMLTaskTest:
            raise unittest.SkipTest(
                "Skip OpenMLTaskTest tests,"
                " it's a base class"
            )
        super(OpenMLTaskTest, cls).setUpClass()

    def test_download_task(self):

        task = openml.tasks.get_task(self.task_id)
        return task

    def test_upload_task(self):

        task = openml.tasks.get_task(self.task_id)
        task.estimation_procedure_id = self.estimation_procedure
        try:
            task.publish()
        except OpenMLServerException as e:
            # 614 is the error code
            # when the task already
            # exists
            if e.code != 614:
                raise e

