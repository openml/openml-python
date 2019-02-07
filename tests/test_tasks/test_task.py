import unittest

import openml
from openml.testing import TestBase
from openml.exceptions import OpenMLServerException


@unittest.skip("Base class does not need to be tested")
class OpenMLTaskTest(TestBase):
    # task id will be set from the
    # extending classes
    def test_download_task(self):

        openml.tasks.get_task(self.task_id)

    def test_upload_task(self):

        task = openml.tasks.get_task(self.task_id)
        task.estimation_procedure_id = 23
        try:
            task.publish()
        except OpenMLServerException as e:
            # 614 is the error code
            # when the task already
            # exists
            if e.code != 614:
                raise e

