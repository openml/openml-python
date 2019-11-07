# License: BSD 3-Clause

import numpy as np

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTaskTest(OpenMLSupervisedTaskTest):

    __test__ = True

    def setUp(self, n_levels: int = 1):

        super(OpenMLRegressionTaskTest, self).setUp()
        self.task_id = 625
        self.task_type_id = 2
        self.estimation_procedure = 7

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLRegressionTaskTest, self).test_get_X_and_Y()
        self.assertEqual((194, 32), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((194,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)

    def test_download_task(self):

        task = super(OpenMLRegressionTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, 2)
        self.assertEqual(task.dataset_id, 105)
