import numpy as np

from tests.test_tasks.test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLRegressionTest(OpenMLSupervisedTaskTest):

    def setup(self):
        self.task_id = 631

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLRegressionTest, self).test_get_X_and_Y()
        self.assertEqual((52, 2), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((52,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)
