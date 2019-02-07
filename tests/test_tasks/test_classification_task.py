import numpy as np

from tests.test_tasks.test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLClassificationTest(OpenMLSupervisedTaskTest):

    def setup(self):

        self.task_id = 11

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLClassificationTest, self).test_get_X_and_Y()
        self.assertEqual((898, 38), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((898, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)
