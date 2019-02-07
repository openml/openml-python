import numpy as np

from tests.test_tasks.test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLLearningCurveTest(OpenMLSupervisedTaskTest):

    def setup(self):

        self.task_id = 67

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLLearningCurveTest, self).test_get_X_and_Y()
        self.assertEqual((345 , 7), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((345 , ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)
