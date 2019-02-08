import numpy as np

from tests.test_tasks import OpenMLSupervisedTaskTest


class OpenMLRegressionTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        self.task_id = 738
        super(OpenMLRegressionTest, self).setUp()

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLRegressionTest, self).test_get_X_and_Y()
        self.assertEqual((2178, 3), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((2178,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)
