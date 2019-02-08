import numpy as np

from tests.test_tasks import OpenMLSupervisedTaskTest


class OpenMLRegressionTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        super(OpenMLRegressionTest, self).setUp()
        self.task_id = 625

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLRegressionTest, self).test_get_X_and_Y()
        self.assertEqual((194, 32), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((194,), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, float)
