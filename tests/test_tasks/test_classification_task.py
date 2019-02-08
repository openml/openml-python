import numpy as np

from tests.test_tasks import OpenMLSupervisedTaskTest


class OpenMLClassificationTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        super(OpenMLClassificationTest, self).setUp()
        self.task_id = 11

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLClassificationTest, self).test_get_X_and_Y()
        self.assertEqual((3196, 36), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((3196, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)
