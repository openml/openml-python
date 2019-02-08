import numpy as np

from tests.test_tasks import OpenMLSupervisedTaskTest


class OpenMLLearningCurveTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        self.task_id = 801
        super(OpenMLLearningCurveTest, self).setUp()

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLLearningCurveTest, self).test_get_X_and_Y()
        self.assertEqual((768, 8), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((768, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)
