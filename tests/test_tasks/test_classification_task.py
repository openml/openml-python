import numpy as np

from tests.test_tasks import OpenMLSupervisedTaskTest


class OpenMLClassificationTaskTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        super(OpenMLClassificationTaskTest, self).setUp()
        self.task_id = 11

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLClassificationTaskTest, self).test_get_X_and_Y()
        self.assertEqual((3196, 36), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((3196, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)

    def test_download_task(self):

        task = super(OpenMLClassificationTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, 1)
        self.assertEqual(task.dataset_id, 2)
