import numpy as np

from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLClassificationTaskTest(OpenMLSupervisedTaskTest):

    def setUp(self):

        super(OpenMLClassificationTaskTest, self).setUp()
        self.task_id = 119
        self.task = super(OpenMLClassificationTaskTest, self)\
            .test_download_task()

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLClassificationTaskTest, self).test_get_X_and_Y()
        self.assertEqual((768, 8), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((768, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)

    def test_download_task(self):

        self.assertEqual(self.task.task_id, self.task_id)
        self.assertEqual(self.task.task_type_id, 1)
        self.assertEqual(self.task.dataset_id, 20)

    def test_class_labels(self):

        self.assertEqual(
            self.task.class_labels,
            ['tested_negative', 'tested_positive']
        )
