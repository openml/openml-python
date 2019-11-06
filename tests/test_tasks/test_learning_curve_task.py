# License: BSD 3-Clause

import numpy as np

from openml.tasks import get_task
from .test_supervised_task import OpenMLSupervisedTaskTest


class OpenMLLearningCurveTaskTest(OpenMLSupervisedTaskTest):

    __test__ = True

    def setUp(self, n_levels: int = 1):

        super(OpenMLLearningCurveTaskTest, self).setUp()
        self.task_id = 801
        self.task_type_id = 3
        self.estimation_procedure = 13

    def test_get_X_and_Y(self):

        X, Y = super(OpenMLLearningCurveTaskTest, self).test_get_X_and_Y()
        self.assertEqual((768, 8), X.shape)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual((768, ), Y.shape)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(Y.dtype, int)

    def test_download_task(self):

        task = super(OpenMLLearningCurveTaskTest, self).test_download_task()
        self.assertEqual(task.task_id, self.task_id)
        self.assertEqual(task.task_type_id, 3)
        self.assertEqual(task.dataset_id, 20)

    def test_class_labels(self):

        task = get_task(self.task_id)
        self.assertEqual(
            task.class_labels,
            ['tested_negative', 'tested_positive']
        )
