# License: BSD 3-Clause
from __future__ import annotations

import numpy as np
import pytest
import openml
from openml.testing import TestBase

class OpenMLRegressionTaskSplitTest(TestBase):
    __test__ = True

    def setUp(self):
        super().setUp()
        self.use_production_server()

    @pytest.mark.production()
    def test_10_fold_cv_splits_integrity(self):
        # task 2280; regression; 10-fold cv
        task_id = 2280
        task = openml.tasks.get_task(task_id)
        
        self.assertEqual(task.task_type_id, openml.tasks.TaskType.SUPERVISED_REGRESSION)
        
        repeats, folds, _ = task.get_split_dimensions()
        self.assertEqual(folds, 10, "Task 2280 should have 10 folds")
        self.assertEqual(repeats, 1, "Task 2280 should have 1 repeat")
        
        # track all test indices to ensure full coverage
        all_test_indices = set()
        
        X, _ = task.get_X_and_y()
        n_instances = X.shape[0]
        
        for fold in range(folds):
            train_indices, test_indices = task.get_train_test_split_indices(fold=fold)
            
            self.assertIsInstance(train_indices, np.ndarray)
            self.assertIsInstance(test_indices, np.ndarray)
            
            intersection = np.intersect1d(train_indices, test_indices)
            self.assertEqual(len(intersection), 0, f"Fold {fold}: Train and test indices overlap")
            
            self.assertTrue(np.all(train_indices < n_instances), f"Fold {fold}: Train indices out of bounds")
            self.assertTrue(np.all(test_indices < n_instances), f"Fold {fold}: Test indices out of bounds")
            self.assertTrue(np.all(train_indices >= 0), f"Fold {fold}: Train indices negative")
            self.assertTrue(np.all(test_indices >= 0), f"Fold {fold}: Test indices negative")
            
            all_test_indices.update(test_indices)
            
        # assert that the union of all test sets covers the entire dataset
        # specific to cross validation (not holdout)
        self.assertEqual(len(all_test_indices), n_instances, "Union of all test sets should cover the entire dataset")
        expected_indices = set(range(n_instances))
        self.assertEqual(all_test_indices, expected_indices, "Test indices should match all instance indices")
