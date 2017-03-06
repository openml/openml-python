import openml, math, collections
import numpy as np
from openml.testing import TestBase
from openml.utils.preprocessing import ConditionalImputer

class OpenMLTaskTest(TestBase):

    def _do_test(self, dataset, X, nominal_indices, clf):
        clf.fit(X)
        X_prime = clf.transform(X)

        # in case of smart guessing nominal attributes, we accept false positives, but no false negatives
        for column_idx in nominal_indices:
            if clf.categorical_features_implied is not None:
                assert column_idx in clf.categorical_features_implied, "False negative with smart nominal detector"

        correction = 0
        for idx, value in enumerate(clf.statistics_):
            if math.isnan(value):
                # imputer can only give nan if all values are unknown
                correction += 1
                assert dataset.features[idx].number_missing_values == len(X), "Imputer calculated nan for usable feature"
            else:
                # check if nominal values get imputed correct
                if idx in nominal_indices:
                    assert value == math.floor(value) == math.ceil(value), "Wrong impute value for nominal feature"

                corrected_index = idx - correction  # for x prime
                # check if imputation succeeded
                counter = collections.Counter(X_prime[:, corrected_index])
                occurances_after = counter[value]
                assert occurances_after >= dataset.features[idx].number_missing_values

        return X_prime

    def test_impute_indices(self):
        task_ids = [2, 24, 41, 42, 59]

        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            X, _ = dataset.get_data(target=task.target_name)
            nominal_indices = dataset.get_features_by_type('nominal', exclude=[task.target_name])
            # print("task id %d indices %s" %(task_id, str(nominal_indices)))
            clf = ConditionalImputer(strategy="median",
                                     strategy_nominal="most_frequent",
                                     categorical_features=nominal_indices,
                                     verbose=True)

            self._do_test(dataset, X, nominal_indices, clf)


    def test_impute_smart(self):
        task_ids = [2, 24, 41, 42, 59]

        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            X, _ = dataset.get_data(target=task.target_name)
            nominal_indices = dataset.get_features_by_type('nominal', exclude=[task.target_name])
            # print("task id %d indices %s" %(task_id, str(nominal_indices)))
            clf = ConditionalImputer(strategy="median",
                                     strategy_nominal="most_frequent",
                                     categorical_features=None,
                                     verbose=True)

            self._do_test(dataset, X, nominal_indices, clf)

    def test_impute_with_constant(self):
        task_ids = [2]

        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            X, _ = dataset.get_data(target=task.target_name)
            nominal_indices = dataset.get_features_by_type('nominal', exclude=[task.target_name])
            clf = ConditionalImputer(strategy="median",
                                     strategy_nominal="most_frequent",
                                     categorical_features=None,
                                     verbose=True,
                                     empty_attribute_constant=-1)

            X_prime = self._do_test(dataset, X, nominal_indices, clf)
            assert np.isnan(np.min(X_prime)) == False, 'Result contains nans'
            assert X_prime.shape == X.shape