import openml, math, collections
from openml.testing import TestBase
from openml.utils.preprocessing import ConditionalImputer

class OpenMLTaskTest(TestBase):

    def test_impute_anneal(self):
        task_id = 2

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, _ = dataset.get_data(target=task.target_name)
        nominal_indeces = dataset.get_features_by_type('nominal', exclude=[38])
        clf = ConditionalImputer(strategy="median", strategy_nominal="most_frequent", indeces_nominal=nominal_indeces)
        clf.fit(X)
        X_prime = clf.transform(X)

        correction = 0
        for idx, value in enumerate(clf.statistics_):
            if math.isnan(value):
                # imputer can only give nan if all values are unknown
                correction += 1
                assert dataset.features[idx].number_missing_values == len(
                    X), "Imputer calculated nan for usable feature"
            else:
                # check if nominal values get imputed correct
                if idx in nominal_indeces:
                    assert value == math.floor(value) == math.ceil(value), "Wrong impute value for nominal feature"

                corrected_index = idx - correction  # for x prime
                # check if imputation succeeded
                counter = collections.Counter(X_prime[:, corrected_index])
                occurances_after = counter[value]
                assert occurances_after >= dataset.features[idx].number_missing_values