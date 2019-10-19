import openml
from openml.testing import TestBase

import unittest
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


class TestPerroneExample(TestBase):

    @unittest.skipIf(LooseVersion(sklearn.__version__) < "0.20", reason="SimpleImputer and "
                                                                        "ColumnTransformer doesn't"
                                                                        "exist in older versions.")
    def test_perrone_example(self):
        openml.config.server = self.production_server

        metric = 'area_under_roc_curve'
        task_ids = [9983, 3485, 3902, 3903, 145878]
        flow_id = 5891
        colnames = ['cost', 'degree', 'gamma', 'kernel']
        eval_df = openml.evaluations.list_evaluations_setups(function=metric,
                                                             task=task_ids,
                                                             flow=[flow_id],
                                                             uploader=[2702],
                                                             output_format='dataframe',
                                                             parameters_in_separate_columns=True)
        self.assertTrue(all(np.isin(task_ids, eval_df['task_id'])))
        self.assertEqual(eval_df.shape[1], 21)
        self.assertGreaterEqual(eval_df.shape[0], 5000)

        eval_df.columns = [column.split('_')[-1] for column in eval_df.columns]
        self.assertTrue(all(np.isin(colnames, eval_df.columns)))

        eval_df = eval_df.sample(frac=1)  # shuffling rows
        eval_df.columns = [column.split('_')[-1] for column in eval_df.columns]
        X = eval_df.loc[:, colnames]
        y = eval_df.loc[:, 'value']
        cat_cols = ['kernel']
        num_cols = ['cost', 'degree', 'gamma']
        # Missing value imputers
        from sklearn.impute import SimpleImputer
        cat_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')
        num_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
        # Creating the one-hot encoder
        enc = OneHotEncoder(handle_unknown='ignore')
        # Pipeline to handle categorical column transformations
        cat_transforms = Pipeline(steps=[('impute', cat_imputer), ('encode', enc)])
        # Combining column transformers
        from sklearn.compose import ColumnTransformer
        ct = ColumnTransformer([('cat', cat_transforms, cat_cols), ('num', num_imputer, num_cols)])
        # Creating the full pipeline with the surrogate model
        clf = RandomForestRegressor(n_estimators=50)
        model = Pipeline(steps=[('preprocess', ct), ('surrogate', clf)])
        model.fit(X, y)
        y_pred = model.predict(X)
        mean_squared_error(y, y_pred)

        def random_sample_configurations(num_samples=100):
            colnames = ['cost', 'degree', 'gamma', 'kernel']
            ranges = [(0.000986, 998.492437),
                      (2.0, 5.0),
                      (0.000988, 913.373845),
                      (['linear', 'polynomial', 'radial', 'sigmoid'])]
            X = pd.DataFrame(np.nan, index=range(num_samples), columns=colnames)
            for i in range(len(colnames)):
                if len(ranges[i]) == 2:
                    col_val = np.random.uniform(low=ranges[i][0], high=ranges[i][1],
                                                size=num_samples)
                else:
                    col_val = np.random.choice(ranges[i], size=num_samples)
                X.iloc[:, i] = col_val
            return X
        configs = random_sample_configurations(num_samples=1000)
        preds = model.predict(configs)
        # tracking the maximum AUC obtained over the functions evaluations
        preds = np.maximum.accumulate(preds)
        # computing regret (1 - predicted_auc)
        regret = 1 - preds
        # plotting the regret curve
        plt.plot(regret)
        plt.title('AUC regret for Random Search on surrogate')
        plt.xlabel('Numbe of function evaluations')
        plt.ylabel('Regret')

        openml.config.server = self.test_server
