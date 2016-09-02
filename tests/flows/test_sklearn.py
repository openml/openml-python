from collections import OrderedDict
import unittest

import numpy as np
import scipy.stats
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree

from openml.flows.sklearn import serialize_object, deserialize_object
from openml.flows import OpenMLFlow


class TestSklearn(unittest.TestCase):

    def test_serialize_model(self):
        model = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                    max_features='auto',
                                                    max_leaf_nodes=2000)

        fixture_name = 'sklearn.tree.tree.DecisionTreeClassifier'
        fixture_description = 'Automatically created sub-component.'
        fixture_parameters = \
            [OrderedDict((('oml:name', 'class_weight'),
                          ('oml:default_value', None))),
             OrderedDict((('oml:name', 'criterion'),
                          ('oml:default_value', 'entropy'))),
             OrderedDict((('oml:name', 'max_depth'),
                          ('oml:default_value', None))),
             OrderedDict((('oml:name', 'max_features'),
                          ('oml:default_value', 'auto'))),
             OrderedDict((('oml:name', 'max_leaf_nodes'),
                          ('oml:default_value', 2000))),
             OrderedDict((('oml:name', 'min_impurity_split'),
                          ('oml:default_value', 1e-07))),
             OrderedDict((('oml:name', 'min_samples_leaf'),
                          ('oml:default_value', 1))),
             OrderedDict((('oml:name', 'min_samples_split'),
                          ('oml:default_value', 2))),
             OrderedDict((('oml:name', 'min_weight_fraction_leaf'),
                          ('oml:default_value', 0.0))),
             OrderedDict((('oml:name', 'presort'),
                          ('oml:default_value', False))),
             OrderedDict((('oml:name', 'random_state'),
                          ('oml:default_value', None))),
             OrderedDict((('oml:name', 'splitter'),
                          ('oml:default_value', 'best')))]

        serialization = serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters, fixture_parameters)

        new_model = deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertEqual(new_model.get_params(), model.get_params())

    def test_serialize_model_with_subcomponent(self):
        model = sklearn.ensemble.AdaBoostClassifier(
            n_estimators=100, base_estimator=sklearn.tree.DecisionTreeClassifier())

        fixture_name = 'sklearn.ensemble.weight_boosting.AdaBoostClassifier' \
                       '(sklearn.tree.tree.DecisionTreeClassifier)'
        fixture_description = 'Automatically created sub-component.'

        serialization = serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters[0]['oml:default_value'],
                         'SAMME.R')
        self.assertIsInstance(serialization.parameters[1]['oml:default_value'],
                              OpenMLFlow)
        self.assertEqual(serialization.parameters[2]['oml:default_value'],
                         1.0)
        self.assertEqual(serialization.parameters[3]['oml:default_value'],
                         100)

        new_model = deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertEqual(new_model.get_params(), model.get_params())

    def test_serialize_pipeline(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        dummy = sklearn.dummy.DummyClassifier(strategy='random')
        model = sklearn.pipeline.Pipeline(steps=(
            ('scaler', scaler), ('dummy', dummy)))

        fixture_name = 'sklearn.pipeline.Pipeline(' \
                       'sklearn.preprocessing.data.StandardScaler,sklearn.dummy.DummyClassifier)'
        fixture_description = 'Automatically created sub-component.'

        serialization = serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)

        # Comparing the pipeline
        # Harder to compare since there are flow objects in the pipeline
        self.assertEqual(len(serialization.parameters), 1)
        self.assertEqual(serialization.parameters[0]['oml:name'], 'steps')
        self.assertEqual(serialization.parameters[0]['oml:default_value'][0][0],
                         'scaler')
        self.assertIsInstance(serialization.parameters[0]['oml:default_value'][0][1],
                              OpenMLFlow)
        self.assertEqual(serialization.parameters[0]['oml:default_value'][1][0],
                         'dummy')
        self.assertIsInstance(serialization.parameters[0]['oml:default_value'][1][1],
                              OpenMLFlow)

        # Checking the sub-component
        self.assertEqual(len(serialization.components), 2)
        self.assertEqual(serialization.parameters[0]['oml:default_value'][0][1],
                         serialization.components[0]['oml:flow'])
        self.assertEqual(serialization.parameters[0]['oml:default_value'][1][1],
                         serialization.components[1]['oml:flow'])


        new_model = deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertEqual(new_model.get_params(), model.get_params())

    def test_serialize_feature_union(self):
        ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        scaler = sklearn.preprocessing.StandardScaler()
        fu = sklearn.pipeline.FeatureUnion(transformer_list=(('ohe', ohe),
                                                             ('scaler', scaler)))
        serialization = serialize_object(fu)
        new_model = deserialize_object(serialization)

        self.assertEqual(type(new_model), type(fu))
        self.assertEqual(new_model.get_params(), fu.get_params())

    def test_serialize_type(self):
        supported_types = [float, np.float, np.float32, np.float64,
                           int, np.int, np.int32, np.int64]

        for supported_type in supported_types:
            serialized = (serialize_object(supported_type))
            deserialized = deserialize_object(serialized)
            self.assertEqual(deserialized, supported_type)

    def test_serialize_rvs(self):
        supported_rvs = [scipy.stats.norm(loc=1, scale=5),
                         scipy.stats.expon(loc=1, scale=5),
                         scipy.stats.randint(low=-3, high=15)]

        for supported_rv in supported_rvs:
            serialized = (serialize_object(supported_rv))
            deserialized = deserialize_object(serialized)
            self.assertEqual(type(deserialized.__dict__['dist']),
                             type(supported_rv.__dict__['dist']))
            del deserialized.__dict__['dist']
            del supported_rv.__dict__['dist']
            self.assertEqual(deserialized.__dict__,
                             supported_rv.__dict__)

    def test_serialize_function(self):
        serialized = serialize_object(sklearn.feature_selection.chi2)
        deserialized = deserialize_object(serialized)
        self.assertEqual(deserialized, sklearn.feature_selection.chi2)

    def test_serialize_parameter_grid(self):
        # We cannot easily test for scipy random variables in here, but they
        # should be covered

        N_FEATURES_OPTIONS = [2, 4, 8]
        C_OPTIONS = [1, 10, 100, 1000]

        # Examples from the scikit-learn documentation
        grids = \
            [[{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}],
             {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]},
             # This will only work with sklearn==0.18
             [{'reduce_dim': [sklearn.decomposition.PCA(iterated_power=7),
                              sklearn.decomposition.NMF()],
               'reduce_dim__n_components': N_FEATURES_OPTIONS,
               'classify__C': C_OPTIONS},
              {'reduce_dim': [sklearn.feature_selection.SelectKBest(
                  sklearn.feature_selection.chi2)],
               'reduce_dim__k': N_FEATURES_OPTIONS,
               'classify__C': C_OPTIONS}]]

        for grid in grids:
            serialized = serialize_object(grid)
            deserialized = deserialize_object(serialized)
            self.assertEqual(deserialized, grid)

    def test_serialize_resampling(self):
        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=4, shuffle=True)
        serialized = serialize_object(kfold)
        deserialized = deserialize_object(serialized)
        self.assertEqual(deserialized, kfold)



