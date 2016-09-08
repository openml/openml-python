from collections import OrderedDict
import json
import unittest

import numpy as np
import sklearn.datasets
import scipy.stats
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree

from openml.flows.sklearn import SklearnToFlowConverter
from openml.flows import OpenMLFlow


class TestSklearn(unittest.TestCase):
    
    def setUp(self):
        self.converter = SklearnToFlowConverter()
        iris = sklearn.datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

    def test_serialize_model(self):
        model = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                    max_features='auto',
                                                    max_leaf_nodes=2000)

        fixture_name = 'sklearn.tree.tree.DecisionTreeClassifier'
        fixture_description = 'Automatically created sub-component.'
        fixture_parameters = \
            OrderedDict((('class_weight', None),
                         ('criterion', 'entropy'),
                         ('max_depth', None),
                         ('max_features', 'auto'),
                         ('max_leaf_nodes', '2000'),
                         ('min_impurity_split', '1e-07'),
                         ('min_samples_leaf', '1'),
                         ('min_samples_split', '2'),
                         ('min_weight_fraction_leaf', '0.0'),
                         ('presort', 'false'),
                         ('random_state', None),
                         ('splitter', 'best')))

        serialization = self.converter.serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters, fixture_parameters)

        new_model = self.converter.deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertEqual(new_model.get_params(), model.get_params())
        new_model.fit(self.X, self.y)

    def test_serialize_model_with_subcomponent(self):
        model = sklearn.ensemble.AdaBoostClassifier(
            n_estimators=100, base_estimator=sklearn.tree.DecisionTreeClassifier())

        fixture_name = 'sklearn.ensemble.weight_boosting.AdaBoostClassifier' \
                       '(sklearn.tree.tree.DecisionTreeClassifier)'
        fixture_description = 'Automatically created sub-component.'

        serialization =  self.converter.serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters['algorithm'], 'SAMME.R')
        self.assertIsInstance(serialization.parameters['base_estimator'], str)
        self.assertEqual(serialization.parameters['learning_rate'], '1.0')
        self.assertEqual(serialization.parameters['n_estimators'], '100')

        new_model = self.converter.deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertIsNot(new_model.base_estimator, model.base_estimator)
        self.assertEqual(new_model.base_estimator.get_params(),
                         model.base_estimator.get_params())
        new_model_params = new_model.get_params()
        del new_model_params['base_estimator']
        model_params = model.get_params()
        del model_params['base_estimator']

        self.assertEqual(new_model_params, model_params)
        new_model.fit(self.X, self.y)

    def test_serialize_pipeline(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        dummy = sklearn.dummy.DummyClassifier(strategy='prior')
        model = sklearn.pipeline.Pipeline(steps=(
            ('scaler', scaler), ('dummy', dummy)))

        fixture_name = 'sklearn.pipeline.Pipeline(' \
                       'sklearn.preprocessing.data.StandardScaler,sklearn.dummy.DummyClassifier)'
        fixture_description = 'Automatically created sub-component.'

        serialization =  self.converter.serialize_object(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)

        # Comparing the pipeline
        # The parameters only have the name of base objects(not the whole flow)
        # as value
        self.assertEqual(len(serialization.parameters), 1)
        # Hard to compare two representations of a dict due to possibly
        # different sorting. Making a json makes it easier
        self.assertEqual(json.loads(serialization.parameters['steps']),
                         [{'oml:serialized_object': 'component_reference', 'value': {'key': 'steps__scaler', 'step_name': 'scaler'}},
                          {'oml:serialized_object': 'component_reference', 'value': {'key': 'steps__dummy', 'step_name': 'dummy'}}])

        # Checking the sub-component
        self.assertEqual(len(serialization.components), 2)
        self.assertIsInstance(serialization.components['steps__scaler'],
                              OpenMLFlow)
        self.assertIsInstance(serialization.components['steps__dummy'],
                              OpenMLFlow)

        #del serialization.model
        new_model = self.converter.deserialize_object(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertEqual([step[0] for step in new_model.steps],
                         [step[0] for step in model.steps])
        self.assertIsNot(new_model.steps[0][1], model.steps[0][1])
        self.assertIsNot(new_model.steps[1][1], model.steps[1][1])

        new_model_params = new_model.get_params()
        del new_model_params['scaler']
        del new_model_params['dummy']
        del new_model_params['steps']
        fu_params = model.get_params()
        del fu_params['scaler']
        del fu_params['dummy']
        del fu_params['steps']

        self.assertEqual(new_model_params, fu_params)
        new_model.fit(self.X, self.y)

    def test_serialize_feature_union(self):
        ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        scaler = sklearn.preprocessing.StandardScaler()
        fu = sklearn.pipeline.FeatureUnion(transformer_list=(('ohe', ohe),
                                                             ('scaler', scaler)))
        serialization =  self.converter.serialize_object(fu)
        new_model = self.converter.deserialize_object(serialization)

        self.assertEqual(type(new_model), type(fu))
        self.assertIsNot(new_model, fu)
        self.assertEqual(new_model.transformer_list[0][0],
                         fu.transformer_list[0][0])
        self.assertEqual(new_model.transformer_list[0][1].get_params(),
                         fu.transformer_list[0][1].get_params())
        self.assertEqual(new_model.transformer_list[1][0],
                         fu.transformer_list[1][0])
        self.assertEqual(new_model.transformer_list[1][1].get_params(),
                         fu.transformer_list[1][1].get_params())

        self.assertEqual([step[0] for step in new_model.transformer_list],
                         [step[0] for step in fu.transformer_list])
        self.assertIsNot(new_model.transformer_list[0][1], fu.transformer_list[0][1])
        self.assertIsNot(new_model.transformer_list[1][1], fu.transformer_list[1][1])

        new_model_params = new_model.get_params()
        del new_model_params['ohe']
        del new_model_params['scaler']
        del new_model_params['transformer_list']
        fu_params = fu.get_params()
        del fu_params['ohe']
        del fu_params['scaler']
        del fu_params['transformer_list']

        self.assertEqual(new_model_params, fu_params)
        new_model.fit(self.X, self.y)

    def test_serialize_type(self):
        supported_types = [float, np.float, np.float32, np.float64,
                           int, np.int, np.int32, np.int64]

        for supported_type in supported_types:
            serialized = self.converter.serialize_object(supported_type)
            deserialized = self.converter.deserialize_object(serialized)
            self.assertEqual(deserialized, supported_type)

    def test_serialize_rvs(self):
        supported_rvs = [scipy.stats.norm(loc=1, scale=5),
                         scipy.stats.expon(loc=1, scale=5),
                         scipy.stats.randint(low=-3, high=15)]

        for supported_rv in supported_rvs:
            serialized = self.converter.serialize_object(supported_rv)
            deserialized = self.converter.deserialize_object(serialized)
            self.assertEqual(type(deserialized.__dict__['dist']),
                             type(supported_rv.__dict__['dist']))
            del deserialized.__dict__['dist']
            del supported_rv.__dict__['dist']
            self.assertEqual(deserialized.__dict__,
                             supported_rv.__dict__)

    def test_serialize_function(self):
        serialized =  self.converter.serialize_object(sklearn.feature_selection.chi2)
        deserialized = self.converter.deserialize_object(serialized)
        self.assertEqual(deserialized, sklearn.feature_selection.chi2)

    def test_serialize_simple_parameter_grid(self):
        # TODO instead a GridSearchCV object should be serialized

        # We cannot easily test for scipy random variables in here, but they
        # should be covered

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
              "criterion": ["gini", "entropy"]}]

        for grid in grids:
            serialized = self.converter.serialize_object(grid)
            deserialized = self.converter.deserialize_object(serialized)

            self.assertEqual(deserialized, grid)
            self.assertIsNot(deserialized, grid)

    @unittest.skip('This feature needs further reworking. If we allow several '
                   'components, we need to register them all in the downstream '
                   'flows. This is so far not implemented.')
    def test_serialize_advanced_grid(self):
        # TODO instead a GridSearchCV object should be serialized

        # This needs to be in its own function because we cannot simply check
        # for the equality of the grid, because scikit-learn objects don't
        # really support the equality operator
        # This will only work with sklearn==0.18
        N_FEATURES_OPTIONS = [2, 4, 8]
        C_OPTIONS = [1, 10, 100, 1000]
        grid = [{'reduce_dim': [sklearn.decomposition.PCA(iterated_power=7),
                                sklearn.decomposition.NMF()],
                 'reduce_dim__n_components': N_FEATURES_OPTIONS,
                 'classify__C': C_OPTIONS},
                {'reduce_dim': [sklearn.feature_selection.SelectKBest(
                                sklearn.feature_selection.chi2)],
                 'reduce_dim__k': N_FEATURES_OPTIONS,
                 'classify__C': C_OPTIONS}]

        serialized = self.converter.serialize_object(grid)
        deserialized = self.converter.deserialize_object(serialized)

        self.assertEqual(grid[0]['reduce_dim'][0].get_params(),
                         deserialized[0]['reduce_dim'][0].get_params())
        self.assertIsNot(grid[0]['reduce_dim'][0],
                         deserialized[0]['reduce_dim'][0])
        self.assertEqual(grid[0]['reduce_dim'][1].get_params(),
                         deserialized[0]['reduce_dim'][1].get_params())
        self.assertIsNot(grid[0]['reduce_dim'][1],
                         deserialized[0]['reduce_dim'][1])
        self.assertEqual(grid[0]['reduce_dim__n_components'],
                         deserialized[0]['reduce_dim__n_components'])
        self.assertEqual(grid[0]['classify__C'],
                         deserialized[0]['classify__C'])
        self.assertEqual(grid[1]['reduce_dim'][0].get_params(),
                         deserialized[1]['reduce_dim'][0].get_params())
        self.assertIsNot(grid[1]['reduce_dim'][0],
                         deserialized[1]['reduce_dim'][0])
        self.assertEqual(grid[1]['reduce_dim__k'],
                         deserialized[1]['reduce_dim__k'])
        self.assertEqual(grid[1]['classify__C'],
                         deserialized[1]['classify__C'])

    def test_serialize_resampling(self):
        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=4, shuffle=True)
        serialized =  self.converter.serialize_object(kfold)
        deserialized = self.converter.deserialize_object(serialized)
        # Best approximation to get_params()
        self.assertEqual(str(deserialized), str(kfold))
        self.assertIsNot(deserialized, kfold)

    def test_hypothetical_parameter_values(self):
        values = ['true', '1', '0.1']
        for value in values:
            serialized = self.converter.serialize_object(value)
            deserialized = self.converter.deserialize_object(value)
            self.assertEqual(deserialized, value)



