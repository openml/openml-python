from collections import OrderedDict
import json
import os
import sys
import unittest
import warnings

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import numpy as np
import scipy.optimize
import scipy.stats
import sklearn.base
import sklearn.datasets
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree
import sklearn.cluster

import openml
from openml.flows import OpenMLFlow, sklearn_to_flow, flow_to_sklearn
from openml.flows.functions import assert_flows_equal
from openml.flows.sklearn_converter import _format_external_version, \
    _check_dependencies, _check_n_jobs
from openml.exceptions import PyOpenMLError

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


__version__ = 0.1


class Model(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, X, y):
        pass


class TestSklearn(unittest.TestCase):
    # Splitting not helpful, these test's don't rely on the server and take less
    # than 1 seconds
    
    def setUp(self):
        iris = sklearn.datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

    @mock.patch('openml.flows.sklearn_converter._check_dependencies')
    def test_serialize_model(self, check_dependencies_mock):
        model = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                    max_features='auto',
                                                    max_leaf_nodes=2000)

        fixture_name = 'sklearn.tree.tree.DecisionTreeClassifier'
        fixture_description = 'Automatically created scikit-learn flow.'
        version_fixture = 'sklearn==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                          % sklearn.__version__
        fixture_parameters = \
            OrderedDict((('class_weight', 'null'),
                         ('criterion', '"entropy"'),
                         ('max_depth', 'null'),
                         ('max_features', '"auto"'),
                         ('max_leaf_nodes', '2000'),
                         ('min_impurity_split', '1e-07'),
                         ('min_samples_leaf', '1'),
                         ('min_samples_split', '2'),
                         ('min_weight_fraction_leaf', '0.0'),
                         ('presort', 'false'),
                         ('random_state', 'null'),
                         ('splitter', '"best"')))

        serialization = sklearn_to_flow(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.class_name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters, fixture_parameters)
        self.assertEqual(serialization.dependencies, version_fixture)

        new_model = flow_to_sklearn(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertEqual(new_model.get_params(), model.get_params())
        new_model.fit(self.X, self.y)

        self.assertEqual(check_dependencies_mock.call_count, 1)


    @mock.patch('openml.flows.sklearn_converter._check_dependencies')
    def test_serialize_model_clustering(self, check_dependencies_mock):
        model = sklearn.cluster.KMeans()

        fixture_name = 'sklearn.cluster.k_means_.KMeans'
        fixture_description = 'Automatically created scikit-learn flow.'
        version_fixture = 'sklearn==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                          % sklearn.__version__
        fixture_parameters = \
            OrderedDict((('algorithm', '"auto"'),
                         ('copy_x', 'true'),
                         ('init', '"k-means++"'),
                         ('max_iter', '300'),
                         ('n_clusters', '8'),
                         ('n_init', '10'),
                         ('n_jobs', '1'),
                         ('precompute_distances', '"auto"'),
                         ('random_state', 'null'),
                         ('tol', '0.0001'),
                         ('verbose', '0')))

        serialization = sklearn_to_flow(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.class_name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters, fixture_parameters)
        self.assertEqual(serialization.dependencies, version_fixture)

        new_model = flow_to_sklearn(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertEqual(new_model.get_params(), model.get_params())
        new_model.fit(self.X)

        self.assertEqual(check_dependencies_mock.call_count, 1)


    def test_serialize_model_with_subcomponent(self):
        model = sklearn.ensemble.AdaBoostClassifier(
            n_estimators=100, base_estimator=sklearn.tree.DecisionTreeClassifier())

        fixture_name = 'sklearn.ensemble.weight_boosting.AdaBoostClassifier' \
                       '(base_estimator=sklearn.tree.tree.DecisionTreeClassifier)'
        fixture_class_name = 'sklearn.ensemble.weight_boosting.AdaBoostClassifier'
        fixture_description = 'Automatically created scikit-learn flow.'
        fixture_subcomponent_name = 'sklearn.tree.tree.DecisionTreeClassifier'
        fixture_subcomponent_class_name = 'sklearn.tree.tree.DecisionTreeClassifier'
        fixture_subcomponent_description = 'Automatically created scikit-learn flow.'

        serialization =  sklearn_to_flow(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.class_name, fixture_class_name)
        self.assertEqual(serialization.description, fixture_description)
        self.assertEqual(serialization.parameters['algorithm'], '"SAMME.R"')
        self.assertIsInstance(serialization.parameters['base_estimator'], str)
        self.assertEqual(serialization.parameters['learning_rate'], '1.0')
        self.assertEqual(serialization.parameters['n_estimators'], '100')
        self.assertEqual(serialization.components['base_estimator'].name,
                         fixture_subcomponent_name)
        self.assertEqual(serialization.components['base_estimator'].class_name,
                         fixture_subcomponent_class_name)
        self.assertEqual(serialization.components['base_estimator'].description,
                         fixture_subcomponent_description)

        new_model = flow_to_sklearn(serialization)

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
                       'scaler=sklearn.preprocessing.data.StandardScaler,' \
                       'dummy=sklearn.dummy.DummyClassifier)'
        fixture_description = 'Automatically created scikit-learn flow.'

        serialization =  sklearn_to_flow(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)

        # Comparing the pipeline
        # The parameters only have the name of base objects(not the whole flow)
        # as value
        self.assertEqual(len(serialization.parameters), 1)
        # Hard to compare two representations of a dict due to possibly
        # different sorting. Making a json makes it easier
        self.assertEqual(json.loads(serialization.parameters['steps']),
                         [{'oml-python:serialized_object':
                               'component_reference', 'value': {'key': 'scaler', 'step_name': 'scaler'}},
                          {'oml-python:serialized_object':
                               'component_reference', 'value': {'key': 'dummy', 'step_name': 'dummy'}}])

        # Checking the sub-component
        self.assertEqual(len(serialization.components), 2)
        self.assertIsInstance(serialization.components['scaler'],
                              OpenMLFlow)
        self.assertIsInstance(serialization.components['dummy'],
                              OpenMLFlow)

        #del serialization.model
        new_model = flow_to_sklearn(serialization)

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

    def test_serialize_pipeline_clustering(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        km = sklearn.cluster.KMeans()
        model = sklearn.pipeline.Pipeline(steps=(
            ('scaler', scaler), ('clusterer', km)))

        fixture_name = 'sklearn.pipeline.Pipeline(' \
                       'scaler=sklearn.preprocessing.data.StandardScaler,' \
                       'clusterer=sklearn.cluster.k_means_.KMeans)'
        fixture_description = 'Automatically created scikit-learn flow.'

        serialization = sklearn_to_flow(model)

        self.assertEqual(serialization.name, fixture_name)
        self.assertEqual(serialization.description, fixture_description)

        # Comparing the pipeline
        # The parameters only have the name of base objects(not the whole flow)
        # as value
        self.assertEqual(len(serialization.parameters), 1)
        # Hard to compare two representations of a dict due to possibly
        # different sorting. Making a json makes it easier
        self.assertEqual(json.loads(serialization.parameters['steps']),
                         [{'oml-python:serialized_object':
                               'component_reference', 'value': {'key': 'scaler', 'step_name': 'scaler'}},
                          {'oml-python:serialized_object':
                               'component_reference', 'value': {'key': 'clusterer', 'step_name': 'clusterer'}}])

        # Checking the sub-component
        self.assertEqual(len(serialization.components), 2)
        self.assertIsInstance(serialization.components['scaler'],
                              OpenMLFlow)
        self.assertIsInstance(serialization.components['clusterer'],
                              OpenMLFlow)

        # del serialization.model
        new_model = flow_to_sklearn(serialization)

        self.assertEqual(type(new_model), type(model))
        self.assertIsNot(new_model, model)

        self.assertEqual([step[0] for step in new_model.steps],
                         [step[0] for step in model.steps])
        self.assertIsNot(new_model.steps[0][1], model.steps[0][1])
        self.assertIsNot(new_model.steps[1][1], model.steps[1][1])

        new_model_params = new_model.get_params()
        del new_model_params['scaler']
        del new_model_params['clusterer']
        del new_model_params['steps']
        fu_params = model.get_params()
        del fu_params['scaler']
        del fu_params['clusterer']
        del fu_params['steps']

        self.assertEqual(new_model_params, fu_params)
        new_model.fit(self.X, self.y)

    def test_serialize_feature_union(self):
        ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        scaler = sklearn.preprocessing.StandardScaler()
        fu = sklearn.pipeline.FeatureUnion(transformer_list=[('ohe', ohe),
                                                             ('scaler', scaler)])
        serialization =  sklearn_to_flow(fu)
        self.assertEqual(serialization.name,
                         'sklearn.pipeline.FeatureUnion('
                         'ohe=sklearn.preprocessing.data.OneHotEncoder,'
                         'scaler=sklearn.preprocessing.data.StandardScaler)')
        new_model = flow_to_sklearn(serialization)

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

        fu.set_params(scaler=None)
        serialization = sklearn_to_flow(fu)
        self.assertEqual(serialization.name,
                         'sklearn.pipeline.FeatureUnion('
                         'ohe=sklearn.preprocessing.data.OneHotEncoder)')
        new_model = flow_to_sklearn(serialization)
        self.assertEqual(type(new_model), type(fu))
        self.assertIsNot(new_model, fu)
        self.assertIs(new_model.transformer_list[1][1], None)

    def test_serialize_feature_union_switched_names(self):
        ohe = sklearn.preprocessing.OneHotEncoder()
        scaler = sklearn.preprocessing.StandardScaler()
        fu1 = sklearn.pipeline.FeatureUnion(transformer_list=[('ohe', ohe), ('scaler', scaler)])
        fu2 = sklearn.pipeline.FeatureUnion(transformer_list=[('scaler', ohe), ('ohe', scaler)])
        fu1_serialization = sklearn_to_flow(fu1)
        fu2_serialization = sklearn_to_flow(fu2)
        self.assertEqual(
            fu1_serialization.name,
            "sklearn.pipeline.FeatureUnion("
            "ohe=sklearn.preprocessing.data.OneHotEncoder,"
            "scaler=sklearn.preprocessing.data.StandardScaler)")
        self.assertEqual(
            fu2_serialization.name,
            "sklearn.pipeline.FeatureUnion("
            "scaler=sklearn.preprocessing.data.OneHotEncoder,"
            "ohe=sklearn.preprocessing.data.StandardScaler)")

    def test_serialize_complex_flow(self):
        ohe = sklearn.preprocessing.OneHotEncoder(categorical_features=[0])
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier())
        model = sklearn.pipeline.Pipeline(steps=(
            ('ohe', ohe), ('scaler', scaler), ('boosting', boosting)))
        parameter_grid = {'n_estimators': [1, 5, 10, 100],
                          'learning_rate': scipy.stats.uniform(0.01, 0.99),
                          'base_estimator__max_depth': scipy.stats.randint(1,
                                                                           10)}
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
        rs = sklearn.model_selection.RandomizedSearchCV(
            estimator=model, param_distributions=parameter_grid, cv=cv)
        serialized = sklearn_to_flow(rs)

        fixture_name = 'sklearn.model_selection._search.RandomizedSearchCV(' \
                       'estimator=sklearn.pipeline.Pipeline(' \
                       'ohe=sklearn.preprocessing.data.OneHotEncoder,' \
                       'scaler=sklearn.preprocessing.data.StandardScaler,' \
                       'boosting=sklearn.ensemble.weight_boosting.AdaBoostClassifier(' \
                       'base_estimator=sklearn.tree.tree.DecisionTreeClassifier)))'
        self.assertEqual(serialized.name, fixture_name)

        # now do deserialization
        deserialized = flow_to_sklearn(serialized)

        # Checks that sklearn_to_flow is idempotent.
        serialized2 = sklearn_to_flow(deserialized)
        self.assertNotEqual(rs, deserialized)
        # Would raise an exception if the flows would be unequal
        assert_flows_equal(serialized, serialized2)

    def test_serialize_type(self):
        supported_types = [float, np.float, np.float32, np.float64,
                           int, np.int, np.int32, np.int64]

        for supported_type in supported_types:
            serialized = sklearn_to_flow(supported_type)
            deserialized = flow_to_sklearn(serialized)
            self.assertEqual(deserialized, supported_type)

    def test_serialize_rvs(self):
        supported_rvs = [scipy.stats.norm(loc=1, scale=5),
                         scipy.stats.expon(loc=1, scale=5),
                         scipy.stats.randint(low=-3, high=15)]

        for supported_rv in supported_rvs:
            serialized = sklearn_to_flow(supported_rv)
            deserialized = flow_to_sklearn(serialized)
            self.assertEqual(type(deserialized.dist), type(supported_rv.dist))
            del deserialized.dist
            del supported_rv.dist
            self.assertEqual(deserialized.__dict__,
                             supported_rv.__dict__)

    def test_serialize_function(self):
        serialized =  sklearn_to_flow(sklearn.feature_selection.chi2)
        deserialized = flow_to_sklearn(serialized)
        self.assertEqual(deserialized, sklearn.feature_selection.chi2)

    def test_serialize_cvobject(self):
        methods = [sklearn.model_selection.KFold(3),
                   sklearn.model_selection.LeaveOneOut()]
        fixtures = [OrderedDict([('oml-python:serialized_object', 'cv_object'),
                                 ('value', OrderedDict([('name', 'sklearn.model_selection._split.KFold'),
                                                        ('parameters', OrderedDict([('n_splits', '3'),
                                                                                    ('random_state', 'null'),
                                                                                    ('shuffle', 'false')]))]))]),
                    OrderedDict([('oml-python:serialized_object', 'cv_object'),
                                 ('value', OrderedDict([('name', 'sklearn.model_selection._split.LeaveOneOut'),
                                                        ('parameters', OrderedDict())]))])]
        for method, fixture in zip(methods, fixtures):
            m = sklearn_to_flow(method)
            self.assertEqual(m, fixture)

            m_new = flow_to_sklearn(m)
            self.assertIsNot(m_new, m)
            self.assertIsInstance(m_new, type(method))

    def test_serialize_simple_parameter_grid(self):

        # We cannot easily test for scipy random variables in here, but they
        # should be covered

        # Examples from the scikit-learn documentation
        models = [sklearn.svm.SVC(), sklearn.ensemble.RandomForestClassifier()]
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

        for grid, model in zip(grids, models):
            serialized = sklearn_to_flow(grid)
            deserialized = flow_to_sklearn(serialized)

            self.assertEqual(deserialized, grid)
            self.assertIsNot(deserialized, grid)

            hpo = sklearn.model_selection.GridSearchCV(
                param_grid=grid, estimator=model)

            serialized = sklearn_to_flow(hpo)
            deserialized = flow_to_sklearn(serialized)
            self.assertEqual(hpo.param_grid, deserialized.param_grid)
            self.assertEqual(hpo.estimator.get_params(),
                             deserialized.estimator.get_params())
            hpo_params = hpo.get_params(deep=False)
            deserialized_params = deserialized.get_params(deep=False)
            del hpo_params['estimator']
            del deserialized_params['estimator']
            self.assertEqual(hpo_params, deserialized_params)

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

        serialized = sklearn_to_flow(grid)
        deserialized = flow_to_sklearn(serialized)

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
        serialized =  sklearn_to_flow(kfold)
        deserialized = flow_to_sklearn(serialized)
        # Best approximation to get_params()
        self.assertEqual(str(deserialized), str(kfold))
        self.assertIsNot(deserialized, kfold)

    def test_hypothetical_parameter_values(self):
        # The hypothetical parameter values of true, 1, 0.1 formatted as a
        # string (and their correct serialization and deserialization) an only
        #  be checked inside a model

        model = Model('true', '1', '0.1')

        serialized = sklearn_to_flow(model)
        deserialized = flow_to_sklearn(serialized)
        self.assertEqual(deserialized.get_params(), model.get_params())
        self.assertIsNot(deserialized, model)

    def test_gaussian_process(self):
        opt = scipy.optimize.fmin_l_bfgs_b
        kernel = sklearn.gaussian_process.kernels.Matern()
        gp = sklearn.gaussian_process.GaussianProcessClassifier(
            kernel=kernel, optimizer=opt)
        self.assertRaisesRegexp(TypeError, "Matern\(length_scale=1, nu=1.5\), "
                                           "<class 'sklearn.gaussian_process.kernels.Matern'>",
                                sklearn_to_flow, gp)

    def test_error_on_adding_component_multiple_times_to_flow(self):
        # this function implicitly checks
        # - openml.flows._check_multiple_occurence_of_component_in_flow()
        pca = sklearn.decomposition.PCA()
        pca2 = sklearn.decomposition.PCA()
        pipeline = sklearn.pipeline.Pipeline((('pca1', pca), ('pca2', pca2)))
        fixture = "Found a second occurence of component sklearn.decomposition.pca.PCA" \
                  " when trying to serialize Pipeline\(steps=\(\('pca1', " \
                  "PCA\(copy=True, iterated_power='auto', n_components=None, " \
                  "random_state=None,\n" \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\), " \
                  "\('pca2', PCA\(copy=True, iterated_power='auto', " \
                  "n_components=None, random_state=None,\n" \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\)\)\)."
        self.assertRaisesRegexp(ValueError, fixture, sklearn_to_flow, pipeline)

        fu = sklearn.pipeline.FeatureUnion((('pca1', pca), ('pca2', pca2)))
        fixture = "Found a second occurence of component sklearn.decomposition.pca.PCA when trying to serialize " \
                  "FeatureUnion\(n_jobs=1,\n" \
                  "       transformer_list=\(\('pca1', PCA\(copy=True, " \
                  "iterated_power='auto'," \
                  " n_components=None, random_state=None,\n" \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\), \('pca2', " \
                  "PCA\(copy=True, iterated_power='auto'," \
                  " n_components=None, random_state=None,\n" \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\)\),\n" \
                  "       transformer_weights=None\)."
        self.assertRaisesRegexp(ValueError, fixture, sklearn_to_flow, fu)

        fs = sklearn.feature_selection.SelectKBest()
        fu2 = sklearn.pipeline.FeatureUnion((('pca1', pca), ('fs', fs)))
        pipeline2 = sklearn.pipeline.Pipeline((('fu', fu2), ('pca2', pca2)))
        fixture = "Found a second occurence of component " \
                  "sklearn.decomposition.pca.PCA when trying to serialize " \
                  "Pipeline\(steps=\(\('fu', FeatureUnion\(n_jobs=1,\n" \
                  "       transformer_list=\(\('pca1', PCA\(copy=True, " \
                  "iterated_power='auto'," \
                  " n_components=None, random_state=None,\n"  \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\), " \
                  "\('fs', SelectKBest\(k=10, score_func=<function " \
                  "f_classif at 0x[a-z0-9]+>\)\)\),\n" \
                  "       transformer_weights=None\)\), \('pca2', " \
                  "PCA\(copy=True, iterated_power='auto'," \
                  " n_components=None, random_state=None,\n" \
                  "  svd_solver='auto', tol=0.0, whiten=False\)\)\)\)."
        self.assertRaisesRegexp(ValueError, fixture, sklearn_to_flow, pipeline2)

    def test_subflow_version_propagated(self):
        this_directory = os.path.dirname(os.path.abspath(__file__))
        tests_directory = os.path.abspath(os.path.join(this_directory, '..', '..'))
        sys.path.append(tests_directory)
        import tests.test_flows.dummy_learn.dummy_forest
        pca = sklearn.decomposition.PCA()
        dummy = tests.test_flows.dummy_learn.dummy_forest.DummyRegressor()
        pipeline = sklearn.pipeline.Pipeline((('pca', pca), ('dummy', dummy)))
        flow = sklearn_to_flow(pipeline)
        # In python2.7, the unit tests work differently on travis-ci; therefore,
        # I put the alternative travis-ci answer here as well. While it has a
        # different value, it is still correct as it is a propagation of the
        # subclasses' module name
        self.assertEqual(flow.external_version, '%s,%s,%s' % (
            _format_external_version('openml', openml.__version__),
            _format_external_version('sklearn', sklearn.__version__),
            _format_external_version('tests', '0.1')))

    @mock.patch('warnings.warn')
    def test_check_dependencies(self, warnings_mock):
        dependencies = ['sklearn==0.1', 'sklearn>=99.99.99', 'sklearn>99.99.99']
        for dependency in dependencies:
            self.assertRaises(ValueError, _check_dependencies, dependency)

    def test_illegal_parameter_names(self):
        # illegal name: estimators
        clf1 = sklearn.ensemble.VotingClassifier(
            estimators=[('estimators', sklearn.ensemble.RandomForestClassifier()),
                        ('whatevs', sklearn.ensemble.ExtraTreesClassifier())])
        clf2 = sklearn.ensemble.VotingClassifier(
            estimators=[('whatevs', sklearn.ensemble.RandomForestClassifier()),
                        ('estimators', sklearn.ensemble.ExtraTreesClassifier())])
        cases = [clf1, clf2]

        for case in cases:
            self.assertRaises(PyOpenMLError, sklearn_to_flow, case)

    def test_illegal_parameter_names_pipeline(self):
        # illegal name: steps
        steps = [
            ('Imputer', sklearn.preprocessing.Imputer(strategy='median')),
            ('OneHotEncoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')),
            ('steps', sklearn.ensemble.BaggingClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier))
        ]
        self.assertRaises(ValueError, sklearn.pipeline.Pipeline, steps=steps)


    def test_illegal_parameter_names_featureunion(self):
        # illegal name: transformer_list
        transformer_list = [
            ('transformer_list', sklearn.preprocessing.Imputer(strategy='median')),
            ('OneHotEncoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ]
        self.assertRaises(ValueError, sklearn.pipeline.FeatureUnion, transformer_list=transformer_list)

    def test_paralizable_check(self):
        # using this model should pass the test (if param distribution is legal)
        singlecore_bagging = sklearn.ensemble.BaggingClassifier()
        # using this model should return false (if param distribution is legal)
        multicore_bagging = sklearn.ensemble.BaggingClassifier(n_jobs=5)
        # using this param distribution should raise an exception
        illegal_param_dist = {"base__n_jobs": [-1, 0, 1] }
        # using this param distribution should not raise an exception
        legal_param_dist = {"base__max_depth": [2, 3, 4]}

        legal_models = [
            sklearn.ensemble.RandomForestClassifier(),
            sklearn.ensemble.RandomForestClassifier(n_jobs=5),
            sklearn.ensemble.RandomForestClassifier(n_jobs=-1),
            sklearn.pipeline.Pipeline(steps=[('bag', sklearn.ensemble.BaggingClassifier(n_jobs=1))]),
            sklearn.pipeline.Pipeline(steps=[('bag', sklearn.ensemble.BaggingClassifier(n_jobs=5))]),
            sklearn.pipeline.Pipeline(steps=[('bag', sklearn.ensemble.BaggingClassifier(n_jobs=-1))]),
            sklearn.model_selection.GridSearchCV(singlecore_bagging, legal_param_dist),
            sklearn.model_selection.GridSearchCV(multicore_bagging, legal_param_dist)
        ]
        illegal_models = [
            sklearn.model_selection.GridSearchCV(singlecore_bagging, illegal_param_dist),
            sklearn.model_selection.GridSearchCV(multicore_bagging, illegal_param_dist)
        ]

        answers = [True, False, False, True, False, False, True, False]

        for i in range(len(legal_models)):
            self.assertTrue(_check_n_jobs(legal_models[i]) == answers[i])

        for i in range(len(illegal_models)):
            self.assertRaises(PyOpenMLError, _check_n_jobs, illegal_models[i])
