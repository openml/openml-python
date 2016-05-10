from collections import OrderedDict
import unittest
import warnings

import numpy as np

import sklearn
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from openml.sklearn.model_selection import StratifiedKFold
import xmltodict
import openml.sklearn.stats

from openml.testing import TestBase
import openml
import openml.flows.flow


class TestFlow(TestBase):
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_flow_list(self):
        def check_flow(flow):
            self.assertIsInstance(flow, dict)
            self.assertEqual(len(flow), 6)

        flows = openml.flows.get_flow_list()
        self.assertGreaterEqual(len(flows), 1448)
        for flow in flows:
            check_flow(flow)

    def test_upload_simple_flow(self):
        flow = openml.OpenMLFlow(name='Test',
                                 model=DummyClassifier(),
                                 description="test description")
        return_code, return_value = flow.publish()

        self.assertEqual(return_code, 200)
        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        new_flow = openml.flows.get_flow(flow_id)

        self.assertEqual(new_flow.name, '%s%s' % (self.sentinel, flow.name))
        self.assertEqual(new_flow.description, flow.description)
        self.assertIn('sklearn', new_flow.dependencies)
        self.assertIn('numpy', new_flow.dependencies)
        self.assertIn('scipy', new_flow.dependencies)

    def test_init_parameters_and_components(self):
        # Deal with hyperparameter being a list or tuple because it is part
        # of a pipeline
        model = Pipeline((('ohe', OneHotEncoder(categorical_features=[
                                                True, False])),
                          ('scaler', StandardScaler(with_mean=False)),
                          ('classifier', AdaBoostClassifier(
                              DecisionTreeClassifier()))))
        param_distributions = OrderedDict()
        param_distributions['classifier__n_estimators'] = \
            openml.sklearn.stats.Discrete([1, 2, 5])
        param_distributions['classifier__base_estimator__max_depth'] = \
            openml.sklearn.stats.RandInt(3, 10)
        cv = StratifiedKFold(n_folds=2, shuffle=True)
        rs = RandomizedSearchCV(estimator=model,
                                param_distributions=param_distributions,
                                cv=cv)
        X = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        rs.fit(X, y)

        flow = openml.OpenMLFlow(description='Test flow!', model=rs,
                                 external_version='Sklearn_0.18.0dev')
        flow.init_parameters_and_components()

        self.assertEqual(len(flow.parameters), 56)

        parameter_dict = {}
        for parameter in flow.parameters:
            if 'oml:default_value' not in parameter:
                continue
            parameter_dict[parameter['oml:name']] = parameter['oml:default_value']

        self.assertEqual(parameter_dict['parameter_distribution__estimator__classifier__n_estimators'],
                         "Unparametrized")
        self.assertEqual(parameter_dict['parameter_distribution__estimator__classifier__base_estimator__max_depth'],
                         "Unparametrized")
        # Check one other if it unparametrized
        self.assertEqual(parameter_dict['parameter_distribution__estimator__classifier__learning_rate'],
                         "Unparametrized")

        self.assertEqual(len(flow.components), 2)

        component_estimator = flow.components[1]['oml:flow']
        self.assertEqual(len(component_estimator.components), 3)
        # External version should be propagated!
        self.assertEqual(component_estimator.components[2][
                             'oml:flow'].external_version,
                         'Sklearn_0.18.0dev')
        self.assertEqual(component_estimator.components[0][
                             'oml:flow'].parameters[0]['oml:default_value'],
                         '[True, False]')


    def test_upload_flow_with_exchangable_component(self):
        # The classifier inside AdaBoost can be changed, therefore we
        # register the classifier inside AdaBoost if it is not registered yet!
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

        flow = openml.OpenMLFlow(description='Test flow!', model=model,
                                 external_version='sklearn_' + sklearn.__version__)
        return_code, return_value = flow.publish()

        self.assertEqual(return_code, 200)
        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        new_flow = openml.flows.get_flow(flow_id)
        self.assertEqual(len(new_flow.parameters), 5)
        for parameter in new_flow.parameters:
            self.assertIn(parameter['name'], AdaBoostClassifier().get_params())
        self.assertEqual(len(new_flow.components), 1)
        self.assertEqual(new_flow.components[0]['identifier'], 'base_estimator')
        component = new_flow.components[0]['flow']
        self.assertGreaterEqual(len(component.parameters), 9)
        for parameter in component.parameters:
            self.assertIn(parameter['name'], DecisionTreeClassifier().get_params())

    def test_upload_flow_reuse_component(self):
        # The classifier inside AdaBoost can be changed, therefore we
        # register the classifier inside AdaBoost if it is not registered yet!
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

        flow = openml.OpenMLFlow(description='Test flow!', model=model,
                                 external_version='sklearn_' + sklearn.__version__)
        return_code, return_value = flow.publish()
        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        flow_1 = openml.flows.flow.get_flow(flow_id)

        return_code, return_value = flow.publish()
        response_dict = xmltodict.parse(return_value)
        flow_id_2 = response_dict['oml:upload_flow']['oml:id']
        flow_2 = openml.flows.flow.get_flow(flow_id_2)
        self.assertEqual(flow_1.components[0]['flow'].id,
                         flow_2.components[0]['flow'].id)

    def test__ensure_flow_exists(self):
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        flow = openml.OpenMLFlow(description='Test flow!', model=model,
                                 external_version='sklearn_' + sklearn.__version__)
        return_code, return_value = flow.publish()
        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        # Update flow name with test sentinel
        flow.name = '%s%s' % (self.sentinel, flow.name)
        flow.components[0]['oml:flow'].name = '%s%s' % (
            self.sentinel, flow.components[0]['oml:flow'].name)

        flow_1 = openml.flows.flow.get_flow(flow_id)

        # The flows are basically the same, except that they are represented
        # by two different instances in the program. Nevertheless, they have
        # the same ID on the server.
        self.assertEqual(flow._ensure_flow_exists(),
                         flow_1._ensure_flow_exists())
        self.assertEqual(flow.components[0]['oml:flow']._ensure_flow_exists(),
                         flow_1.components[0]['flow']._ensure_flow_exists())

    def test_upload_complicated_flow(self):
        digits = load_digits()
        X = digits.data
        y = digits.target
        categorical_features = list(range(X.shape[1]))

        model = Pipeline([('ohe', OneHotEncoder(
            categorical_features=categorical_features, handle_unknown='ignore')),
                          ('scaler', StandardScaler(with_mean=False)),
                          ('fu', FeatureUnion([('sp', SelectPercentile()),
                                               ('pca', TruncatedSVD(
                                                    n_components=10))])),
                          ('classifier', AdaBoostClassifier(
                              DecisionTreeClassifier()))])
        parameters = {'classifier__n_estimators':
                          openml.sklearn.stats.Discrete([1, 2, 5, 10]),
                      'classifier__base_estimator__max_depth':
                          openml.sklearn.stats.LogUniformInt(2, 0, 3)}
        cv = StratifiedKFold(n_folds=2, shuffle=True)
        rs = RandomizedSearchCV(model, parameters, cv=cv, n_iter=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X.copy(), y.copy())

        flow = openml.OpenMLFlow(description='Test flow!',
                                 model=rs,
                                 external_version='sklearn_' + sklearn.__version__)
        return_code, return_value = flow.publish()
        self.assertEqual(return_code, 200)

        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        new_flow = openml.flows.get_flow(flow_id)
        # Pipeline and random search
        self.assertEqual(len(new_flow.components), 2)
        # steps
        self.assertEqual(len(new_flow.components[0]['flow'].parameters), 3)
        self.assertEqual(len(new_flow.components[1]['flow'].parameters), 1)
        self.assertEqual(new_flow.components[1]['flow'].parameters[0]['default_value'],
                         '(("ohe", "sklearn.preprocessing.data.OneHotEncoder"), '
                         '("scaler", "sklearn.preprocessing.data.StandardScaler"), '
                         '("fu", "sklearn.pipeline.FeatureUnion"), '
                         '("classifier", "sklearn.ensemble.weight_boosting.AdaBoostClassifier"))')
        # Components of the pipeline
        self.assertEqual(len(new_flow.components[1]['flow'].components), 4)
        self.assertEqual(len(new_flow.components[1]['flow'].components[
                                 0]['flow'].parameters), 5)

        # Check that the downloaded flow can actually be used for something!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X.copy(), y.copy())

    def test_get_flow(self):
        flow = openml.flows.get_flow(1185)
        self.assertIsInstance(flow, openml.OpenMLFlow)
        self.assertEqual(flow.id, 1185)
        self.assertEqual(len(flow.parameters), 14)
        for parameter in flow.parameters:
            self.assertEqual(len(parameter), 4)
        self.assertEqual(len(flow.components), 1)
        for component in flow.components:
            self.assertEqual(len(component), 2)
            self.assertIsInstance(component['flow'], openml.OpenMLFlow)
            self.assertIsNone(component['flow'].model)

    def test__check_dependency(self):
        # simple dependency on one line
        input = 'numpy'
        fixture = {'numpy': True}
        output = openml.flows.flow._check_dependencies(input, 3)
        self.assertEqual(output, fixture)

        # all dependency operators
        current_version = np.__version__

        for operator in ['=', '==', '>=', '<=', ]:
            input = 'numpy%s%s' % (operator, current_version)
            fixture = {'numpy%s%s' % (operator, current_version): True}
            output = openml.flows.flow._check_dependencies(input, 3)
            self.assertEqual(output, fixture)

        for operator in ['>', '<', '!=']:
            input = 'numpy%s%s' % (operator, current_version)
            fixture = {'numpy%s%s' % (operator, current_version): False}
            output = openml.flows.flow._check_dependencies(input, 3)
            self.assertEqual(output, fixture)

        # two dependencies seperated by whitespace
        input = 'numpy scipy'
        fixture = {'numpy': True, 'scipy': True}
        output = openml.flows.flow._check_dependencies(input, 3)
        self.assertEqual(output, fixture)

    def test__construct_model_for_flow(self):
        flow_xml = """<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4253</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.model_selection._search.RandomizedSearchCV(openml.sklearn.model_selection.StratifiedKFold__sklearn.pipeline.Pipeline(sklearn.preprocessing.data.OneHotEncoder__sklearn.preprocessing.data.StandardScaler__sklearn.pipeline.FeatureUnion(sklearn.feature_selection.univariate_selection.SelectPercentile__sklearn.decomposition.truncated_svd.TruncatedSVD)__sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier)))</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Test flow!</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>cv</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>openml.sklearn.model_selection.StratifiedKFold</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>error_score</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>raise</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>estimator</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>sklearn.pipeline.Pipeline</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>fit_params</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value></oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>iid</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_iter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>2</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_jobs</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>1</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__cv</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__cv__n_folds</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__cv__random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__cv__shuffle</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__error_score</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__algorithm</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__class_weight</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__criterion</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__max_depth</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__max_features</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__max_leaf_nodes</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__min_samples_leaf</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__min_samples_split</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__min_weight_fraction_leaf</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__presort</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__base_estimator__splitter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__learning_rate</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__n_estimators</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__classifier__random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__n_jobs</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca__algorithm</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca__n_components</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca__n_iter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca__random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__pca__tol</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__sp</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__sp__percentile</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__sp__score_func</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__transformer_list</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__fu__transformer_weights</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe__categorical_features</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe__dtype</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe__handle_unknown</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe__n_values</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__ohe__sparse</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__scaler</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__scaler__copy</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__scaler__with_mean</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__scaler__with_std</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__estimator__steps</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__fit_params</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__iid</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__n_iter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__n_jobs</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__param_distributions</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__pre_dispatch</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__refit</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__scoring</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>parameter_distribution__verbose</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>Unparametrized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>pre_dispatch</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>2*n_jobs</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>refit</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>scoring</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>verbose</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>0</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:component>
  <oml:identifier>cv</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4254</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>openml.sklearn.model_selection.StratifiedKFold</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>n_folds</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>2</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>shuffle</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
<oml:component>
  <oml:identifier>estimator</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4255</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.pipeline.Pipeline(sklearn.preprocessing.data.OneHotEncoder__sklearn.preprocessing.data.StandardScaler__sklearn.pipeline.FeatureUnion(sklearn.feature_selection.univariate_selection.SelectPercentile__sklearn.decomposition.truncated_svd.TruncatedSVD)__sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier))</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>steps</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>((&quot;ohe&quot;, &quot;sklearn.preprocessing.data.OneHotEncoder&quot;), (&quot;scaler&quot;, &quot;sklearn.preprocessing.data.StandardScaler&quot;), (&quot;fu&quot;, &quot;sklearn.pipeline.FeatureUnion&quot;), (&quot;classifier&quot;, &quot;sklearn.ensemble.weight_boosting.AdaBoostClassifier&quot;))</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:component>
  <oml:identifier>ohe</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4256</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.preprocessing.data.OneHotEncoder</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>categorical_features</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>dtype</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>&lt;class 'numpy.float64'&gt;</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>handle_unknown</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>ignore</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_values</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>auto</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>sparse</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
<oml:component>
  <oml:identifier>scaler</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4257</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.preprocessing.data.StandardScaler</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>copy</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>with_mean</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>False</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>with_std</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>True</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
<oml:component>
  <oml:identifier>fu</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4258</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.pipeline.FeatureUnion(sklearn.feature_selection.univariate_selection.SelectPercentile__sklearn.decomposition.truncated_svd.TruncatedSVD)</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>n_jobs</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>1</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>transformer_list</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>((&quot;sp&quot;, &quot;sklearn.feature_selection.univariate_selection.SelectPercentile&quot;), (&quot;pca&quot;, &quot;sklearn.decomposition.truncated_svd.TruncatedSVD&quot;))</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>transformer_weights</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:component>
  <oml:identifier>sp</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4259</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.feature_selection.univariate_selection.SelectPercentile</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>percentile</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>10</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>score_func</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>&lt;function f_classif at 0x7fd422669e18&gt;</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
<oml:component>
  <oml:identifier>pca</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4260</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.decomposition.truncated_svd.TruncatedSVD</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>algorithm</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>randomized</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_components</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>10</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_iter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>5</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>tol</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>0.0</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
</oml:flow>
</oml:component>
<oml:component>
  <oml:identifier>classifier</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4261</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier)</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>algorithm</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>SAMME.R</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>base_estimator</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>sklearn.tree.tree.DecisionTreeClassifier</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>learning_rate</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>1.0</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>n_estimators</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>50</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:component>
  <oml:identifier>base_estimator</oml:identifier>

<oml:flow xmlns:oml="http://openml.org/openml">
  <oml:id>4262</oml:id>
<oml:uploader>86</oml:uploader>
<oml:name>sklearn.tree.tree.DecisionTreeClassifier</oml:name>
<oml:version>1</oml:version>
<oml:external_version>sklearn_0.18.dev0</oml:external_version>
<oml:description>Automatically created sub-component.</oml:description>
<oml:upload_date>2016-05-10 08:34:57</oml:upload_date>
<oml:parameter>
	<oml:name>class_weight</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>criterion</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>gini</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>max_depth</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>max_features</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>max_leaf_nodes</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>min_samples_leaf</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>1</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>min_samples_split</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>2</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>min_weight_fraction_leaf</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>0.0</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>presort</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>False</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>random_state</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>None</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
<oml:parameter>
	<oml:name>splitter</oml:name>
	<oml:data_type></oml:data_type>
	<oml:default_value>best</oml:default_value>
	<oml:description></oml:description>
</oml:parameter>
</oml:flow>
</oml:component>
</oml:flow>
</oml:component>
</oml:flow>
</oml:component>
</oml:flow>
"""

        flow_dict = xmltodict.parse(flow_xml)
        # This calls the model creation part
        flow = openml.flows.flow._create_flow_from_dict(flow_dict)
        model = flow.model
        X = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=int)
        model.fit(X, y)

        self.assertIsInstance(model, RandomizedSearchCV)
        self.assertIsInstance(model.cv, openml.sklearn.model_selection.StratifiedKFold)
        # De-serialization of patched distributions
        self.assertIsInstance(model.param_distributions, dict)
        self.assertEqual(len(model.param_distributions), 2)
        print(model.param_distributions)
        self.assertEqual(str(model.param_distributions[
                                  'classifier__base_estimator__max_depth']),
                         str(openml.sklearn.stats.LogUniformInt(2, 0, 3)))
        self.assertEqual(str(model.param_distributions[
                                'classifier__n_estimators']),
                         str(openml.sklearn.stats.Discrete([1, 2, 5, 10])))
        self.assertIsInstance(model.estimator, Pipeline)
        self.assertEqual(model.estimator.steps[0][0], 'ohe')
        ohe = model.estimator.steps[0][1]
        self.assertIsInstance(ohe, OneHotEncoder)
        # De-serialization of lists
        self.assertEqual(ohe.categorical_features, [False, False])
        # De-serialization of type
        self.assertIsInstance(ohe.dtype, type)
        self.assertEqual(ohe.dtype, np.float64)
        # Check that the component furthest down in the component tree is
        # created
        self.assertIsInstance(model.estimator.steps[-1][1].base_estimator,
                              DecisionTreeClassifier)
