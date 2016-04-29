import unittest

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import RandomizedSearchCV
import xmltodict

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

        self.assertEqual(new_flow.name, flow.name)
        self.assertEqual(new_flow.description, flow.description)
        self.assertIn('sklearn', new_flow.dependencies)
        self.assertIn('numpy', new_flow.dependencies)
        self.assertIn('scipy', new_flow.dependencies)

    def test_upload_flow_with_exchangable_component(self):
        # The classifier inside AdaBoost can be changed, therefore we
        # register the classifier inside AdaBoost if it is not registered yet!
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

        flow = openml.OpenMLFlow(description='Test flow!', model=model)
        return_code, return_value = flow.publish()

        self.assertEqual(return_code, 200)
        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        new_flow = openml.flows.get_flow(flow_id)
        print(new_flow.parameters, new_flow.components)
        self.assertEqual(len(new_flow.parameters), 4)
        for parameter in new_flow.parameters:
            self.assertIn(parameter['name'], AdaBoostClassifier().get_params())
        self.assertEqual(len(new_flow.components), 1)
        self.assertEqual(new_flow.components[0]['identifier'], 'base_estimator')
        component = new_flow.components[0]['flow']
        self.assertGreaterEqual(len(component.parameters), 9)
        print(component.parameters)
        for parameter in component.parameters:
            self.assertIn(parameter['name'], DecisionTreeClassifier().get_params())

    def test_upload_complicated_flow(self):
        model = Pipeline((('scaler', StandardScaler()),
                          ('fu', FeatureUnion((('poly', PolynomialFeatures()),
                                               ('pca', PCA())))),
                          ('classifier', AdaBoostClassifier(
                              DecisionTreeClassifier()))))
        parameters = {'classifier__n_estimators': [50, 100, 500, 1000]}
        gridsearch = RandomizedSearchCV(model, parameters)

        flow = openml.OpenMLFlow(description='Test flow!',
                                 model=gridsearch)
        return_code, return_value = flow.publish()
        self.assertEqual(return_code, 200)

        response_dict = xmltodict.parse(return_value)
        flow_id = response_dict['oml:upload_flow']['oml:id']
        new_flow = openml.flows.get_flow(flow_id)
        # Pipeline
        self.assertEqual(len(new_flow.components), 1)
        # steps
        self.assertEqual(len(new_flow.components[0]['flow'].parameters), 1)
        self.assertEqual(new_flow.components[0]['flow'].parameters[0]['default_value'],
                         "((scaler, sklearn.preprocessing.data.StandardScaler), "
                         "(fu, sklearn.pipeline.FeatureUnion), "
                         "(classifier, sklearn.ensemble.weight_boosting.AdaBoostClassifier))")
        # Components of the pipeline
        self.assertEqual(len(new_flow.components[0]['flow'].components), 3)

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
    <oml:description>Test flow!</oml:description>
    <oml:id>-1</oml:id>
    <oml:version>1</oml:version>
    <oml:upload_date>00.00.00</oml:upload_date>
    <oml:uploader>-1</oml:uploader>
    <oml:parameter>
        <oml:name>cv</oml:name>
    </oml:parameter>
    <oml:parameter>
        <oml:name>error_score</oml:name>
        <oml:default_value>raise</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>fit_params</oml:name>
    </oml:parameter>
    <oml:parameter>
        <oml:name>iid</oml:name>
        <oml:default_value>True</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>n_iter</oml:name>
        <oml:default_value>10</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>n_jobs</oml:name>
        <oml:default_value>1</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>param_distributions</oml:name>
        <oml:default_value>{'classifier__n_estimators': [50, 100, 500, 1000]}</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>pre_dispatch</oml:name>
        <oml:default_value>2*n_jobs</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>random_state</oml:name>
    </oml:parameter>
    <oml:parameter>
        <oml:name>refit</oml:name>
        <oml:default_value>True</oml:default_value>
    </oml:parameter>
    <oml:parameter>
        <oml:name>scoring</oml:name>
    </oml:parameter>
    <oml:parameter>
        <oml:name>verbose</oml:name>
    </oml:parameter>
    <oml:component>
        <oml:identifier>estimator</oml:identifier>
        <oml:flow xmlns:oml="http://openml.org/openml">
            <oml:id>-1</oml:id>
            <oml:version>1</oml:version>
            <oml:upload_date>00.00.00</oml:upload_date>
            <oml:uploader>-1</oml:uploader>
            <oml:description>Automatically created sub-component.</oml:description>
            <oml:parameter>
                <oml:name>steps</oml:name>
                <oml:default_value>(("scaler", "sklearn.preprocessing.data.StandardScaler"), ("fu", "sklearn.pipeline.FeatureUnion"), ("classifier", "sklearn.ensemble.weight_boosting.AdaBoostClassifier"))</oml:default_value>
            </oml:parameter>
            <oml:component>
                <oml:identifier>scaler</oml:identifier>
                <oml:flow xmlns:oml="http://openml.org/openml">
                    <oml:id>-1</oml:id>
                    <oml:version>1</oml:version>
                    <oml:upload_date>00.00.00</oml:upload_date>
                    <oml:uploader>-1</oml:uploader>
                    <oml:description>Automatically created sub-component.</oml:description>
                    <oml:parameter>
                        <oml:name>copy</oml:name>
                        <oml:default_value>True</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>with_mean</oml:name>
                        <oml:default_value>True</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>with_std</oml:name>
                        <oml:default_value>True</oml:default_value>
                    </oml:parameter>
                    <oml:name>sklearn.preprocessing.data.StandardScaler</oml:name>
                </oml:flow>
            </oml:component>
            <oml:component>
                <oml:identifier>fu</oml:identifier>
                <oml:flow xmlns:oml="http://openml.org/openml">
                    <oml:id>-1</oml:id>
                    <oml:version>1</oml:version>
                    <oml:upload_date>00.00.00</oml:upload_date>
                    <oml:uploader>-1</oml:uploader>
                    <oml:description>Automatically created sub-component.</oml:description>
                    <oml:parameter>
                        <oml:name>n_jobs</oml:name>
                        <oml:default_value>1</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>transformer_list</oml:name>
                        <oml:default_value>(("poly", "sklearn.preprocessing.data.PolynomialFeatures"), ("pca", "sklearn.decomposition.pca.PCA"))</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>transformer_weights</oml:name>
                    </oml:parameter>
                    <oml:component>
                        <oml:identifier>poly</oml:identifier>
                        <oml:flow xmlns:oml="http://openml.org/openml">
                            <oml:id>-1</oml:id>
                    <oml:version>1</oml:version>
                    <oml:upload_date>00.00.00</oml:upload_date>
                    <oml:uploader>-1</oml:uploader>
                            <oml:description>Automatically created sub-component.</oml:description>
                            <oml:parameter>
                                <oml:name>degree</oml:name>
                                <oml:default_value>2</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>include_bias</oml:name>
                                <oml:default_value>True</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>interaction_only</oml:name>
                            </oml:parameter>
                            <oml:name>sklearn.preprocessing.data.PolynomialFeatures</oml:name>
                        </oml:flow>
                    </oml:component>
                    <oml:component>
                        <oml:identifier>pca</oml:identifier>
                        <oml:flow xmlns:oml="http://openml.org/openml">
                            <oml:id>-1</oml:id>
                            <oml:version>1</oml:version>
                            <oml:upload_date>00.00.00</oml:upload_date>
                            <oml:uploader>-1</oml:uploader>
                            <oml:description>Automatically created sub-component.</oml:description>
                            <oml:parameter>
                                <oml:name>copy</oml:name>
                                <oml:default_value>True</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>n_components</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>whiten</oml:name>
                            </oml:parameter>
                            <oml:name>sklearn.decomposition.pca.PCA</oml:name>
                        </oml:flow>
                    </oml:component>
                    <oml:name>sklearn.pipeline.FeatureUnion(sklearn.preprocessing.data.PolynomialFeatures,sklearn.decomposition.pca.PCA)</oml:name>
                </oml:flow>
            </oml:component>
            <oml:component>
                <oml:identifier>classifier</oml:identifier>
                <oml:flow xmlns:oml="http://openml.org/openml">
                    <oml:id>-1</oml:id>
                    <oml:version>1</oml:version>
                    <oml:upload_date>00.00.00</oml:upload_date>
                    <oml:uploader>-1</oml:uploader>
                    <oml:description>Automatically created sub-component.</oml:description>
                    <oml:parameter>
                        <oml:name>algorithm</oml:name>
                        <oml:default_value>SAMME.R</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>learning_rate</oml:name>
                        <oml:default_value>1.0</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>n_estimators</oml:name>
                        <oml:default_value>50</oml:default_value>
                    </oml:parameter>
                    <oml:parameter>
                        <oml:name>random_state</oml:name>
                    </oml:parameter>
                    <oml:component>
                        <oml:identifier>base_estimator</oml:identifier>
                        <oml:flow xmlns:oml="http://openml.org/openml">
                            <oml:id>-1</oml:id>
                            <oml:version>1</oml:version>
                            <oml:upload_date>00.00.00</oml:upload_date>
                            <oml:uploader>-1</oml:uploader>
                            <oml:description>Automatically created sub-component.</oml:description>
                            <oml:parameter>
                                <oml:name>class_weight</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>criterion</oml:name>
                                <oml:default_value>gini</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>max_depth</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>max_features</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>max_leaf_nodes</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>min_samples_leaf</oml:name>
                                <oml:default_value>1</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>min_samples_split</oml:name>
                                <oml:default_value>2</oml:default_value>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>min_weight_fraction_leaf</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>presort</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>random_state</oml:name>
                            </oml:parameter>
                            <oml:parameter>
                                <oml:name>splitter</oml:name>
                                <oml:default_value>best</oml:default_value>
                            </oml:parameter>
                            <oml:name>sklearn.tree.tree.DecisionTreeClassifier</oml:name>
                        </oml:flow>
                    </oml:component>
                    <oml:name>sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier)</oml:name>
                </oml:flow>
            </oml:component>
            <oml:name>sklearn.pipeline.Pipeline(sklearn.preprocessing.data.StandardScaler,sklearn.pipeline.FeatureUnion(sklearn.preprocessing.data.PolynomialFeatures,sklearn.decomposition.pca.PCA),sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier))</oml:name>
        </oml:flow>
    </oml:component>
    <oml:name>sklearn.grid_search.RandomizedSearchCV(sklearn.pipeline.Pipeline(sklearn.preprocessing.data.StandardScaler,sklearn.pipeline.FeatureUnion(sklearn.preprocessing.data.PolynomialFeatures,sklearn.decomposition.pca.PCA),sklearn.ensemble.weight_boosting.AdaBoostClassifier(sklearn.tree.tree.DecisionTreeClassifier)))</oml:name>
</oml:flow>"""

        flow_dict = xmltodict.parse(flow_xml)
        # This calls the model creation part
        flow = openml.flows.flow._create_flow_from_dict(flow_dict)
        self.assertIsInstance(flow.model, RandomizedSearchCV)