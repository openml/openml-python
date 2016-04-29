import unittest

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
