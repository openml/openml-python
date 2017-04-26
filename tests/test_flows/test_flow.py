import collections
import copy
import hashlib
import re
import sys
import time

if sys.version_info[0] >= 3:
    from unittest import mock
else:
    import mock

import scipy.stats
import sklearn
import sklearn.datasets
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.naive_bayes
import sklearn.tree
import xmltodict

from openml.testing import TestBase
from openml._api_calls import _perform_api_call
import openml
from openml.flows.sklearn_converter import _format_external_version


class TestFlow(TestBase):


    def test_get_flow(self):
        # We need to use the production server here because 4024 is not the test
        # server
        openml.config.server = self.production_server

        flow = openml.flows.get_flow(4024)
        self.assertIsInstance(flow, openml.OpenMLFlow)
        self.assertEqual(flow.flow_id, 4024)
        self.assertEqual(len(flow.parameters), 24)
        self.assertEqual(len(flow.components), 1)

        subflow_1 = list(flow.components.values())[0]
        self.assertIsInstance(subflow_1, openml.OpenMLFlow)
        self.assertEqual(subflow_1.flow_id, 4025)
        self.assertEqual(len(subflow_1.parameters), 14)
        self.assertEqual(subflow_1.parameters['E'], 'CC')
        self.assertEqual(len(subflow_1.components), 1)

        subflow_2 = list(subflow_1.components.values())[0]
        self.assertIsInstance(subflow_2, openml.OpenMLFlow)
        self.assertEqual(subflow_2.flow_id, 4026)
        self.assertEqual(len(subflow_2.parameters), 13)
        self.assertEqual(subflow_2.parameters['I'], '10')
        self.assertEqual(len(subflow_2.components), 1)

        subflow_3 = list(subflow_2.components.values())[0]
        self.assertIsInstance(subflow_3, openml.OpenMLFlow)
        self.assertEqual(subflow_3.flow_id, 1724)
        self.assertEqual(len(subflow_3.parameters), 11)
        self.assertEqual(subflow_3.parameters['L'], '-1')
        self.assertEqual(len(subflow_3.components), 0)

    def test_from_xml_to_xml(self):
        # Get the raw xml thing
        # TODO maybe get this via get_flow(), which would have to be refactored to allow getting only the xml dictionary
        # TODO: no sklearn flows.
        for flow_id in [3, 5, 7, 9, ]:
            flow_xml = _perform_api_call("flow/%d" % flow_id)
            flow_dict = xmltodict.parse(flow_xml)

            flow = openml.OpenMLFlow._from_dict(flow_dict)
            new_xml = flow._to_xml()

            flow_xml = flow_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            flow_xml = re.sub(r'^$', '', flow_xml)
            new_xml = new_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            new_xml = re.sub(r'^$', '', new_xml)

            self.assertEqual(new_xml, flow_xml)

    def test_to_xml_from_xml(self):
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier())
        model = sklearn.pipeline.Pipeline(steps=(
            ('scaler', scaler), ('boosting', boosting)))
        flow = openml.flows.sklearn_to_flow(model)
        flow.flow_id = -234
        # end of setup

        xml = flow._to_xml()
        xml_dict = xmltodict.parse(xml)
        new_flow = openml.flows.OpenMLFlow._from_dict(xml_dict)

        # Would raise exception if they are not legal
        openml.flows.functions.check_flows_equal(new_flow, flow)
        self.assertIsNot(new_flow, flow)

    def test_publish_flow(self):
        flow = openml.OpenMLFlow(name='sklearn.dummy.DummyClassifier',
                                 class_name='sklearn.dummy.DummyClassifier',
                                 description="test description",
                                 model=sklearn.dummy.DummyClassifier(),
                                 components=collections.OrderedDict(),
                                 parameters=collections.OrderedDict(),
                                 parameters_meta_info=collections.OrderedDict(),
                                 external_version=_format_external_version(
                                     'sklearn', sklearn.__version__),
                                 tags=[],
                                 language='English',
                                 dependencies=None)

        flow, _ = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        self.assertIsInstance(flow.flow_id, int)

    def test_semi_legal_flow(self):
        # TODO: Test if parameters are set correctly!
        # should not throw error as it contains two differentiable forms of Bagging
        # i.e., Bagging(Bagging(J48)) and Bagging(J48)
        semi_legal = sklearn.ensemble.BaggingClassifier(
            base_estimator=sklearn.ensemble.BaggingClassifier(
                base_estimator=sklearn.tree.DecisionTreeClassifier()))
        flow = openml.flows.sklearn_to_flow(semi_legal)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()

    @mock.patch('openml.flows.functions.get_flow')
    @mock.patch('openml.flows.flow._perform_api_call')
    def test_publish_error(self, api_call_mock, get_flow_mock):
        model = sklearn.ensemble.RandomForestClassifier()
        flow = openml.flows.sklearn_to_flow(model)
        api_call_mock.return_value = "<oml:upload_flow>\n" \
                                     "    <oml:id>1</oml:id>\n" \
                                     "</oml:upload_flow>"
        get_flow_mock.return_value = flow

        flow.publish()
        self.assertEqual(api_call_mock.call_count, 1)
        self.assertEqual(get_flow_mock.call_count, 1)

        flow_copy = copy.deepcopy(flow)
        flow_copy.name = flow_copy.name[:-1]
        get_flow_mock.return_value = flow_copy

        with self.assertRaises(ValueError) as context_manager:
            flow.publish()

        fixture = "Flow was not stored correctly on the server. " \
                  "New flow ID is 1. Please check manually and remove " \
                  "the flow if necessary! Error is:\n" \
                  "'Flow sklearn.ensemble.forest.RandomForestClassifier: values for attribute 'name' differ: " \
                  "'sklearn.ensemble.forest.RandomForestClassifier' vs 'sklearn.ensemble.forest.RandomForestClassifie'.'"

        self.assertEqual(context_manager.exception.args[0], fixture)
        self.assertEqual(api_call_mock.call_count, 2)
        self.assertEqual(get_flow_mock.call_count, 2)

    def test_illegal_flow(self):
        # should throw error as it contains two imputers
        illegal = sklearn.pipeline.Pipeline(steps=[('imputer1', sklearn.preprocessing.Imputer()),
                                                   ('imputer2', sklearn.preprocessing.Imputer()),
                                                   ('classif', sklearn.tree.DecisionTreeClassifier())])
        self.assertRaises(ValueError, openml.flows.sklearn_to_flow, illegal)

    def test_nonexisting_flow_exists(self):
        def get_sentinel():
            # Create a unique prefix for the flow. Necessary because the flow is
            # identified by its name and external version online. Having a unique
            #  name allows us to publish the same flow in each test run
            md5 = hashlib.md5()
            md5.update(str(time.time()).encode('utf-8'))
            sentinel = md5.hexdigest()[:10]
            sentinel = 'TEST%s' % sentinel
            return sentinel

        name = get_sentinel() + get_sentinel()
        version = get_sentinel()

        flow_id = openml.flows.flow_exists(name, version)
        self.assertFalse(flow_id)

    def test_existing_flow_exists(self):
        # create a flow
        nb = sklearn.naive_bayes.GaussianNB()
        flow = openml.flows.sklearn_to_flow(nb)
        flow, _ = self._add_sentinel_to_flow_name(flow, None)
        #publish the flow
        flow = flow.publish()
        #redownload the flow
        flow = openml.flows.get_flow(flow.flow_id)

        # check if flow exists can find it
        flow = openml.flows.get_flow(flow.flow_id)
        downloaded_flow_id = openml.flows.flow_exists(flow.name, flow.external_version)
        self.assertEquals(downloaded_flow_id, flow.flow_id)


    def test_sklearn_to_upload_to_flow(self):
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target

        # Test a more complicated flow
        ohe = sklearn.preprocessing.OneHotEncoder(categorical_features=[1])
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        pca = sklearn.decomposition.TruncatedSVD()
        fs = sklearn.feature_selection.SelectPercentile(
            score_func=sklearn.feature_selection.f_classif, percentile=30)
        fu = sklearn.pipeline.FeatureUnion(transformer_list=[
            ('pca', pca), ('fs', fs)])
        boosting = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier())
        model = sklearn.pipeline.Pipeline(steps=[('ohe', ohe), ('scaler', scaler),
                                                 ('fu', fu), ('boosting', boosting)])
        parameter_grid = {'boosting__n_estimators': [1, 5, 10, 100],
                          'boosting__learning_rate': scipy.stats.uniform(0.01, 0.99),
                          'boosting__base_estimator__max_depth': scipy.stats.randint(1, 10)}
        cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
        rs = sklearn.model_selection.RandomizedSearchCV(
            estimator=model, param_distributions=parameter_grid, cv=cv)
        rs.fit(X, y)
        flow = openml.flows.sklearn_to_flow(rs)
        flow.tags.extend(['openml-python', 'unittest'])
        flow, sentinel = self._add_sentinel_to_flow_name(flow, None)

        flow.publish()
        self.assertIsInstance(flow.flow_id, int)

        # Check whether we can load the flow again
        # Remove the sentinel from the name again so that we can reinstantiate
        # the object again
        new_flow = openml.flows.get_flow(flow_id=flow.flow_id)

        local_xml = flow._to_xml()
        server_xml = new_flow._to_xml()

        local_xml = re.sub('<oml:id>[0-9]+</oml:id>', '', local_xml)
        server_xml = re.sub('<oml:id>[0-9]+</oml:id>', '', server_xml)
        server_xml = re.sub('<oml:uploader>[0-9]+</oml:uploader>', '', server_xml)
        server_xml = re.sub('<oml:version>[0-9]+</oml:version>', '', server_xml)
        server_xml = re.sub('<oml:upload_date>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}</oml:upload_date>', '', server_xml)

        for i in range(10):
            # Make sure that we replace all occurences of two newlines
            local_xml = local_xml.replace(sentinel, '')
            local_xml = local_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            local_xml = re.sub(r'(^$)', '', local_xml)
            server_xml = server_xml.replace(sentinel, '')
            server_xml = server_xml.replace('  ', '').replace('\t', '').strip().replace('\n\n', '\n').replace('&quot;', '"')
            server_xml = re.sub(r'^$', '', server_xml)

        self.assertEqual(server_xml, local_xml)

        # Would raise exception if they are not equal!
        openml.flows.functions.check_flows_equal(new_flow, flow)
        self.assertIsNot(new_flow, flow)

        fixture_name = '%ssklearn.model_selection._search.RandomizedSearchCV(' \
                       'estimator=sklearn.pipeline.Pipeline(' \
                       'ohe=sklearn.preprocessing.data.OneHotEncoder,' \
                       'scaler=sklearn.preprocessing.data.StandardScaler,' \
                       'fu=sklearn.pipeline.FeatureUnion(' \
                       'pca=sklearn.decomposition.truncated_svd.TruncatedSVD,' \
                       'fs=sklearn.feature_selection.univariate_selection.SelectPercentile),' \
                       'boosting=sklearn.ensemble.weight_boosting.AdaBoostClassifier(' \
                       'base_estimator=sklearn.tree.tree.DecisionTreeClassifier)))' \
                        % sentinel

        self.assertEqual(new_flow.name, fixture_name)
        self.assertTrue('openml-python' in new_flow.tags)
        self.assertTrue('unittest' in new_flow.tags)
        new_flow.model.fit(X, y)
