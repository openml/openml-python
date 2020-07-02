# License: BSD 3-Clause

import hashlib
import time
import unittest.mock

import openml
import openml.exceptions
import openml.extensions.sklearn
from openml.testing import TestBase
from typing import Dict
import pandas as pd

import sklearn.tree
import sklearn.naive_bayes
import sklearn.base


def get_sentinel():
    # Create a unique prefix for the flow. Necessary because the flow is
    # identified by its name and external version online. Having a unique
    #  name allows us to publish the same flow in each test run
    md5 = hashlib.md5()
    md5.update(str(time.time()).encode("utf-8"))
    sentinel = md5.hexdigest()[:10]
    sentinel = "TEST%s" % sentinel
    return sentinel


class TestSetupFunctions(TestBase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.extension = openml.extensions.sklearn.SklearnExtension()
        super().setUp()

    def test_nonexisting_setup_exists(self):
        # first publish a non-existing flow
        sentinel = get_sentinel()
        # because of the sentinel, we can not use flows that contain subflows
        dectree = sklearn.tree.DecisionTreeClassifier()
        flow = self.extension.model_to_flow(dectree)
        flow.name = "TEST%s%s" % (sentinel, flow.name)
        flow.publish()
        TestBase._mark_entity_for_removal("flow", (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))

        # although the flow exists (created as of previous statement),
        # we can be sure there are no setups (yet) as it was just created
        # and hasn't been ran
        setup_id = openml.setups.setup_exists(flow)
        self.assertFalse(setup_id)

    def _existing_setup_exists(self, classif):

        flow = self.extension.model_to_flow(classif)
        flow.name = "TEST%s%s" % (get_sentinel(), flow.name)
        flow.publish()
        TestBase._mark_entity_for_removal("flow", (flow.flow_id, flow.name))
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], flow.flow_id))

        # although the flow exists, we can be sure there are no
        # setups (yet) as it hasn't been ran
        setup_id = openml.setups.setup_exists(flow)
        self.assertFalse(setup_id)
        setup_id = openml.setups.setup_exists(flow)
        self.assertFalse(setup_id)

        # now run the flow on an easy task:
        task = openml.tasks.get_task(115)  # diabetes
        run = openml.runs.run_flow_on_task(flow, task)
        # spoof flow id, otherwise the sentinel is ignored
        run.flow_id = flow.flow_id
        run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], run.run_id))
        # download the run, as it contains the right setup id
        run = openml.runs.get_run(run.run_id)

        # execute the function we are interested in
        setup_id = openml.setups.setup_exists(flow)
        self.assertEqual(setup_id, run.setup_id)

    def test_existing_setup_exists_1(self):
        def side_effect(self):
            self.var_smoothing = 1e-9
            self.priors = None

        with unittest.mock.patch.object(
            sklearn.naive_bayes.GaussianNB, "__init__", side_effect,
        ):
            # Check a flow with zero hyperparameters
            nb = sklearn.naive_bayes.GaussianNB()
            self._existing_setup_exists(nb)

    def test_exisiting_setup_exists_2(self):
        # Check a flow with one hyperparameter
        self._existing_setup_exists(sklearn.naive_bayes.GaussianNB())

    def test_existing_setup_exists_3(self):
        # Check a flow with many hyperparameters
        self._existing_setup_exists(
            sklearn.tree.DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=3,
                # Not setting the random state will make this flow fail as running it
                # will add a random random_state.
                random_state=1,
            )
        )

    def test_get_setup(self):
        # no setups in default test server
        openml.config.server = "https://www.openml.org/api/v1/xml/"

        # contains all special cases, 0 params, 1 param, n params.
        # Non scikitlearn flows.
        setups = [18, 19, 20, 118]
        num_params = [8, 0, 3, 1]

        for idx in range(len(setups)):
            current = openml.setups.get_setup(setups[idx])
            assert current.flow_id > 0
            if num_params[idx] == 0:
                self.assertIsNone(current.parameters)
            else:
                self.assertEqual(len(current.parameters), num_params[idx])

    def test_setup_list_filter_flow(self):
        openml.config.server = self.production_server

        flow_id = 5873

        setups = openml.setups.list_setups(flow=flow_id)

        self.assertGreater(len(setups), 0)  # TODO: please adjust 0
        for setup_id in setups.keys():
            self.assertEqual(setups[setup_id].flow_id, flow_id)

    def test_list_setups_empty(self):
        setups = openml.setups.list_setups(setup=[0])
        if len(setups) > 0:
            raise ValueError("UnitTest Outdated, got somehow results")

        self.assertIsInstance(setups, dict)

    def test_list_setups_output_format(self):
        openml.config.server = self.production_server
        flow_id = 6794
        setups = openml.setups.list_setups(flow=flow_id, output_format="object", size=10)
        self.assertIsInstance(setups, Dict)
        self.assertIsInstance(setups[list(setups.keys())[0]], openml.setups.setup.OpenMLSetup)
        self.assertEqual(len(setups), 10)

        setups = openml.setups.list_setups(flow=flow_id, output_format="dataframe", size=10)
        self.assertIsInstance(setups, pd.DataFrame)
        self.assertEqual(len(setups), 10)

        setups = openml.setups.list_setups(flow=flow_id, output_format="dict", size=10)
        self.assertIsInstance(setups, Dict)
        self.assertIsInstance(setups[list(setups.keys())[0]], Dict)
        self.assertEqual(len(setups), 10)

    def test_setuplist_offset(self):
        # TODO: remove after pull on live for better testing
        # openml.config.server = self.production_server

        size = 10
        setups = openml.setups.list_setups(offset=0, size=size)
        self.assertEqual(len(setups), size)
        setups2 = openml.setups.list_setups(offset=size, size=size)
        self.assertEqual(len(setups2), size)

        all = set(setups.keys()).union(setups2.keys())

        self.assertEqual(len(all), size * 2)

    def test_get_cached_setup(self):
        openml.config.cache_directory = self.static_cache_dir
        openml.setups.functions._get_cached_setup(1)

    def test_get_uncached_setup(self):
        openml.config.cache_directory = self.static_cache_dir
        with self.assertRaises(openml.exceptions.OpenMLCacheException):
            openml.setups.functions._get_cached_setup(10)
