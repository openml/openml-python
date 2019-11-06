# License: BSD 3-Clause

import numpy as np
import random
import os
from time import time

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from openml.testing import TestBase, SimpleImputer
import openml
import openml.extensions.sklearn

import pytest


class TestRun(TestBase):
    # Splitting not helpful, these test's don't rely on the server and take
    # less than 1 seconds

    def test_tagging(self):

        runs = openml.runs.list_runs(size=1)
        run_id = list(runs.keys())[0]
        run = openml.runs.get_run(run_id)
        tag = "testing_tag_{}_{}".format(self.id(), time())
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 0)
        run.push_tag(tag)
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 1)
        self.assertIn(run_id, run_list)
        run.remove_tag(tag)
        run_list = openml.runs.list_runs(tag=tag)
        self.assertEqual(len(run_list), 0)

    def _test_run_obj_equals(self, run, run_prime):
        for dictionary in ['evaluations', 'fold_evaluations',
                           'sample_evaluations']:
            if getattr(run, dictionary) is not None:
                self.assertDictEqual(getattr(run, dictionary),
                                     getattr(run_prime, dictionary))
            else:
                # should be none or empty
                other = getattr(run_prime, dictionary)
                if other is not None:
                    self.assertDictEqual(other, dict())
        self.assertEqual(run._to_xml(),
                         run_prime._to_xml())

        numeric_part = \
            np.array(np.array(run.data_content)[:, 0:-2], dtype=float)
        numeric_part_prime = \
            np.array(np.array(run_prime.data_content)[:, 0:-2], dtype=float)
        string_part = np.array(run.data_content)[:, -2:]
        string_part_prime = np.array(run_prime.data_content)[:, -2:]
        np.testing.assert_array_almost_equal(numeric_part, numeric_part_prime)
        np.testing.assert_array_equal(string_part, string_part_prime)

        if run.trace is not None:
            run_trace_content = run.trace.trace_to_arff()['data']
        else:
            run_trace_content = None

        if run_prime.trace is not None:
            run_prime_trace_content = run_prime.trace.trace_to_arff()['data']
        else:
            run_prime_trace_content = None

        if run_trace_content is not None:
            def _check_array(array, type_):
                for line in array:
                    for entry in line:
                        self.assertIsInstance(entry, type_)

            int_part = [line[:3] for line in run_trace_content]
            _check_array(int_part, int)
            int_part_prime = [line[:3] for line in run_prime_trace_content]
            _check_array(int_part_prime, int)

            float_part = np.array(
                np.array(run_trace_content)[:, 3:4],
                dtype=float,
            )
            float_part_prime = np.array(
                np.array(run_prime_trace_content)[:, 3:4],
                dtype=float,
            )
            bool_part = [line[4] for line in run_trace_content]
            bool_part_prime = [line[4] for line in run_prime_trace_content]
            for bp, bpp in zip(bool_part, bool_part_prime):
                self.assertIn(bp, ['true', 'false'])
                self.assertIn(bpp, ['true', 'false'])
            string_part = np.array(run_trace_content)[:, 5:]
            string_part_prime = np.array(run_prime_trace_content)[:, 5:]

            np.testing.assert_array_almost_equal(int_part, int_part_prime)
            np.testing.assert_array_almost_equal(float_part, float_part_prime)
            self.assertEqual(bool_part, bool_part_prime)
            np.testing.assert_array_equal(string_part, string_part_prime)
        else:
            self.assertIsNone(run_prime_trace_content)

    def test_to_from_filesystem_vanilla(self):

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', DecisionTreeClassifier(max_depth=1)),
        ])
        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(
            model=model,
            task=task,
            add_local_measures=False,
            avoid_duplicate_runs=False,
            upload_flow=True
        )

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        # The flow has been uploaded to server, so only the reference flow_id should be present
        self.assertTrue(run_prime.flow_id is not None)
        self.assertTrue(run_prime.flow is None)
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()
        TestBase._mark_entity_for_removal('run', run_prime.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1],
                                                            run_prime.run_id))

    @pytest.mark.flaky()
    def test_to_from_filesystem_search(self):

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', DecisionTreeClassifier(max_depth=1)),
        ])
        model = GridSearchCV(
            estimator=model,
            param_grid={
                "classifier__max_depth": [1, 2, 3, 4, 5],
                "imputer__strategy": ['mean', 'median'],
            }
        )

        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(
            model=model,
            task=task,
            add_local_measures=False,
            avoid_duplicate_runs=False,
        )

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)

        run_prime = openml.runs.OpenMLRun.from_filesystem(cache_path)
        self._test_run_obj_equals(run, run_prime)
        run_prime.publish()
        TestBase._mark_entity_for_removal('run', run_prime.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1],
                                                            run_prime.run_id))

    def test_to_from_filesystem_no_model(self):

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', DummyClassifier()),
        ])
        task = openml.tasks.get_task(119)
        run = openml.runs.run_model_on_task(
            model=model,
            task=task,
            add_local_measures=False,
        )

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path, store_model=False)
        # obtain run from filesystem
        openml.runs.OpenMLRun.from_filesystem(cache_path, expect_model=False)
        # assert default behaviour is throwing an error
        with self.assertRaises(ValueError, msg='Could not find model.pkl'):
            openml.runs.OpenMLRun.from_filesystem(cache_path)

    def test_publish_with_local_loaded_flow(self):
        """
        Publish a run tied to a local flow after it has first been saved to
         and loaded from disk.
        """
        extension = openml.extensions.sklearn.SklearnExtension()

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', DummyClassifier()),
        ])
        task = openml.tasks.get_task(119)

        # Make sure the flow does not exist on the server yet.
        flow = extension.model_to_flow(model)
        self._add_sentinel_to_flow_name(flow)
        self.assertFalse(openml.flows.flow_exists(flow.name, flow.external_version))

        run = openml.runs.run_flow_on_task(
            flow=flow,
            task=task,
            add_local_measures=False,
            avoid_duplicate_runs=False,
            upload_flow=False
        )

        # Make sure that the flow has not been uploaded as requested.
        self.assertFalse(openml.flows.flow_exists(flow.name, flow.external_version))

        cache_path = os.path.join(
            self.workdir,
            'runs',
            str(random.getrandbits(128)),
        )
        run.to_filesystem(cache_path)
        # obtain run from filesystem
        loaded_run = openml.runs.OpenMLRun.from_filesystem(cache_path)
        loaded_run.publish()
        TestBase._mark_entity_for_removal('run', loaded_run.run_id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split('/')[-1],
                                                            loaded_run.run_id))

        # make sure the flow is published as part of publishing the run.
        self.assertTrue(openml.flows.flow_exists(flow.name, flow.external_version))
        openml.runs.get_run(loaded_run.run_id)
