# License: BSD 3-Clause
from typing import Optional, List

import openml
import openml.study
from openml.testing import TestBase
import pandas as pd
import pytest


class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True

    def test_get_study_old(self):
        openml.config.server = self.production_server

        study = openml.study.get_study(34)
        self.assertEqual(len(study.data), 105)
        self.assertEqual(len(study.tasks), 105)
        self.assertEqual(len(study.flows), 27)
        self.assertEqual(len(study.setups), 30)
        self.assertIsNone(study.runs)

    def test_get_study_new(self):
        openml.config.server = self.production_server

        study = openml.study.get_study(123)
        self.assertEqual(len(study.data), 299)
        self.assertEqual(len(study.tasks), 299)
        self.assertEqual(len(study.flows), 5)
        self.assertEqual(len(study.setups), 1253)
        self.assertEqual(len(study.runs), 1693)

    def test_get_openml100(self):
        openml.config.server = self.production_server

        study = openml.study.get_study("OpenML100", "tasks")
        self.assertIsInstance(study, openml.study.OpenMLBenchmarkSuite)
        study_2 = openml.study.get_suite("OpenML100")
        self.assertIsInstance(study_2, openml.study.OpenMLBenchmarkSuite)
        self.assertEqual(study.study_id, study_2.study_id)

    def test_get_study_error(self):
        openml.config.server = self.production_server

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected entity type 'task' reported by the server, expected 'run'",
        ):
            openml.study.get_study(99)

    def test_get_suite(self):
        openml.config.server = self.production_server

        study = openml.study.get_suite(99)
        self.assertEqual(len(study.data), 72)
        self.assertEqual(len(study.tasks), 72)
        self.assertIsNone(study.flows)
        self.assertIsNone(study.runs)
        self.assertIsNone(study.setups)

    def test_get_suite_error(self):
        openml.config.server = self.production_server

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected entity type 'run' reported by the server, expected 'task'",
        ):
            openml.study.get_suite(123)

    def test_publish_benchmark_suite(self):
        fixture_alias = None
        fixture_name = "unit tested benchmark suite"
        fixture_descr = "bla"
        fixture_task_ids = [1, 2, 3]

        study = openml.study.create_benchmark_suite(
            alias=fixture_alias,
            name=fixture_name,
            description=fixture_descr,
            task_ids=fixture_task_ids,
        )
        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], study.id))

        self.assertGreater(study.id, 0)

        # verify main meta data
        study_downloaded = openml.study.get_suite(study.id)
        self.assertEqual(study_downloaded.alias, fixture_alias)
        self.assertEqual(study_downloaded.name, fixture_name)
        self.assertEqual(study_downloaded.description, fixture_descr)
        self.assertEqual(study_downloaded.main_entity_type, "task")
        # verify resources
        self.assertIsNone(study_downloaded.flows)
        self.assertIsNone(study_downloaded.setups)
        self.assertIsNone(study_downloaded.runs)
        self.assertGreater(len(study_downloaded.data), 0)
        self.assertLessEqual(len(study_downloaded.data), len(fixture_task_ids))
        self.assertSetEqual(set(study_downloaded.tasks), set(fixture_task_ids))

        # attach more tasks
        tasks_additional = [4, 5, 6]
        openml.study.attach_to_study(study.id, tasks_additional)
        study_downloaded = openml.study.get_suite(study.id)
        # verify again
        self.assertSetEqual(set(study_downloaded.tasks), set(fixture_task_ids + tasks_additional))
        # test detach function
        openml.study.detach_from_study(study.id, fixture_task_ids)
        study_downloaded = openml.study.get_suite(study.id)
        self.assertSetEqual(set(study_downloaded.tasks), set(tasks_additional))

        # test status update function
        openml.study.update_suite_status(study.id, "deactivated")
        study_downloaded = openml.study.get_suite(study.id)
        self.assertEqual(study_downloaded.status, "deactivated")
        # can't delete study, now it's not longer in preparation

    def _test_publish_empty_study_is_allowed(self, explicit: bool):
        runs: Optional[List[int]] = [] if explicit else None
        kind = "explicit" if explicit else "implicit"

        study = openml.study.create_study(
            name=f"empty-study-{kind}",
            description=f"a study with no runs attached {kind}ly",
            run_ids=runs,
        )

        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], study.id))

        self.assertGreater(study.id, 0)
        study_downloaded = openml.study.get_study(study.id)
        self.assertEqual(study_downloaded.main_entity_type, "run")
        self.assertIsNone(study_downloaded.runs)

    def test_publish_empty_study_explicit(self):
        self._test_publish_empty_study_is_allowed(explicit=True)

    def test_publish_empty_study_implicit(self):
        self._test_publish_empty_study_is_allowed(explicit=False)

    @pytest.mark.flaky()
    def test_publish_study(self):
        # get some random runs to attach
        run_list = openml.evaluations.list_evaluations("predictive_accuracy", size=10)
        self.assertEqual(len(run_list), 10)

        fixt_alias = None
        fixt_name = "unit tested study"
        fixt_descr = "bla"
        fixt_flow_ids = set([evaluation.flow_id for evaluation in run_list.values()])
        fixt_task_ids = set([evaluation.task_id for evaluation in run_list.values()])
        fixt_setup_ids = set([evaluation.setup_id for evaluation in run_list.values()])

        study = openml.study.create_study(
            alias=fixt_alias,
            benchmark_suite=None,
            name=fixt_name,
            description=fixt_descr,
            run_ids=list(run_list.keys()),
        )
        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], study.id))
        self.assertGreater(study.id, 0)
        study_downloaded = openml.study.get_study(study.id)
        self.assertEqual(study_downloaded.alias, fixt_alias)
        self.assertEqual(study_downloaded.name, fixt_name)
        self.assertEqual(study_downloaded.description, fixt_descr)
        self.assertEqual(study_downloaded.main_entity_type, "run")

        self.assertSetEqual(set(study_downloaded.runs), set(run_list.keys()))
        self.assertSetEqual(set(study_downloaded.setups), set(fixt_setup_ids))
        self.assertSetEqual(set(study_downloaded.flows), set(fixt_flow_ids))
        self.assertSetEqual(set(study_downloaded.tasks), set(fixt_task_ids))

        # test whether the list run function also handles study data fine
        run_ids = openml.runs.list_runs(study=study.id)
        self.assertSetEqual(set(run_ids), set(study_downloaded.runs))

        # test whether the list evaluation function also handles study data fine
        run_ids = openml.evaluations.list_evaluations(
            "predictive_accuracy", size=None, study=study.id
        )
        self.assertSetEqual(set(run_ids), set(study_downloaded.runs))

        # attach more runs
        run_list_additional = openml.runs.list_runs(size=10, offset=10)
        openml.study.attach_to_study(study.id, list(run_list_additional.keys()))
        study_downloaded = openml.study.get_study(study.id)
        # verify again
        all_run_ids = set(run_list_additional.keys()) | set(run_list.keys())
        self.assertSetEqual(set(study_downloaded.runs), all_run_ids)

        # test detach function
        openml.study.detach_from_study(study.id, list(run_list.keys()))
        study_downloaded = openml.study.get_study(study.id)
        self.assertSetEqual(set(study_downloaded.runs), set(run_list_additional.keys()))

        # test status update function
        openml.study.update_study_status(study.id, "deactivated")
        study_downloaded = openml.study.get_study(study.id)
        self.assertEqual(study_downloaded.status, "deactivated")

        res = openml.study.delete_study(study.id)
        self.assertTrue(res)

    def test_study_attach_illegal(self):
        run_list = openml.runs.list_runs(size=10)
        self.assertEqual(len(run_list), 10)
        run_list_more = openml.runs.list_runs(size=20)
        self.assertEqual(len(run_list_more), 20)

        study = openml.study.create_study(
            alias=None,
            benchmark_suite=None,
            name="study with illegal runs",
            description="none",
            run_ids=list(run_list.keys()),
        )
        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], study.id))
        study_original = openml.study.get_study(study.id)

        with self.assertRaisesRegex(
            openml.exceptions.OpenMLServerException, "Problem attaching entities."
        ):
            # run id does not exists
            openml.study.attach_to_study(study.id, [0])

        with self.assertRaisesRegex(
            openml.exceptions.OpenMLServerException, "Problem attaching entities."
        ):
            # some runs already attached
            openml.study.attach_to_study(study.id, list(run_list_more.keys()))
        study_downloaded = openml.study.get_study(study.id)
        self.assertListEqual(study_original.runs, study_downloaded.runs)

    def test_study_list(self):
        study_list = openml.study.list_studies(status="in_preparation", output_format="dataframe")
        # might fail if server is recently reset
        self.assertGreaterEqual(len(study_list), 2)

    def test_study_list_output_format(self):
        study_list = openml.study.list_studies(status="in_preparation", output_format="dataframe")
        self.assertIsInstance(study_list, pd.DataFrame)
