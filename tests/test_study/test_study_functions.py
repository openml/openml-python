# License: BSD 3-Clause
from __future__ import annotations

import pandas as pd
import pytest

import openml
import openml.study
from openml.testing import TestBase


class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True

    @pytest.mark.production()
    def test_get_study_old(self):
        openml.config.server = self.production_server

        study = openml.study.get_study(34)
        assert len(study.data) == 105
        assert len(study.tasks) == 105
        assert len(study.flows) == 27
        assert len(study.setups) == 30
        assert study.runs is None

    @pytest.mark.production()
    def test_get_study_new(self):
        openml.config.server = self.production_server

        study = openml.study.get_study(123)
        assert len(study.data) == 299
        assert len(study.tasks) == 299
        assert len(study.flows) == 5
        assert len(study.setups) == 1253
        assert len(study.runs) == 1693

    @pytest.mark.production()
    def test_get_openml100(self):
        openml.config.server = self.production_server

        study = openml.study.get_study("OpenML100", "tasks")
        assert isinstance(study, openml.study.OpenMLBenchmarkSuite)
        study_2 = openml.study.get_suite("OpenML100")
        assert isinstance(study_2, openml.study.OpenMLBenchmarkSuite)
        assert study.study_id == study_2.study_id

    @pytest.mark.production()
    def test_get_study_error(self):
        openml.config.server = self.production_server

        with pytest.raises(
            ValueError, match="Unexpected entity type 'task' reported by the server, expected 'run'"
        ):
            openml.study.get_study(99)

    @pytest.mark.production()
    def test_get_suite(self):
        openml.config.server = self.production_server

        study = openml.study.get_suite(99)
        assert len(study.data) == 72
        assert len(study.tasks) == 72
        assert study.flows is None
        assert study.runs is None
        assert study.setups is None

    @pytest.mark.production()
    def test_get_suite_error(self):
        openml.config.server = self.production_server

        with pytest.raises(
            ValueError, match="Unexpected entity type 'run' reported by the server, expected 'task'"
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

        assert study.id > 0

        # verify main meta data
        study_downloaded = openml.study.get_suite(study.id)
        assert study_downloaded.alias == fixture_alias
        assert study_downloaded.name == fixture_name
        assert study_downloaded.description == fixture_descr
        assert study_downloaded.main_entity_type == "task"
        # verify resources
        assert study_downloaded.flows is None
        assert study_downloaded.setups is None
        assert study_downloaded.runs is None
        assert len(study_downloaded.data) > 0
        assert len(study_downloaded.data) <= len(fixture_task_ids)
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
        assert study_downloaded.status == "deactivated"
        # can't delete study, now it's not longer in preparation

    def _test_publish_empty_study_is_allowed(self, explicit: bool):
        runs: list[int] | None = [] if explicit else None
        kind = "explicit" if explicit else "implicit"

        study = openml.study.create_study(
            name=f"empty-study-{kind}",
            description=f"a study with no runs attached {kind}ly",
            run_ids=runs,
        )

        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info("collected from {}: {}".format(__file__.split("/")[-1], study.id))

        assert study.id > 0
        study_downloaded = openml.study.get_study(study.id)
        assert study_downloaded.main_entity_type == "run"
        assert study_downloaded.runs is None

    def test_publish_empty_study_explicit(self):
        self._test_publish_empty_study_is_allowed(explicit=True)

    def test_publish_empty_study_implicit(self):
        self._test_publish_empty_study_is_allowed(explicit=False)

    @pytest.mark.flaky()
    def test_publish_study(self):
        # get some random runs to attach
        run_list = openml.evaluations.list_evaluations("predictive_accuracy", size=10)
        assert len(run_list) == 10

        fixt_alias = None
        fixt_name = "unit tested study"
        fixt_descr = "bla"
        fixt_flow_ids = {evaluation.flow_id for evaluation in run_list.values()}
        fixt_task_ids = {evaluation.task_id for evaluation in run_list.values()}
        fixt_setup_ids = {evaluation.setup_id for evaluation in run_list.values()}

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
        assert study.id > 0
        study_downloaded = openml.study.get_study(study.id)
        assert study_downloaded.alias == fixt_alias
        assert study_downloaded.name == fixt_name
        assert study_downloaded.description == fixt_descr
        assert study_downloaded.main_entity_type == "run"

        self.assertSetEqual(set(study_downloaded.runs), set(run_list.keys()))
        self.assertSetEqual(set(study_downloaded.setups), set(fixt_setup_ids))
        self.assertSetEqual(set(study_downloaded.flows), set(fixt_flow_ids))
        self.assertSetEqual(set(study_downloaded.tasks), set(fixt_task_ids))

        # test whether the list run function also handles study data fine
        run_ids = openml.runs.list_runs(study=study.id)
        self.assertSetEqual(set(run_ids), set(study_downloaded.runs))

        # test whether the list evaluation function also handles study data fine
        run_ids = openml.evaluations.list_evaluations(
            "predictive_accuracy",
            size=None,
            study=study.id,
        )
        self.assertSetEqual(set(run_ids), set(study_downloaded.runs))

        # attach more runs, since we fetch 11 here, at least one is non-overlapping
        run_list_additional = openml.runs.list_runs(size=11, offset=10)
        run_list_additional = set(run_list_additional) - set(run_ids)
        openml.study.attach_to_study(study.id, list(run_list_additional))
        study_downloaded = openml.study.get_study(study.id)
        # verify again
        all_run_ids = run_list_additional | set(run_list.keys())
        self.assertSetEqual(set(study_downloaded.runs), all_run_ids)

        # test detach function
        openml.study.detach_from_study(study.id, list(run_list.keys()))
        study_downloaded = openml.study.get_study(study.id)
        self.assertSetEqual(set(study_downloaded.runs), run_list_additional)

        # test status update function
        openml.study.update_study_status(study.id, "deactivated")
        study_downloaded = openml.study.get_study(study.id)
        assert study_downloaded.status == "deactivated"

        res = openml.study.delete_study(study.id)
        assert res

    def test_study_attach_illegal(self):
        run_list = openml.runs.list_runs(size=10)
        assert len(run_list) == 10
        run_list_more = openml.runs.list_runs(size=20)
        assert len(run_list_more) == 20

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

        with pytest.raises(
            openml.exceptions.OpenMLServerException, match="Problem attaching entities."
        ):
            # run id does not exists
            openml.study.attach_to_study(study.id, [0])

        with pytest.raises(
            openml.exceptions.OpenMLServerException, match="Problem attaching entities."
        ):
            # some runs already attached
            openml.study.attach_to_study(study.id, list(run_list_more.keys()))
        study_downloaded = openml.study.get_study(study.id)
        self.assertListEqual(study_original.runs, study_downloaded.runs)

    def test_study_list(self):
        study_list = openml.study.list_studies(status="in_preparation", output_format="dataframe")
        # might fail if server is recently reset
        assert len(study_list) >= 2

    def test_study_list_output_format(self):
        study_list = openml.study.list_studies(status="in_preparation", output_format="dataframe")
        assert isinstance(study_list, pd.DataFrame)
