# License: BSD 3-Clause
from __future__ import annotations

import pytest
import unittest

import openml
import openml.study
from openml.testing import TestBase

from unittest import mock
import requests

class StudyMockServer:
    """Helper class to encapsulate all mock XML generation and response mocking."""

    @staticmethod
    def make_response(xml: str | Exception):
        if isinstance(xml, Exception):
            return xml
        r = mock.Mock(spec=["status_code", "text", "headers"])
        r.status_code = 200
        r.headers = {}
        r.text = xml
        return r

    @staticmethod
    def build_study_xml(
        study_id, n_data, n_tasks, n_flows, n_setups, n_runs=None,
        main_entity_type="run", alias=None, status="active",
        name="Dummy Study", description="desc", run_ids=None,
        task_ids=None, visibility="public"
    ):
        def repeat(tag, n):
            return "".join(f"<oml:{tag}_id>{i}</oml:{tag}_id>" for i in range(1, n+1))
        
        runs_block = ""
        if run_ids is not None:
            runs_block = f"<oml:runs>" + "".join(f"<oml:run_id>{rid}</oml:run_id>" for rid in run_ids) + f"</oml:runs>"
        elif n_runs is not None:
            runs_block = f"<oml:runs>{repeat('run', n_runs)}</oml:runs>"

        tasks_block = "".join(f"<oml:task_id>{tid}</oml:task_id>" for tid in task_ids) if task_ids else repeat("task", n_tasks)
        alias_block = f"<oml:alias>{alias}</oml:alias>" if alias else ""

        linked_entities_block = ""
        if main_entity_type == "task":
            tids = task_ids if task_ids else range(1, n_tasks + 1)
            linked_entities = "".join(f"<oml:entity><oml:id>{tid}</oml:id><oml:type>task</oml:type></oml:entity>" for tid in tids)
            linked_entities_block = f"<oml:linked_entities>{linked_entities}</oml:linked_entities>"

        return f"""
        <oml:study>
            <oml:id>{study_id}</oml:id>
            {alias_block}
            <oml:main_entity_type>{main_entity_type}</oml:main_entity_type>
            <oml:name>{name}</oml:name>
            <oml:description>{description}</oml:description>
            <oml:visibility>{visibility}</oml:visibility>
            <oml:status>{status}</oml:status>
            <oml:creation_date>2020-01-01</oml:creation_date>
            <oml:creator>tester</oml:creator>
            <oml:data>{repeat("data", n_data)}</oml:data>
            {linked_entities_block}
            <oml:tasks>{tasks_block}</oml:tasks>
            <oml:flows>{repeat("flow", n_flows)}</oml:flows>
            <oml:setups>{repeat("setup", n_setups)}</oml:setups>
            {runs_block}
        </oml:study>
        """

    @staticmethod
    def build_study_upload_xml(study_id: int):
        return f'<oml:study_upload xmlns:oml="http://openml.org/openml"><oml:id>{study_id}</oml:id></oml:study_upload>'

    @staticmethod
    def build_study_attach_xml(study_id: int, n_entities: int):
        return f'<oml:study_attach xmlns:oml="http://openml.org/openml"><oml:id>{study_id}</oml:id><oml:main_entity_type>run</oml:main_entity_type><oml:linked_entities>{n_entities}</oml:linked_entities></oml:study_attach>'

    @staticmethod
    def build_study_detach_xml(study_id: int, n_entities: int):
        return f'<oml:study_detach xmlns:oml="http://openml.org/openml"><oml:id>{study_id}</oml:id><oml:main_entity_type>run</oml:main_entity_type><oml:linked_entities>{n_entities}</oml:linked_entities></oml:study_detach>'

    @staticmethod
    def build_status_update_xml(study_id: int, status: str):
        return f'<oml:study_status_update xmlns:oml="http://openml.org/openml"><oml:id>{study_id}</oml:id><oml:status>{status}</oml:status></oml:study_status_update>'

    @staticmethod
    def build_delete_xml(study_id: int):
        return f'<oml:study_delete xmlns:oml="http://openml.org/openml"><oml:id>{study_id}</oml:id></oml:study_delete>'

    @staticmethod
    def build_evaluations_xml(run_ids):
        evaluations = "".join(f"<oml:evaluation><oml:run_id>{rid}</oml:run_id><oml:task_id>{rid}</oml:task_id><oml:setup_id>{rid}</oml:setup_id><oml:flow_id>{rid}</oml:flow_id><oml:flow_name>dummy</oml:flow_name><oml:data_id>1</oml:data_id><oml:data_name>dummy</oml:data_name><oml:function>predictive_accuracy</oml:function><oml:upload_time>2020-01-01</oml:upload_time><oml:uploader>3229</oml:uploader><oml:value>0.5</oml:value></oml:evaluation>" for rid in run_ids)
        return f'<oml:evaluations xmlns:oml="http://openml.org/openml">{evaluations}</oml:evaluations>'

    @staticmethod
    def build_users_xml():
        return '<oml:users xmlns:oml="http://openml.org/openml"><oml:user><oml:id>3229</oml:id><oml:username>tester</oml:username></oml:user></oml:users>'

    @staticmethod
    def build_runs_xml(run_ids):
        runs = "".join(f"<oml:run><oml:run_id>{rid}</oml:run_id><oml:task_id>{rid}</oml:task_id><oml:task_type_id>1</oml:task_type_id><oml:setup_id>{rid}</oml:setup_id><oml:flow_id>{rid}</oml:flow_id><oml:uploader>3229</oml:uploader><oml:upload_time>2020-01-01</oml:upload_time><oml:error_message></oml:error_message><oml:run_details></oml:run_details></oml:run>" for rid in run_ids)
        return f'<oml:runs xmlns:oml="http://openml.org/openml">{runs}</oml:runs>'

    @staticmethod
    def setup_publish_benchmark_suite_mocks(mock_get, mock_post, fixture_name, fixture_descr):
        mock_post.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_study_upload_xml(146)),
            StudyMockServer.make_response(StudyMockServer.build_study_attach_xml(146, 6)),
            StudyMockServer.make_response(StudyMockServer.build_study_detach_xml(146, 3)),
            StudyMockServer.make_response(StudyMockServer.build_status_update_xml(146, "deactivated")),
        ]
        mock_get.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=146, name=fixture_name, description=fixture_descr, main_entity_type="task", task_ids=[1,2,3], n_data=1, n_tasks=3, n_flows=0, n_setups=0, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=146, name=fixture_name, description=fixture_descr, main_entity_type="task", task_ids=[1,2,3,4,5,6], n_data=1, n_tasks=6, n_flows=0, n_setups=0, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=146, name=fixture_name, description=fixture_descr, main_entity_type="task", task_ids=[4,5,6], n_data=1, n_tasks=3, n_flows=0, n_setups=0, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=146, name=fixture_name, description=fixture_descr, main_entity_type="task", task_ids=[4,5,6], n_data=1, n_tasks=3, n_flows=0, n_setups=0, status="deactivated")),
        ]

    @staticmethod
    def setup_publish_study_mocks(mock_get, mock_post, mock_delete, fixt_name, fixt_descr):
        mock_get.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_evaluations_xml(range(1,11))),
            StudyMockServer.make_response(StudyMockServer.build_users_xml()),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(name=fixt_name, description=fixt_descr, study_id=157, main_entity_type="run", n_data=1, n_tasks=10, n_flows=10, n_setups=10, n_runs=10, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_runs_xml(range(1,11))),
            StudyMockServer.make_response(StudyMockServer.build_evaluations_xml(range(1,11))),
            StudyMockServer.make_response(StudyMockServer.build_users_xml()),
            StudyMockServer.make_response(StudyMockServer.build_runs_xml(range(11,22))),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=157, name=fixt_name, description=fixt_descr, main_entity_type="run", n_data=1, n_tasks=20, n_flows=20, n_setups=20, n_runs=21, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=157, name=fixt_name, description=fixt_descr, main_entity_type="run", n_data=1, n_tasks=10, n_flows=10, n_setups=10, run_ids=range(11, 22), status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=157, name=fixt_name, description=fixt_descr, main_entity_type="run", n_data=1, n_tasks=10, n_flows=10, n_setups=10, run_ids=range(11, 22), status="deactivated")),
        ]
        mock_post.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_study_upload_xml(157)),
            StudyMockServer.make_response(StudyMockServer.build_study_attach_xml(157, 21)),
            StudyMockServer.make_response(StudyMockServer.build_study_detach_xml(157, 11)),
            StudyMockServer.make_response(StudyMockServer.build_status_update_xml(157, "deactivated")),
        ]
        mock_delete.return_value = StudyMockServer.make_response(StudyMockServer.build_delete_xml(157))

    @staticmethod
    def setup_study_attach_illegal_mocks(mock_get, mock_post, mock_delete):
        mock_get.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_runs_xml(range(1,11))),
            StudyMockServer.make_response(StudyMockServer.build_runs_xml(range(1,21))),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=300, name="study with illegal runs", description="none", main_entity_type="run", n_data=1, n_tasks=10, n_flows=10, n_setups=10, n_runs=10, status="in_preparation")),
            StudyMockServer.make_response(StudyMockServer.build_study_xml(study_id=300, name="study with illegal runs", description="none", main_entity_type="run", n_data=1, n_tasks=10, n_flows=10, n_setups=10, n_runs=10, status="in_preparation")),
        ]
        mock_post.side_effect = [
            StudyMockServer.make_response(StudyMockServer.build_study_upload_xml(300)),
            openml.exceptions.OpenMLServerException("Problem attaching entities."),
            openml.exceptions.OpenMLServerException("Problem attaching entities."),
        ]
        mock_delete.return_value = StudyMockServer.make_response(StudyMockServer.build_delete_xml(300))


class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True

    @pytest.mark.production_server()
    @pytest.mark.xfail(reason="failures_issue_1544", strict=False)
    @mock.patch.object(requests.Session, "get")
    def test_get_study_old(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=34, n_data=105, n_tasks=105, n_flows=27, n_setups=30, n_runs=None)
        )

        study = openml.study.get_study(34)
        assert len(study.data) == 105
        assert len(study.tasks) == 105
        assert len(study.flows) == 27
        assert len(study.setups) == 30
        assert study.runs is None

    @pytest.mark.production_server()
    @mock.patch.object(requests.Session, "get")
    def test_get_study_new(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=123, n_data=299, n_tasks=299, n_flows=5, n_setups=1253, n_runs=1693)
        )
        study = openml.study.get_study(123)
        assert len(study.data) == 299
        assert len(study.tasks) == 299
        assert len(study.flows) == 5
        assert len(study.setups) == 1253
        assert len(study.runs) == 1693

    @pytest.mark.production_server()
    @mock.patch.object(requests.Session, "get")
    def test_get_openml100(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=99, alias="OpenML100", n_data=100, n_tasks=100, n_flows=0, n_setups=0, n_runs=None, main_entity_type="task")
        )
        study = openml.study.get_study("OpenML100", "tasks")
        assert isinstance(study, openml.study.OpenMLBenchmarkSuite)
        study_2 = openml.study.get_suite("OpenML100")
        assert isinstance(study_2, openml.study.OpenMLBenchmarkSuite)
        assert study.study_id == study_2.study_id

    @pytest.mark.production_server()
    @mock.patch.object(requests.Session, "get")
    def test_get_study_error(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=99, n_data=1, n_tasks=1, n_flows=0, n_setups=0, n_runs=None, main_entity_type="task")
        )

        with pytest.raises(
            ValueError, match="Unexpected entity type 'task' reported by the server, expected 'run'"
        ):
            openml.study.get_study(99)

    @pytest.mark.production_server()
    @mock.patch.object(requests.Session, "get")
    def test_get_suite(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=99, n_data=72, n_tasks=72, n_flows=0, n_setups=0, n_runs=None, main_entity_type="task")
        )

        study = openml.study.get_suite(99)
        assert len(study.data) == 72
        assert len(study.tasks) == 72
        assert study.flows is None
        assert study.runs is None
        assert study.setups is None

    @pytest.mark.production_server()
    @mock.patch.object(requests.Session, "get")
    def test_get_suite_error(self, mock_get):
        mock_get.return_value = StudyMockServer.make_response(
            StudyMockServer.build_study_xml(study_id=123, n_data=1, n_tasks=1, n_flows=0, n_setups=0, n_runs=None, main_entity_type="run")
        )

        with pytest.raises(
            ValueError, match="Unexpected entity type 'run' reported by the server, expected 'task'"
        ):
            openml.study.get_suite(123)

    @pytest.mark.test_server()
    @mock.patch.object(requests.Session, "post")
    @mock.patch.object(requests.Session, "get")
    def test_publish_benchmark_suite(self, mock_get, mock_post):
        fixture_alias = None
        fixture_name = "unit tested benchmark suite"
        fixture_descr = "bla"
        fixture_task_ids = [1, 2, 3]
        
        StudyMockServer.setup_publish_benchmark_suite_mocks(mock_get, mock_post, fixture_name, fixture_descr)
        study = openml.study.create_benchmark_suite(
            alias=fixture_alias,
            name=fixture_name,
            description=fixture_descr,
            task_ids=fixture_task_ids,
        )
        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {study.id}")

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
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {study.id}")

        assert study.id > 0
        study_downloaded = openml.study.get_study(study.id)
        assert study_downloaded.main_entity_type == "run"
        assert study_downloaded.runs is None

    @pytest.mark.test_server()
    @mock.patch.object(requests.Session, "post")
    @mock.patch.object(requests.Session, "get")
    def test_publish_empty_study_explicit(self, mock_get, mock_post):
        mock_post.side_effect = [StudyMockServer.make_response(StudyMockServer.build_study_upload_xml(200))]
        empty_study_xml = StudyMockServer.build_study_xml(study_id=200, name="empty-study-explicit", description="a study with no runs attached explicitly", main_entity_type="run", task_ids=None, n_data=0, n_tasks=0, n_flows=0, n_setups=0, n_runs=None, status="in_preparation")
        mock_get.side_effect = [StudyMockServer.make_response(empty_study_xml), StudyMockServer.make_response(empty_study_xml)]
        self._test_publish_empty_study_is_allowed(explicit=True)

    @pytest.mark.test_server()
    @mock.patch.object(requests.Session, "post")
    @mock.patch.object(requests.Session, "get")
    def test_publish_empty_study_implicit(self, mock_get, mock_post):
        mock_post.side_effect = [StudyMockServer.make_response(StudyMockServer.build_study_upload_xml(200))]
        empty_study_xml = StudyMockServer.build_study_xml(study_id=200, name="empty-study-implicit", description="a study with no runs attached implicitly", main_entity_type="run", task_ids=None, n_data=0, n_tasks=0, n_flows=0, n_setups=0, n_runs=None, status="in_preparation")
        mock_get.side_effect = [StudyMockServer.make_response(empty_study_xml), StudyMockServer.make_response(empty_study_xml)]
        
        self._test_publish_empty_study_is_allowed(explicit=False)

    @pytest.mark.flaky()
    @pytest.mark.test_server()
    @mock.patch.object(requests.Session, "delete")
    @mock.patch.object(requests.Session, "post")
    @mock.patch.object(requests.Session, "get")
    def test_publish_study(self, mock_get, mock_post, mock_delete):
        fixt_alias = None
        fixt_name = "unit tested study"
        fixt_descr = "bla"

        StudyMockServer.setup_publish_study_mocks(mock_get, mock_post, mock_delete, fixt_name, fixt_descr)

        # get some random runs to attach
        run_list = openml.evaluations.list_evaluations("predictive_accuracy", size=10)
        assert len(run_list) == 10

        
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
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {study.id}")
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
        run_ids = openml.runs.list_runs(study=study.id) # returns DF
        self.assertSetEqual(set(run_ids["run_id"]), set(study_downloaded.runs))

        # test whether the list evaluation function also handles study data fine
        run_ids = openml.evaluations.list_evaluations( # returns list of objects
            "predictive_accuracy",
            size=None,
            study=study.id,
            output_format="object", # making the default explicit
        )
        self.assertSetEqual(set(run_ids), set(study_downloaded.runs))

        # attach more runs, since we fetch 11 here, at least one is non-overlapping
        run_list_additional = openml.runs.list_runs(size=11, offset=10)
        run_list_additional = set(run_list_additional["run_id"]) - set(run_ids)
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

    @pytest.mark.test_server()
    @mock.patch.object(requests.Session, "delete")
    @mock.patch.object(requests.Session, "post")
    @mock.patch.object(requests.Session, "get")
    def test_study_attach_illegal(self, mock_get, mock_post, mock_delete):

        StudyMockServer.setup_study_attach_illegal_mocks(mock_get, mock_post, mock_delete)

        run_list = openml.runs.list_runs(size=10)
        assert len(run_list) == 10
        run_list_more = openml.runs.list_runs(size=20)
        assert len(run_list_more) == 20

        study = openml.study.create_study(
            alias=None,
            benchmark_suite=None,
            name="study with illegal runs",
            description="none",
            run_ids=list(run_list["run_id"]),
        )
        study.publish()
        TestBase._mark_entity_for_removal("study", study.id)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {study.id}")
        study_original = openml.study.get_study(study.id)

        with pytest.raises(
            openml.exceptions.OpenMLServerException,
            match="Problem attaching entities.",
        ):
            # run id does not exists
            openml.study.attach_to_study(study.id, [0])

        with pytest.raises(
            openml.exceptions.OpenMLServerException,
            match="Problem attaching entities.",
        ):
            # some runs already attached
            openml.study.attach_to_study(study.id, list(run_list_more["run_id"]))
        study_downloaded = openml.study.get_study(study.id)
        self.assertListEqual(study_original.runs, study_downloaded.runs)

    @unittest.skip("It is unclear when we can expect the test to pass or fail.")
    @mock.patch.object(requests.Session, "get")
    def test_study_list(self, mock_get):
        studies_xml = """
        <oml:study_list xmlns:oml="http://openml.org/openml">
            <oml:study><oml:id>1</oml:id><oml:name>study-one</oml:name><oml:status>in_preparation</oml:status></oml:study>
            <oml:study><oml:id>2</oml:id><oml:name>study-two</oml:name><oml:status>in_preparation</oml:status></oml:study>
        </oml:study_list>
        """
        mock_get.return_value = StudyMockServer.make_response(studies_xml)
        study_list = openml.study.list_studies(status="in_preparation")
        # might fail if server is recently reset
        assert len(study_list) >= 2
