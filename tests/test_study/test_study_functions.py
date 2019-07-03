import openml
import openml.study
from openml.testing import TestBase, cleanup_fixture
import pandas as pd


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

        study = openml.study.get_study('OpenML100', 'tasks')
        self.assertIsInstance(study, openml.study.OpenMLBenchmarkSuite)
        study_2 = openml.study.get_suite('OpenML100')
        self.assertIsInstance(study_2, openml.study.OpenMLBenchmarkSuite)
        self.assertEqual(study.id, study_2.id)

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
        fixture_name = 'unit tested benchmark suite'
        fixture_descr = 'bla'
        fixture_task_ids = [1, 2, 3]

        study = openml.study.create_benchmark_suite(
            alias=fixture_alias,
            name=fixture_name,
            description=fixture_descr,
            task_ids=fixture_task_ids
        )
        study_id = study.publish()
        self._track_test_server_dumps('study', study_id)
        print("\ntest_study_functions: {}".format(study_id))

        self.assertGreater(study_id, 0)

        # verify main meta data
        study_downloaded = openml.study.get_suite(study_id)
        self.assertEqual(study_downloaded.alias, fixture_alias)
        self.assertEqual(study_downloaded.name, fixture_name)
        self.assertEqual(study_downloaded.description, fixture_descr)
        self.assertEqual(study_downloaded.main_entity_type, 'task')
        # verify resources
        self.assertIsNone(study_downloaded.flows)
        self.assertIsNone(study_downloaded.setups)
        self.assertIsNone(study_downloaded.runs)
        self.assertGreater(len(study_downloaded.data), 0)
        self.assertLessEqual(len(study_downloaded.data), len(fixture_task_ids))
        self.assertSetEqual(set(study_downloaded.tasks), set(fixture_task_ids))

        # attach more tasks
        tasks_additional = [4, 5, 6]
        openml.study.attach_to_study(study_id, tasks_additional)
        study_downloaded = openml.study.get_suite(study_id)
        # verify again
        self.assertSetEqual(set(study_downloaded.tasks),
                            set(fixture_task_ids + tasks_additional))
        # test detach function
        openml.study.detach_from_study(study_id, fixture_task_ids)
        study_downloaded = openml.study.get_suite(study_id)
        self.assertSetEqual(set(study_downloaded.tasks),
                            set(tasks_additional))

        # test status update function
        openml.study.update_suite_status(study_id, 'deactivated')
        study_downloaded = openml.study.get_suite(study_id)
        self.assertEqual(study_downloaded.status, 'deactivated')
        # can't delete study, now it's not longer in preparation

    def test_publish_study(self):
        # get some random runs to attach
        run_list = openml.runs.list_runs(size=10)
        self.assertEqual(len(run_list), 10)

        fixt_alias = None
        fixt_name = 'unit tested study'
        fixt_descr = 'bla'
        fixt_flow_ids = set([run['flow_id'] for run in run_list.values()])
        fixt_task_ids = set([run['task_id'] for run in run_list.values()])
        fixt_setup_ids = set([run['setup_id']for run in run_list.values()])

        study = openml.study.create_study(
            alias=fixt_alias,
            benchmark_suite=None,
            name=fixt_name,
            description=fixt_descr,
            run_ids=list(run_list.keys())
        )
        study_id = study.publish()
        self.assertGreater(study_id, 0)
        study_downloaded = openml.study.get_study(study_id)
        self.assertEqual(study_downloaded.alias, fixt_alias)
        self.assertEqual(study_downloaded.name, fixt_name)
        self.assertEqual(study_downloaded.description, fixt_descr)
        self.assertEqual(study_downloaded.main_entity_type, 'run')

        self.assertSetEqual(set(study_downloaded.runs), set(run_list.keys()))
        self.assertSetEqual(set(study_downloaded.setups), set(fixt_setup_ids))
        self.assertSetEqual(set(study_downloaded.flows), set(fixt_flow_ids))
        self.assertSetEqual(set(study_downloaded.tasks), set(fixt_task_ids))

        # attach more runs
        run_list_additional = openml.runs.list_runs(size=10, offset=10)
        openml.study.attach_to_study(study_id,
                                     list(run_list_additional.keys()))
        study_downloaded = openml.study.get_study(study_id)
        # verify again
        all_run_ids = set(run_list_additional.keys()) | set(run_list.keys())
        self.assertSetEqual(set(study_downloaded.runs), all_run_ids)

        # test detach function
        openml.study.detach_from_study(study_id, list(run_list.keys()))
        study_downloaded = openml.study.get_study(study_id)
        self.assertSetEqual(set(study_downloaded.runs),
                            set(run_list_additional.keys()))

        # test status update function
        openml.study.update_study_status(study_id, 'deactivated')
        study_downloaded = openml.study.get_study(study_id)
        self.assertEqual(study_downloaded.status, 'deactivated')

        res = openml.study.delete_study(study_id)
        self.assertTrue(res)

    def test_study_attach_illegal(self):
        run_list = openml.runs.list_runs(size=10)
        self.assertEqual(len(run_list), 10)
        run_list_more = openml.runs.list_runs(size=20)
        self.assertEqual(len(run_list_more), 20)

        study = openml.study.create_study(
            alias=None,
            benchmark_suite=None,
            name='study with illegal runs',
            description='none',
            run_ids=list(run_list.keys())
        )
        study_id = study.publish()
        study_original = openml.study.get_study(study_id)

        with self.assertRaisesRegex(openml.exceptions.OpenMLServerException,
                                    'Problem attaching entities.'):
            # run id does not exists
            openml.study.attach_to_study(study_id, [0])

        with self.assertRaisesRegex(openml.exceptions.OpenMLServerException,
                                    'Problem attaching entities.'):
            # some runs already attached
            openml.study.attach_to_study(study_id, list(run_list_more.keys()))
        study_downloaded = openml.study.get_study(study_id)
        self.assertListEqual(study_original.runs, study_downloaded.runs)

    def test_study_list(self):
        study_list = openml.study.list_studies(status='in_preparation')
        # might fail if server is recently resetted
        self.assertGreater(len(study_list), 2)

    def test_study_list_output_format(self):
        study_list = openml.study.list_studies(status='in_preparation',
                                               output_format='dataframe')
        self.assertIsInstance(study_list, pd.DataFrame)
        self.assertGreater(len(study_list), 2)
