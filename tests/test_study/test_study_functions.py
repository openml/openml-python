import openml
import openml.study
import unittest
from openml.testing import TestBase

class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True
    
    @unittest.skip('Production server does not yet return knowledge types (This line should not be merged in develop)')
    def test_get_study(self):
        openml.config.server = self.production_server

        study_id = 34

        study = openml.study.get_study(study_id)
        self.assertEqual(len(study.data), 105)
        self.assertEqual(len(study.tasks), 105)
        self.assertEqual(len(study.flows), 27)
        self.assertEqual(len(study.setups), 30)

    def test_get_tasks(self):
        study_id = 14

        study = openml.study.get_study(study_id, 'tasks')
        self.assertEqual(study.data, None)
        self.assertGreater(len(study.tasks), 0)
        self.assertEqual(study.flows, None)
        self.assertEqual(study.setups, None)
    
    def test_publish_benchmark_suite(self):
        fixture_alias = None
        fixture_name = 'unit tested study'
        fixture_descr = 'bla'
        fixture_task_ids = [1, 2, 3]
        
        study = openml.study.benchmark_suite_create(
            alias=fixture_alias,
            name=fixture_name,
            description=fixture_descr,
            task_ids=fixture_task_ids
        )
        study_id = study.publish()
        self.assertGreater(study_id, 0)
        
        study_downloaded = openml.study.get_study(study_id)
        self.assertEqual(study_downloaded.alias, fixture_alias)
        self.assertEqual(study_downloaded.name, fixture_name)
        self.assertEqual(study_downloaded.description, fixture_descr)
        self.assertEqual(study_downloaded.flows, None)
        self.assertEqual(study_downloaded.setups, None)
        self.assertEqual(study_downloaded.runs, None)
        self.assertGreater(len(study_downloaded.data), 0)
        self.assertLessEqual(len(study_downloaded.data), len(fixture_task_ids))
        self.assertEqual(study_downloaded.tasks, fixture_task_ids)
