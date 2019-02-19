import openml
import openml.study
from openml.testing import TestBase

class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True

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
