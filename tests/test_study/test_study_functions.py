import openml
import openml.study
from openml.testing import TestBase

class TestStudyFunctions(TestBase):
    _multiprocess_can_split_ = True

    def test_get_study(self):
        openml.config.server = self.production_server

        study_id = 34

        study = openml.study.get_study(study_id)
        self.assertEquals(len(study.data), 105)
        self.assertEquals(len(study.tasks), 105)
        self.assertEquals(len(study.flows), 27)
        self.assertEquals(len(study.setups), 30)

    def test_get_tasks(self):
        study_id = 14

        study = openml.study.get_study(study_id, 'tasks')
        self.assertEquals(study.data, None)
        self.assertGreater(len(study.tasks), 0)
        self.assertEquals(study.flows, None)
        self.assertEquals(study.setups, None)