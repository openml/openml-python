import openml
import openml.evaluations
from openml.testing import TestBase

class TestEvaluationFunctions(TestBase):

    def test_evaluation_list(self):
        openml.config.server = self.production_server

        task_id = 7312

        res = openml.evaluations.list_evaluations("predictive_accuracy", task_id)

        self.assertGreater(len(res), 100)

