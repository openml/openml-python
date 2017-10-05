import openml
import openml.evaluations
from openml.testing import TestBase

class TestEvaluationFunctions(TestBase):
    _multiprocess_can_split_ = True

    def test_evaluation_list_filter_task(self):
        openml.config.server = self.production_server

        task_id = 7312

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", task=[task_id])

        self.assertGreater(len(evaluations), 100)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].task_id, task_id)

    def test_evaluation_list_filter_uploader_ID_16(self):
        openml.config.server = self.production_server

        uploader_id = 16

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", uploader=[uploader_id])

        self.assertGreater(len(evaluations), 100)

    def test_evaluation_list_filter_uploader_ID_10(self):
        openml.config.server = self.production_server

        setup_id = 10

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", setup=[setup_id])

        self.assertGreater(len(evaluations), 100)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].setup_id, setup_id)

    def test_evaluation_list_filter_flow(self):
        openml.config.server = self.production_server

        flow_id = 100

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", flow=[flow_id])

        self.assertGreater(len(evaluations), 2)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].flow_id, flow_id)

    def test_evaluation_list_filter_run(self):
        openml.config.server = self.production_server

        run_id = 1

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", id=[run_id])

        self.assertEquals(len(evaluations), 1)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].run_id, run_id)

    def test_evaluation_list_limit(self):
        openml.config.server = self.production_server

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", size=100, offset=100)
        self.assertEquals(len(evaluations), 100)
