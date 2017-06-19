import unittest
import openml
import openml.evaluations
from openml.testing import TestBase

class TestEvaluationFunctions(TestBase):

    def test_evaluation_list_filter_task(self):
        openml.config.server = self.production_server

        task_id = 7312

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", task_id=[task_id])

        self.assertGreater(len(evaluations), 100)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].task_id, task_id)


    def test_evaluation_list_filter_uploader(self):
        openml.config.server = self.production_server

        uploader_id = 16

        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", uploader=[uploader_id])

        self.assertGreater(len(evaluations), 100)
        for run_id in evaluations.keys():
            self.assertEquals(evaluations[run_id].uploader, uploader_id)
