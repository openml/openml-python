import unittest
import openml
import openml.evaluations
from openml.testing import TestBase

class TestEvaluationFunctions(TestBase):

    def test_evaluation_list(self):
        openml.config.server = self.production_server

        res = openml.evaluations.list_evaluations("predictive_accuracy", 59)

        self.assertGreater(len(res), 100)

