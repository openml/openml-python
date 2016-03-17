import unittest
from sklearn.linear_model import LogisticRegression
import openml
from openml.testing import TestBase


class TestRun(TestBase):
    def test_run_iris(self):
        task = openml.tasks.download_task(self.connector, 10107)
        clf = LogisticRegression()
        run = openml.runs.openml_run(self.connector, task, clf)
        return_code, return_value = run.publish(self.connector)
        self.assertEqual(return_code, 200)
        # self.assertTrue("This is a read-only account" in return_value)

    ############################################################################
    # Runs
    @unittest.skip('The method which is tested by this function doesnt exist')
    def test_download_run_list(self):
        def check_run(run):
            self.assertIsInstance(run, dict)
            self.assertEqual(len(run), 6)

        runs = self.connector.get_runs_list(task_id=1)
        self.assertGreaterEqual(len(runs), 800)
        for run in runs:
            check_run(run)

        runs = self.connector.get_runs_list(flow_id=1)
        self.assertGreaterEqual(len(runs), 1)
        for run in runs:
            check_run(run)

        runs = self.connector.get_runs_list(setup_id=1)
        self.assertGreaterEqual(len(runs), 260)
        for run in runs:
            check_run(run)

    def test_download_run(self):
        run = openml.runs.download_run(self.connector, 473350)
        self.assertEqual(run.dataset_id, 1167)
        self.assertEqual(run.evaluations['f_measure'], 0.624668)
