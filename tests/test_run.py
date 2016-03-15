import unittest
from openml import APIConnector, OpenMLRun
from sklearn.linear_model import LogisticRegression


class TestRun(unittest.TestCase):
    def test_run_iris(self):
        connector = APIConnector()
        task = connector.download_task(10107)
        clf = LogisticRegression()
        run = OpenMLRun.openml_run(connector, task, clf)
        return_code, dataset_xml = run.publish(connector)
        print(dataset_xml)
        self.assertEqual(return_code, 200)
