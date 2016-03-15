from openml import APIConnector, OpenMLRun
from sklearn.linear_model import LogisticRegression


def test_run_iris():
    connector = APIConnector()
    task = connector.download_task(10107)
    clf = LogisticRegression()
    run = OpenMLRun.openml_run(connector, task, clf)
    run.upload(connector)
