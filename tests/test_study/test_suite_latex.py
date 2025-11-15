# License: BSD 3-Clause
import pandas as pd
import pytest
from types import SimpleNamespace
from pathlib import Path

import openml
from openml.exceptions import OpenMLServerException
from openml.study.study import OpenMLBenchmarkSuite


@pytest.mark.server()
def test_suite_metadata_and_latex_server(tmp_path):
    """
    Integration test using a stable public suite (OpenML-CC18 = 99).
    If the current server (e.g. test server) does not have suite 99, skip.
    """
    try:
        suite = openml.study.get_suite(99)  # may raise if suite absent on this server
    except OpenMLServerException:
        pytest.skip("Suite 99 not available on this server; skipping integration test.")

    df = suite.metadata
    assert isinstance(df, pd.DataFrame)
    assert "did" in df.columns
    assert "tid" in df.columns
    assert "name" in df.columns

    out = tmp_path / "suite.tex"
    latex = suite.to_latex(str(out), caption="Test caption", label="tab:test")
    assert out.exists()
    assert isinstance(latex, str)
    assert len(latex) > 0
    assert "\\caption{Test caption}" in latex or "Test caption" in latex

def test_suite_metadata_and_latex_mocked(monkeypatch, tmp_path):
    """
    Unit test without network: construct an OpenMLBenchmarkSuite and patch functions
    used by the metadata property (openml.tasks.get_task and openml.datasets.list_datasets).
    """
    suite = OpenMLBenchmarkSuite(
        suite_id=999999,
        alias=None,
        name="Fake Suite",
        description="",
        status=None,
        creation_date=None,
        creator=None,
        tags=None,
        data=None,
        tasks=[101, 102],
    )

    class DummyTask:
        def __init__(self, tid, dataset_id):
            self.id = tid
            self.dataset_id = dataset_id

    def fake_get_task(tid, download_data=False, download_qualities=False):
        mapping = {101: 11, 102: 22}
        return DummyTask(tid, mapping[tid])

    monkeypatch.setattr(openml.tasks, "get_task", fake_get_task)

    fake_df = pd.DataFrame(
        {
            "did": [11, 22],
            "name": ["Dataset A", "Dataset B"],
            "NumberOfInstances": [100, 200],
            "NumberOfFeatures": [5, 10],
            "NumberOfClasses": [2, 3],
        }
    )

    monkeypatch.setattr(openml.datasets, "list_datasets", lambda data_id, **kwargs: fake_df)

    md = suite.metadata
    assert isinstance(md, pd.DataFrame)
    assert set(md["did"].tolist()) == {11, 22}
    # tid mapping preserved
    assert md.loc[md["did"] == 11, "tid"].iloc[0] == 101

    out = tmp_path / "fake-suite.tex"
    latex = suite.to_latex(str(out), caption="Fake", label="tab:fake")
    assert out.exists()
    assert isinstance(latex, str)
    assert "\\caption{Fake}" in latex or "Fake" in latex