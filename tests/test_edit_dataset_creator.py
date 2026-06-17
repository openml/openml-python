import openml
import pandas as pd

def test_edit_dataset_updates_creator(monkeypatch):
    # Use a mock dataset and mock API to avoid real uploads

    df = pd.DataFrame({"x": [1,2,3], "y": [4,5,6]})
    ds = openml.datasets.create_dataset(
        name="edit_test",
        description="test",
        creator="First Creator",
        contributor=None,
        collection_date="2024",
        language="English",
        licence="CC0",
        attributes="auto",
        data=df,
        default_target_attribute="y",
        ignore_attribute=None,
        citation="N/A",
    )

    # monkeypatch publish so it returns fake id
    class Dummy:
        dataset_id = 999999

    monkeypatch.setattr(ds, "publish", lambda: Dummy())

    # monkeypatch edit_dataset to capture outgoing XML
    captured = {}

    def fake_api_call(call, method, data, file_elements):
        captured["xml"] = file_elements["edit_parameters"][1]
        return "<oml:data_edit><oml:id>999999</oml:id></oml:data_edit>"

    monkeypatch.setattr(openml._api_calls, "_perform_api_call", fake_api_call)

    # run edit
    openml.datasets.edit_dataset(data_id=999999, creator="Updated Creator")

    assert "<oml:creator>Updated Creator</oml:creator>" in captured["xml"]
