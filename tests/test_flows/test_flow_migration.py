# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd
import pytest
import requests

import openml
from openml.exceptions import OpenMLCacheException
from openml.flows import OpenMLFlow
from openml.flows import functions as flow_functions


@pytest.fixture()
def dummy_flow() -> OpenMLFlow:
    return OpenMLFlow(
        name="TestFlow",
        description="test",
        model=None,
        components=OrderedDict(),
        parameters=OrderedDict(),
        parameters_meta_info=OrderedDict(),
        external_version="1",
        tags=[],
        language="English",
        dependencies="",
        class_name="x",
    )


def test_flow_exists_delegates_to_backend(monkeypatch):
    from openml._api import api_context

    calls: dict[str, Any] = {}

    def fake_exists(name: str, external_version: str) -> int:
        calls["args"] = (name, external_version)
        return 42

    monkeypatch.setattr(api_context.backend.flows, "exists", fake_exists)

    result = openml.flows.flow_exists(name="foo", external_version="v1")

    assert result == 42
    assert calls["args"] == ("foo", "v1")


def test_list_flows_delegates_to_backend(monkeypatch):
    from openml._api import api_context

    calls: list[tuple[int, int, str | None, str | None]] = []
    df = pd.DataFrame({
        "id": [1, 2],
        "full_name": ["a", "b"],
        "name": ["a", "b"],
        "version": ["1", "1"],
        "external_version": ["v1", "v1"],
        "uploader": ["u", "u"],
    }).set_index("id")

    def fake_list_page(limit: int | None, offset: int | None, tag: str | None, uploader: str | None):
        calls.append((limit or 0, offset or 0, tag, uploader))
        return df

    monkeypatch.setattr(api_context.backend.flows, "list_page", fake_list_page)

    result = openml.flows.list_flows(offset=0, size=5, tag="t", uploader="u")

    assert result.equals(df)
    # _list_all passes batch_size as limit; expect one call
    assert calls == [(5, 0, "t", "u")]


def test_get_flow_description_fetches_and_caches(monkeypatch, tmp_path, dummy_flow):
    from openml._api import api_context

    # Force cache miss
    def raise_cache(_fid: int) -> None:
        raise OpenMLCacheException("no cache")

    monkeypatch.setattr(flow_functions, "_get_cached_flow", raise_cache)

    def fake_cache_dir(_key: str, id_: int):
        path = tmp_path / str(id_)
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(openml.utils, "_create_cache_directory_for_id", fake_cache_dir)

    xml_text = "<oml:flow>test</oml:flow>"
    response = requests.Response()
    response.status_code = 200
    response._content = xml_text.encode()

    def fake_get(flow_id: int, *, return_response: bool = False):
        if return_response:
            return dummy_flow, response
        return dummy_flow

    monkeypatch.setattr(api_context.backend.flows, "get", fake_get)

    flow = flow_functions._get_flow_description(123)

    assert flow is dummy_flow
    cached = (tmp_path / "123" / "flow.xml").read_text()
    assert cached == xml_text
    cached = (tmp_path / "123" / "flow.xml").read_text()
    assert cached == xml_text


def test_delete_flow_delegates_to_backend(monkeypatch):
    from openml._api import api_context

    calls: dict[str, Any] = {}

    def fake_delete(flow_id: int) -> None:
        calls["flow_id"] = flow_id

    monkeypatch.setattr(api_context.backend.flows, "delete", fake_delete)

    result = openml.flows.delete_flow(flow_id=999)

    assert result is True
    assert calls["flow_id"] == 999


def test_v2_flow_exists_found(monkeypatch):
    """Test FlowsV2.exists() when flow is found."""
    from openml._api.resources.flows import FlowsV2
    from openml._api.http.client import HTTPClient
    from openml._api.config import settings

    http_client = HTTPClient(settings.api.v2)
    flows_v2 = FlowsV2(http_client)

    # Mock HTTP response
    mock_response = requests.Response()
    mock_response.status_code = 200
    mock_response._content = b'{"flow_id": 123}'

    def fake_get(path: str):
        assert path == "flows/exists/weka.ZeroR/Weka_3.9.0/"
        return mock_response

    monkeypatch.setattr(http_client, "get", fake_get)

    result = flows_v2.exists("weka.ZeroR", "Weka_3.9.0")

    assert result == 123


def test_v2_flow_exists_not_found(monkeypatch):
    """Test FlowsV2.exists() when flow is not found (404)."""
    from openml._api.resources.flows import FlowsV2
    from openml._api.http.client import HTTPClient
    from openml._api.config import settings

    http_client = HTTPClient(settings.api.v2)
    flows_v2 = FlowsV2(http_client)

    def fake_get(path: str):
        raise requests.exceptions.HTTPError("404 Not Found")

    monkeypatch.setattr(http_client, "get", fake_get)

    result = flows_v2.exists("nonexistent.Flow", "v1.0.0")

    assert result is False


def test_v2_flow_get(monkeypatch, dummy_flow):
    """Test FlowsV2.get() converts v2 JSON to OpenMLFlow."""
    from openml._api.resources.flows import FlowsV2
    from openml._api.http.client import HTTPClient
    from openml._api.config import settings

    http_client = HTTPClient(settings.api.v2)
    flows_v2 = FlowsV2(http_client)

    # Mock v2 JSON response
    v2_json = {
        "id": 1,
        "uploader": 16,
        "name": "weka.ZeroR",
        "class_name": "weka.classifiers.rules.ZeroR",
        "version": 1,
        "external_version": "Weka_3.9.0_12024",
        "description": "Weka implementation of ZeroR",
        "upload_date": "2017-03-24T14:26:38",
        "language": "English",
        "dependencies": "Weka_3.9.0",
        "parameter": [
            {
                "name": "batch-size",
                "data_type": "option",
                "default_value": 100,
                "description": "Batch size for processing",
            }
        ],
        "subflows": [],
        "tag": ["weka", "OpenmlWeka"],
    }

    mock_response = requests.Response()
    mock_response.status_code = 200
    mock_response._content = b'{}'

    def fake_json():
        return v2_json

    mock_response.json = fake_json

    def fake_get(path: str):
        assert path == "flows/1/"
        return mock_response

    monkeypatch.setattr(http_client, "get", fake_get)

    flow = flows_v2.get(1)

    assert isinstance(flow, OpenMLFlow)
    assert flow.flow_id == 1
    assert flow.name == "weka.ZeroR"
    assert flow.external_version == "Weka_3.9.0_12024"
    assert flow.uploader == "16"
    assert len(flow.parameters) == 1
    assert "batch-size" in flow.parameters

