# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd
import pytest
import requests

import openml
from openml._api import api_context
from openml.exceptions import OpenMLCacheException
from openml.flows import OpenMLFlow
from openml.flows import functions as flow_functions


@pytest.fixture(scope="function")
def reset_api_to_v1() -> None:
    """Fixture to ensure API is set to V1 for each test."""
    api_context.set_version("v1", strict=False)
    yield
    api_context.set_version("v1", strict=False)


@pytest.fixture(scope="function")
def api_v2() -> None:
    """Fixture to set API to V2 for tests."""
    api_context.set_version("v2", strict=True)
    yield
    api_context.set_version("v1", strict=False)


def test_list_flow_v1(reset_api_to_v1) -> None:
    """Test listing flows using V1 API."""
    flows_df = flow_functions.list_flows()
    assert isinstance(flows_df, pd.DataFrame)
    assert not flows_df.empty


def test_flow_exists_v1(reset_api_to_v1) -> None:
    """Test flow_exists() using V1 API."""
    # Known existing flow
    name = "weka.OneR"
    external_version = "Weka_3.9.0_10153"

    exists = flow_functions.flow_exists(name, external_version)
    assert exists != False

    # Known non-existing flow
    name = "non.existing.Flow"
    external_version = "0.0.1"

    exists = flow_functions.flow_exists(name, external_version)
    assert exists is False


def test_get_flows_v1(reset_api_to_v1) -> None:
    """Test get() method returns a valid OpenMLFlow object using V1 API."""
    # Get the flow with ID 2 (weka.OneR)
    flow_id = 2
    flow = flow_functions.get_flow(flow_id)

    assert isinstance(flow, OpenMLFlow)
    assert flow.flow_id == flow_id
    assert isinstance(flow.name, str)
    assert len(flow.name) > 0
    assert isinstance(flow.external_version, str)


def test_flow_publish_v1(reset_api_to_v1) -> None:
    """Test publishing a flow using V1 API."""
    from openml_sklearn.extension import SklearnExtension
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    extension = SklearnExtension()
    dt_flow = extension.model_to_flow(clf)

    # Publish the flow
    published_flow = dt_flow.publish()

    # Verify the published flow has an ID
    assert isinstance(published_flow, OpenMLFlow)
    assert getattr(published_flow, "id", None) is not None


def test_get_flows_v2(api_v2) -> None:
    """Test get() method returns a valid OpenMLFlow object using V2 API."""
    # Get the flow with ID 2 (weka.OneR)
    flow_id = 2

    # Now get the full flow details
    flow = flow_functions.get_flow(flow_id)

    # Verify it's an OpenMLFlow with expected attributes
    assert isinstance(flow, OpenMLFlow)
    assert flow.flow_id == flow_id
    assert isinstance(flow.name, str)
    assert len(flow.name) > 0
    assert isinstance(flow.external_version, str)


def test_flow_exists_v2(api_v2) -> None:
    """Test flow_exists() using V2 API."""
    # Known existing flow
    name = "weka.OneR"
    external_version = "Weka_3.9.0_10153"

    exists = flow_functions.flow_exists(name, external_version)
    assert exists != False

    # Known non-existing flow
    name = "non.existing.Flow"
    external_version = "0.0.1"

    exists = flow_functions.flow_exists(name, external_version)
    assert exists == False
    