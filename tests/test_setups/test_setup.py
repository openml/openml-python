# License: BSD 3-Clause
from __future__ import annotations

import pytest

import openml
from openml.setups import OpenMLParameter, OpenMLSetup


@pytest.fixture
def sample_parameters():
    param1 = OpenMLParameter(
        input_id=1,
        flow_id=100,
        flow_name="sklearn.tree.DecisionTreeClassifier",
        full_name="sklearn.tree.DecisionTreeClassifier(1)",
        parameter_name="max_depth",
        data_type="int",
        default_value="None",
        value="5",
    )
    param2 = OpenMLParameter(
        input_id=2,
        flow_id=100,
        flow_name="sklearn.tree.DecisionTreeClassifier",
        full_name="sklearn.tree.DecisionTreeClassifier(2)",
        parameter_name="min_samples_split",
        data_type="int",
        default_value="2",
        value="3",
    )
    return {1: param1, 2: param2}


@pytest.fixture
def setup(sample_parameters):
    return OpenMLSetup(setup_id=42, flow_id=100, parameters=sample_parameters)


def test_id_property(setup):
    assert setup.id == 42


def test_openml_url(setup):
    assert setup.openml_url == f"{openml.config.get_server_base_url()}/s/42"


def test_repr(setup):
    repr_str = repr(setup)
    assert "OpenML Setup" in repr_str
    assert "Setup ID" in repr_str
    assert "42" in repr_str
    assert "Flow ID" in repr_str
    assert "# of Parameters" in repr_str


def test_repr_none_parameters():
    s = OpenMLSetup(setup_id=7, flow_id=200, parameters=None)
    assert "# of Parameters" in repr(s)


def test_to_dict(setup):
    result = setup._to_dict()
    assert result["setup_id"] == 42
    assert result["flow_id"] == 100
    assert len(result["parameters"]) == 2


def test_to_dict_none_parameters():
    s = OpenMLSetup(setup_id=7, flow_id=200, parameters=None)
    assert s._to_dict()["parameters"] is None


def test_publish_raises(setup):
    with pytest.raises(NotImplementedError, match="Setups cannot be published directly"):
        setup.publish()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("setup_id", "bad", "setup id should be int"),
        ("flow_id", "bad", "flow id should be int"),
        ("parameters", "bad", "parameters should be dict"),
    ],
)
def test_invalid_init(field, value, match):
    kwargs = {"setup_id": 1, "flow_id": 1, "parameters": None, field: value}
    with pytest.raises(ValueError, match=match):
        OpenMLSetup(**kwargs)
