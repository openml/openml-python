# License: BSD 3-Clause  
from __future__ import annotations  
  
import pytest  
import hashlib
import time
import sklearn.tree
import sklearn.naive_bayes
import openml
from openml_sklearn import SklearnExtension
from openml.testing import TestBase

  
from openml._api import SetupV1API, SetupV2API
from openml.setups.setup import OpenMLSetup
from openml.exceptions import OpenMLNotSupportedError

def get_sentinel():
    # Create a unique prefix for the flow. Necessary because the flow is
    # identified by its name and external version online. Having a unique
    #  name allows us to publish the same flow in each test run
    md5 = hashlib.md5()
    md5.update(str(time.time()).encode("utf-8"))
    sentinel = md5.hexdigest()[:10]
    return f"TEST{sentinel}"  

@pytest.fixture
def setup_v1(http_client_v1, minio_client) -> SetupV1API:
    return SetupV1API(http=http_client_v1, minio=minio_client)

@pytest.fixture
def setup_v2(http_client_v2, minio_client) -> SetupV2API:
    return SetupV2API(http=http_client_v2, minio=minio_client)


@pytest.mark.uses_test_server()
def test_v1_list(setup_v1):
    setups = setup_v1.list(limit=10, offset=0)
    
    assert isinstance(setups, list)
    assert len(setups) > 0
    assert all(isinstance(s, OpenMLSetup) for s in setups)

@pytest.mark.uses_test_server()
def test_v1_get(setup_v1):
    setup_id = 1
    setup = setup_v1.get(setup_id)
    
    assert isinstance(setup, OpenMLSetup)
    assert setup.setup_id == setup_id

@pytest.mark.sklearn()
@pytest.mark.uses_test_server()
def test_v1_exists_nonexisting_setup(setup_v1):
    """Test exists() returns False when setup doesn't exist"""
    # first publish a non-existing flow
    sentinel = get_sentinel()
    # because of the sentinel, we can not use flows that contain subflows
    dectree = sklearn.tree.DecisionTreeClassifier()
    flow = SklearnExtension().model_to_flow(dectree)
    flow.name = f"{sentinel}{flow.name}"
    flow.publish()
    TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")
    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    # although the flow exists (created as of previous statement),
    # we can be sure there are no setups (yet) as it was just created
    # and hasn't been ran
    setup_id = setup_v1.exists(flow, openml_param_settings)
    assert not setup_id

@pytest.mark.sklearn()
@pytest.mark.uses_test_server()
def test_v1_exists_existing_setup(setup_v1):
    """Test exists() returns setup_id when setup exists"""
    flow =SklearnExtension().model_to_flow(
        sklearn.naive_bayes.GaussianNB()
    )
    flow.name = f"{get_sentinel()}{flow.name}"
    flow.publish()
    TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")
    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    # now run the flow on an easy task:
    task = openml.tasks.get_task(115)
    run = openml.runs.run_flow_on_task(flow, task)
    # spoof flow id, otherwise the sentinel is ignored
    run.flow_id = flow.flow_id
    run.publish()
    TestBase._mark_entity_for_removal("run", run.run_id)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {run.run_id}")
    # download the run, as it contains the right setup id
    run = openml.runs.get_run(run.run_id)
    # execute the function we are interested in
    setup_id = setup_v1.exists(flow, openml_param_settings)
    assert setup_id == run.setup_id

@pytest.mark.uses_test_server()
def test_v2_list(setup_v2):
    with pytest.raises(OpenMLNotSupportedError):
        setup_v2.list(limit=10, offset=0)

@pytest.mark.uses_test_server()
def test_v2_get(setup_v2):
    with pytest.raises(OpenMLNotSupportedError):
        setup_v2.get(1)

@pytest.mark.sklearn()
@pytest.mark.uses_test_server()
def test_v2_exists_nonexisting_setup(setup_v2):
    # first publish a non-existing flow
    sentinel = get_sentinel()
    # because of the sentinel, we can not use flows that contain subflows
    dectree = sklearn.tree.DecisionTreeClassifier()
    flow = SklearnExtension().model_to_flow(dectree)
    flow.name = f"{sentinel}{flow.name}"
    flow.publish()
    TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")
    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    # although the flow exists (created as of previous statement),
    # we can be sure there are no setups (yet) as it was just created
    # and hasn't been ran
    with pytest.raises(OpenMLNotSupportedError):
        setup_v2.exists(flow, openml_param_settings)

    
@pytest.mark.sklearn()
@pytest.mark.uses_test_server()
def test_v2_exists_existing_setup(setup_v2):
    flow =SklearnExtension().model_to_flow(
        sklearn.naive_bayes.GaussianNB()
    )
    flow.name = f"{get_sentinel()}{flow.name}"
    flow.publish()
    TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")
    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    # now run the flow on an easy task:
    task = openml.tasks.get_task(115)
    run = openml.runs.run_flow_on_task(flow, task)
    # spoof flow id, otherwise the sentinel is ignored
    run.flow_id = flow.flow_id
    run.publish()
    TestBase._mark_entity_for_removal("run", run.run_id)
    TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {run.run_id}")
    # download the run, as it contains the right setup id
    run = openml.runs.get_run(run.run_id)
    with pytest.raises(OpenMLNotSupportedError):
        setup_v2.exists(flow, openml_param_settings)
