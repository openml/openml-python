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

  
from openml._api.resources.setup import SetupV1API, SetupV2API
from openml.setups.setup import OpenMLSetup
from openml.testing import TestAPIBase
from openml.enums import APIVersion
from openml._api.resources.base.fallback import FallbackProxy
from openml.exceptions import OpenMLNotSupportedError

def get_sentinel():
    # Create a unique prefix for the flow. Necessary because the flow is
    # identified by its name and external version online. Having a unique
    #  name allows us to publish the same flow in each test run
    md5 = hashlib.md5()
    md5.update(str(time.time()).encode("utf-8"))
    sentinel = md5.hexdigest()[:10]
    return f"TEST{sentinel}"  
  
@pytest.mark.uses_test_server()
class TestSetupV1(TestAPIBase):  
    """Tests for V1 XML API implementation of setups."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V1]
        self.resource = SetupV1API(http_client)
        self.extension = SklearnExtension()
  
    def test_list(self):
        setups = self.resource.list(limit=10, offset=0)
        
        assert isinstance(setups, list)
        assert len(setups) > 0
        assert all(isinstance(s, OpenMLSetup) for s in setups)
    
    def test_get(self):
        setup_id = 1
        setup = self.resource.get(setup_id)
        
        assert isinstance(setup, OpenMLSetup)
        assert setup.setup_id == setup_id

    @pytest.mark.sklearn()
    def test_exists_nonexisting_setup(self):
        """Test exists() returns False when setup doesn't exist"""
        # first publish a non-existing flow
        sentinel = get_sentinel()
        # because of the sentinel, we can not use flows that contain subflows
        dectree = sklearn.tree.DecisionTreeClassifier()
        flow = self.extension.model_to_flow(dectree)
        flow.name = f"{sentinel}{flow.name}"
        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")

        openml_param_settings = flow.extension.obtain_parameter_values(flow)
        # although the flow exists (created as of previous statement),
        # we can be sure there are no setups (yet) as it was just created
        # and hasn't been ran
        setup_id = self.resource.exists(flow, openml_param_settings)
        assert not setup_id
    
    @pytest.mark.sklearn()
    def test_exists_existing_setup(self):
        """Test exists() returns setup_id when setup exists"""
        flow = self.extension.model_to_flow(sklearn.naive_bayes.GaussianNB())
        flow.name = f"{get_sentinel()}{flow.name}"
        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")

        openml_param_settings = flow.extension.obtain_parameter_values(flow)
        # although the flow exists, we can be sure there are no
        # setups (yet) as it hasn't been ran
        setup_id = self.resource.exists(flow, openml_param_settings)
        assert not setup_id

        # now run the flow on an easy task:
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        run = openml.runs.run_flow_on_task(flow, task)
        # spoof flow id, otherwise the sentinel is ignored
        run.flow_id = flow.flow_id
        run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {run.run_id}")
        # download the run, as it contains the right setup id
        run = openml.runs.get_run(run.run_id)
        # execute the function we are interested in
        setup_id = self.resource.exists(flow, openml_param_settings)
        assert setup_id == run.setup_id

@pytest.mark.uses_test_server()
class TestSetupV2(TestAPIBase): 
    """Tests for V2 JSON API implementation of setups."""  
  
    _multiprocess_can_split_ = True  
  
    def setUp(self) -> None:    
        super().setUp() 
        http_client = self.http_clients[APIVersion.V2]
        self.resource = SetupV2API(http_client)
        self.extension = SklearnExtension()

    def test_list(self):
        with pytest.raises(OpenMLNotSupportedError):
            self.resource.list(limit=10, offset=0)
    
    def test_get(self):
        with pytest.raises(OpenMLNotSupportedError):
            setup_id = 1
            self.resource.get(setup_id)

    @pytest.mark.sklearn()
    def test_exists_nonexisting_setup(self):
        # first publish a non-existing flow
        sentinel = get_sentinel()
        # because of the sentinel, we can not use flows that contain subflows
        dectree = sklearn.tree.DecisionTreeClassifier()
        flow = self.extension.model_to_flow(dectree)
        flow.name = f"{sentinel}{flow.name}"
        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")

        openml_param_settings = flow.extension.obtain_parameter_values(flow)
        # although the flow exists (created as of previous statement),
        # we can be sure there are no setups (yet) as it was just created
        # and hasn't been ran
        with pytest.raises(OpenMLNotSupportedError):
            self.resource.exists(flow, openml_param_settings)
    
    @pytest.mark.sklearn()
    def test_exists_existing_setup(self):
        flow = self.extension.model_to_flow(sklearn.naive_bayes.GaussianNB())
        flow.name = f"{get_sentinel()}{flow.name}"
        flow.publish()
        TestBase._mark_entity_for_removal("flow", flow.flow_id, flow.name)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {flow.flow_id}")

        openml_param_settings = flow.extension.obtain_parameter_values(flow)
        # now run the flow on an easy task:
        task = openml.tasks.get_task(115)  # diabetes; crossvalidation
        run = openml.runs.run_flow_on_task(flow, task)
        # spoof flow id, otherwise the sentinel is ignored
        run.flow_id = flow.flow_id
        run.publish()
        TestBase._mark_entity_for_removal("run", run.run_id)
        TestBase.logger.info(f"collected from {__file__.split('/')[-1]}: {run.run_id}")
        # download the run, as it contains the right setup id
        run = openml.runs.get_run(run.run_id)
        # execute the function we are interested in

        with pytest.raises(OpenMLNotSupportedError):
            self.resource.exists(flow, openml_param_settings)


class TestSetupsCombined(TestAPIBase):
    def setUp(self):
        super().setUp()
        self.v1_client = self.http_clients[APIVersion.V1]
        self.v2_client = self.http_clients[APIVersion.V2]
        self.resource_v1 = SetupV1API(self.v1_client)
        self.resource_v2 = SetupV2API(self.v2_client)