# License: BSD 3-Clause
from __future__ import annotations

from collections import OrderedDict

import inspect
import numpy as np
import pytest
from unittest.mock import patch
import openml.testing
from openml.extensions import Extension, get_extension_by_flow, get_extension_by_model, register_extension


class DummyFlow:
    external_version = "DummyFlow==0.1"
    name = "Dummy Flow"
    flow_id = 1
    dependencies = None


class DummyModel:
    pass


class DummyExtension1:
    @staticmethod
    def can_handle_flow(flow):
        return inspect.stack()[2].filename.endswith("test_functions.py")

    @staticmethod
    def can_handle_model(model):
        return inspect.stack()[2].filename.endswith("test_functions.py")


class DummyExtension2:
    @staticmethod
    def can_handle_flow(flow):
        return False

    @staticmethod
    def can_handle_model(model):
        return False


class DummyExtension(Extension):
    @classmethod
    def can_handle_flow(cls, flow):
        return isinstance(flow, DummyFlow)

    @classmethod
    def can_handle_model(cls, model):
        return isinstance(model, DummyModel)

    def flow_to_model(
        self,
        flow,
        initialize_with_defaults=False,
        strict_version=True,
    ):
        if not isinstance(flow, DummyFlow):
            raise ValueError("Invalid flow")

        model = DummyModel()
        model.defaults = initialize_with_defaults
        model.strict_version = strict_version
        return model

    def model_to_flow(self, model):
        if not isinstance(model, DummyModel):
            raise ValueError("Invalid model")
        return DummyFlow()

    def get_version_information(self):
        return ["dummy==1.0"]

    def create_setup_string(self, model):
        return "DummyModel()"

    def is_estimator(self, model):
        return isinstance(model, DummyModel)

    def seed_model(self, model, seed):
        model.seed = seed
        return model

    def _run_model_on_fold(
        self,
        model,
        task,
        X_train,
        rep_no,
        fold_no,
        y_train=None,
        X_test=None,
    ):
        preds = np.zeros(len(X_train))
        probs = None
        measures = OrderedDict()
        trace = None
        return preds, probs, measures, trace

    def obtain_parameter_values(self, flow, model=None):
        return []

    def check_if_model_fitted(self, model):
        return False

    def instantiate_model_from_hpo_class(self, model, trace_iteration):
        return DummyModel()



class TestInit(openml.testing.TestBase):

    def test_get_extension_by_flow(self):
            # We replace the global list with a new empty list [] ONLY for this block
            with patch("openml.extensions.extensions", []):
                assert get_extension_by_flow(DummyFlow()) is None
                
                with pytest.raises(ValueError, match="No extension registered which can handle flow:"):
                    get_extension_by_flow(DummyFlow(), raise_if_no_extension=True)
                
                register_extension(DummyExtension1)
                assert isinstance(get_extension_by_flow(DummyFlow()), DummyExtension1)
                
                register_extension(DummyExtension2)
                assert isinstance(get_extension_by_flow(DummyFlow()), DummyExtension1)
                
                register_extension(DummyExtension1)
                with pytest.raises(
                    ValueError, match="Multiple extensions registered which can handle flow:"
                ):
                    get_extension_by_flow(DummyFlow())

    def test_get_extension_by_model(self):
        # Again, we start with a fresh empty list automatically
        with patch("openml.extensions.extensions", []):
            assert get_extension_by_model(DummyModel()) is None
            
            with pytest.raises(ValueError, match="No extension registered which can handle model:"):
                get_extension_by_model(DummyModel(), raise_if_no_extension=True)
            
            register_extension(DummyExtension1)
            assert isinstance(get_extension_by_model(DummyModel()), DummyExtension1)
            
            register_extension(DummyExtension2)
            assert isinstance(get_extension_by_model(DummyModel()), DummyExtension1)
            
            register_extension(DummyExtension1)
            with pytest.raises(
                ValueError, match="Multiple extensions registered which can handle model:"
            ):
                get_extension_by_model(DummyModel())


def test_flow_to_model_with_defaults():
    """Test flow_to_model with initialize_with_defaults=True."""
    ext = DummyExtension()
    flow = DummyFlow()

    model = ext.flow_to_model(flow, initialize_with_defaults=True)

    assert isinstance(model, DummyModel)
    assert model.defaults is True

def test_flow_to_model_strict_version():
    """Test flow_to_model with strict_version parameter."""
    ext = DummyExtension()
    flow = DummyFlow()

    model_strict = ext.flow_to_model(flow, strict_version=True)
    model_non_strict = ext.flow_to_model(flow, strict_version=False)

    assert isinstance(model_strict, DummyModel)
    assert model_strict.strict_version is True

    assert isinstance(model_non_strict, DummyModel)
    assert model_non_strict.strict_version is False

def test_model_to_flow_conversion():
    """Test converting a model back to flow representation."""
    ext = DummyExtension()
    model = DummyModel()

    flow = ext.model_to_flow(model)

    assert isinstance(flow, DummyFlow)


def test_invalid_flow_raises_error():
    """Test that invalid flow raises appropriate error."""
    class InvalidFlow:
        pass

    ext = DummyExtension()
    flow = InvalidFlow()

    with pytest.raises(ValueError, match="Invalid flow"):
        ext.flow_to_model(flow)


@patch("openml.extensions.extensions", [])
def test_extension_not_found_error_message():
    """Test error message contains helpful information."""
    class UnknownModel:
        pass

    with pytest.raises(ValueError, match="No extension registered"):
        get_extension_by_model(UnknownModel(), raise_if_no_extension=True)

 
def test_register_same_extension_twice():
    """Test behavior when registering same extension twice."""
    # Using a context manager here to isolate the list
    with patch("openml.extensions.extensions", []):
        register_extension(DummyExtension)
        register_extension(DummyExtension)

        matches = [
            ext for ext in openml.extensions.extensions
            if ext is DummyExtension
        ]
        assert len(matches) == 2


@patch("openml.extensions.extensions", [])
def test_extension_priority_order():
    """Test that extensions are checked in registration order."""    
    class DummyExtensionA(DummyExtension):
        pass
    class DummyExtensionB(DummyExtension):
        pass

    register_extension(DummyExtensionA)
    register_extension(DummyExtensionB)

    assert openml.extensions.extensions[0] is DummyExtensionA
    assert openml.extensions.extensions[1] is DummyExtensionB