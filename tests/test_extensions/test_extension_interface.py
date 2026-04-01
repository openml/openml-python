# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.extensions.extension_interface module."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import Mock

import pytest

from openml.extensions.extension_interface import Extension


# Concrete implementation for testing the abstract Extension class
class ConcreteExtension(Extension):
    """Concrete implementation of Extension for testing."""

    @classmethod
    def can_handle_flow(cls, flow):
        return flow.external_version == "test_version"

    @classmethod
    def can_handle_model(cls, model):
        return hasattr(model, "test_marker")

    def flow_to_model(self, flow, initialize_with_defaults=False, strict_version=True):
        model = Mock()
        model.name = flow.name
        return model

    def model_to_flow(self, model):
        from openml.flows import OpenMLFlow
        flow = Mock(spec=OpenMLFlow)
        flow.name = getattr(model, "name", "default_flow")
        return flow

    def get_version_information(self):
        return ["test-library==1.0.0", "numpy==1.20.0"]

    def create_setup_string(self, model):
        return f"ConcreteModel({getattr(model, 'params', {})})"

    def is_estimator(self, model):
        return hasattr(model, "fit") and hasattr(model, "predict")

    def seed_model(self, model, seed):
        if hasattr(model, "random_state"):
            model.random_state = seed
        return model

    def _run_model_on_fold(self, model, task, X_train, rep_no, fold_no, y_train=None, X_test=None):
        import numpy as np
        predictions = np.array([0, 1, 0, 1])
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
        user_defined_measures = OrderedDict([("custom_metric", 0.95)])
        trace = None
        return predictions, probabilities, user_defined_measures, trace

    def obtain_parameter_values(self, flow, model=None):
        return [
            {"oml:name": "param1", "oml:value": "value1", "oml:component": 1},
            {"oml:name": "param2", "oml:value": "value2", "oml:component": 1},
        ]

    def check_if_model_fitted(self, model):
        return hasattr(model, "is_fitted") and model.is_fitted

    def instantiate_model_from_hpo_class(self, model, trace_iteration):
        instantiated = Mock()
        instantiated.params = trace_iteration
        return instantiated


class TestExtensionCanHandleFlow:
    """Test can_handle_flow class method."""

    def test_can_handle_flow_true(self):
        """Test can_handle_flow returns True for matching flow."""
        flow = Mock()
        flow.external_version = "test_version"
        
        assert ConcreteExtension.can_handle_flow(flow) is True

    def test_can_handle_flow_false(self):
        """Test can_handle_flow returns False for non-matching flow."""
        flow = Mock()
        flow.external_version = "other_version"
        
        assert ConcreteExtension.can_handle_flow(flow) is False

    def test_can_handle_flow_is_classmethod(self):
        """Test that can_handle_flow is a class method."""
        # Should be callable on class without instance
        flow = Mock()
        flow.external_version = "test_version"
        
        result = ConcreteExtension.can_handle_flow(flow)
        assert result is True


class TestExtensionCanHandleModel:
    """Test can_handle_model class method."""

    def test_can_handle_model_true(self):
        """Test can_handle_model returns True for matching model."""
        model = Mock()
        model.test_marker = True
        
        assert ConcreteExtension.can_handle_model(model) is True

    def test_can_handle_model_false(self):
        """Test can_handle_model returns False for non-matching model."""
        model = Mock(spec=[])
        # No test_marker attribute
        
        assert ConcreteExtension.can_handle_model(model) is False

    def test_can_handle_model_is_classmethod(self):
        """Test that can_handle_model is a class method."""
        model = Mock()
        model.test_marker = True
        
        result = ConcreteExtension.can_handle_model(model)
        assert result is True


class TestExtensionFlowToModel:
    """Test flow_to_model method."""

    def test_flow_to_model_basic(self):
        """Test basic flow to model conversion."""
        extension = ConcreteExtension()
        flow = Mock()
        flow.name = "TestFlow"
        
        model = extension.flow_to_model(flow)
        
        assert model.name == "TestFlow"

    def test_flow_to_model_with_defaults(self):
        """Test flow to model with initialize_with_defaults flag."""
        extension = ConcreteExtension()
        flow = Mock()
        flow.name = "DefaultFlow"
        
        model = extension.flow_to_model(flow, initialize_with_defaults=True)
        
        assert model is not None

    def test_flow_to_model_strict_version(self):
        """Test flow to model with strict_version flag."""
        extension = ConcreteExtension()
        flow = Mock()
        flow.name = "StrictFlow"
        
        model = extension.flow_to_model(flow, strict_version=True)
        
        assert model is not None

    def test_flow_to_model_non_strict_version(self):
        """Test flow to model with strict_version=False."""
        extension = ConcreteExtension()
        flow = Mock()
        flow.name = "NonStrictFlow"
        
        model = extension.flow_to_model(flow, strict_version=False)
        
        assert model is not None


class TestExtensionModelToFlow:
    """Test model_to_flow method."""

    def test_model_to_flow_basic(self):
        """Test basic model to flow conversion."""
        extension = ConcreteExtension()
        model = Mock()
        model.name = "TestModel"
        
        flow = extension.model_to_flow(model)
        
        assert flow.name == "TestModel"

    def test_model_to_flow_without_name(self):
        """Test model to flow when model has no name."""
        extension = ConcreteExtension()
        model = Mock(spec=[])  # No name attribute
        
        flow = extension.model_to_flow(model)
        
        assert flow.name == "default_flow"


class TestExtensionGetVersionInformation:
    """Test get_version_information method."""

    def test_get_version_information(self):
        """Test get_version_information returns list of versions."""
        extension = ConcreteExtension()
        
        versions = extension.get_version_information()
        
        assert isinstance(versions, list)
        assert "test-library==1.0.0" in versions
        assert "numpy==1.20.0" in versions

    def test_get_version_information_returns_list(self):
        """Test that return type is list."""
        extension = ConcreteExtension()
        
        versions = extension.get_version_information()
        
        assert isinstance(versions, list)
        assert len(versions) > 0


class TestExtensionCreateSetupString:
    """Test create_setup_string method."""

    def test_create_setup_string_basic(self):
        """Test create_setup_string returns string representation."""
        extension = ConcreteExtension()
        model = Mock()
        model.params = {"param1": "value1"}
        
        setup_str = extension.create_setup_string(model)
        
        assert isinstance(setup_str, str)
        assert "ConcreteModel" in setup_str

    def test_create_setup_string_without_params(self):
        """Test create_setup_string when model has no params."""
        extension = ConcreteExtension()
        model = Mock(spec=[])
        
        setup_str = extension.create_setup_string(model)
        
        assert isinstance(setup_str, str)


class TestExtensionIsEstimator:
    """Test is_estimator method."""

    def test_is_estimator_true(self):
        """Test is_estimator returns True for estimator-like model."""
        extension = ConcreteExtension()
        model = Mock()
        model.fit = Mock()
        model.predict = Mock()
        
        assert extension.is_estimator(model) is True

    def test_is_estimator_false_no_fit(self):
        """Test is_estimator returns False when no fit method."""
        extension = ConcreteExtension()
        model = Mock(spec=['predict'])
        model.predict = Mock()
        # No fit method
        
        assert extension.is_estimator(model) is False

    def test_is_estimator_false_no_predict(self):
        """Test is_estimator returns False when no predict method."""
        extension = ConcreteExtension()
        model = Mock(spec=['fit'])
        model.fit = Mock()
        # No predict method
        
        assert extension.is_estimator(model) is False


class TestExtensionSeedModel:
    """Test seed_model method."""

    def test_seed_model_sets_random_state(self):
        """Test seed_model sets random_state attribute."""
        extension = ConcreteExtension()
        model = Mock()
        model.random_state = None
        
        seeded_model = extension.seed_model(model, 42)
        
        assert seeded_model.random_state == 42

    def test_seed_model_returns_model(self):
        """Test seed_model returns the model."""
        extension = ConcreteExtension()
        model = Mock()
        model.random_state = None
        
        result = extension.seed_model(model, 123)
        
        assert result is model

    def test_seed_model_without_random_state(self):
        """Test seed_model when model doesn't have random_state."""
        extension = ConcreteExtension()
        model = Mock(spec=[])  # No random_state attribute
        
        result = extension.seed_model(model, 42)
        
        assert result is model


class TestExtensionRunModelOnFold:
    """Test _run_model_on_fold method."""

    def test_run_model_on_fold_returns_tuple(self):
        """Test _run_model_on_fold returns correct tuple."""
        extension = ConcreteExtension()
        model = Mock()
        task = Mock()
        X_train = Mock()
        y_train = Mock()
        X_test = Mock()
        
        result = extension._run_model_on_fold(
            model, task, X_train, 0, 0, y_train, X_test
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_run_model_on_fold_predictions(self):
        """Test _run_model_on_fold returns predictions."""
        import numpy as np
        extension = ConcreteExtension()
        
        predictions, probabilities, measures, trace = extension._run_model_on_fold(
            Mock(), Mock(), Mock(), 0, 0
        )
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 4

    def test_run_model_on_fold_probabilities(self):
        """Test _run_model_on_fold returns probabilities."""
        import numpy as np
        extension = ConcreteExtension()
        
        predictions, probabilities, measures, trace = extension._run_model_on_fold(
            Mock(), Mock(), Mock(), 0, 0
        )
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (4, 2)

    def test_run_model_on_fold_measures(self):
        """Test _run_model_on_fold returns user-defined measures."""
        extension = ConcreteExtension()
        
        predictions, probabilities, measures, trace = extension._run_model_on_fold(
            Mock(), Mock(), Mock(), 0, 0
        )
        
        assert isinstance(measures, OrderedDict)
        assert "custom_metric" in measures
        assert measures["custom_metric"] == 0.95


class TestExtensionObtainParameterValues:
    """Test obtain_parameter_values method."""

    def test_obtain_parameter_values_returns_list(self):
        """Test obtain_parameter_values returns list."""
        extension = ConcreteExtension()
        flow = Mock()
        
        params = extension.obtain_parameter_values(flow)
        
        assert isinstance(params, list)
        assert len(params) == 2

    def test_obtain_parameter_values_structure(self):
        """Test obtain_parameter_values returns correct structure."""
        extension = ConcreteExtension()
        flow = Mock()
        
        params = extension.obtain_parameter_values(flow)
        
        assert all(isinstance(p, dict) for p in params)
        assert all("oml:name" in p for p in params)
        assert all("oml:value" in p for p in params)
        assert all("oml:component" in p for p in params)

    def test_obtain_parameter_values_with_model(self):
        """Test obtain_parameter_values with explicit model."""
        extension = ConcreteExtension()
        flow = Mock()
        model = Mock()
        
        params = extension.obtain_parameter_values(flow, model)
        
        assert isinstance(params, list)


class TestExtensionCheckIfModelFitted:
    """Test check_if_model_fitted method."""

    def test_check_if_model_fitted_true(self):
        """Test check_if_model_fitted returns True for fitted model."""
        extension = ConcreteExtension()
        model = Mock()
        model.is_fitted = True
        
        assert extension.check_if_model_fitted(model) is True

    def test_check_if_model_fitted_false(self):
        """Test check_if_model_fitted returns False for unfitted model."""
        extension = ConcreteExtension()
        model = Mock()
        model.is_fitted = False
        
        assert extension.check_if_model_fitted(model) is False

    def test_check_if_model_fitted_no_attribute(self):
        """Test check_if_model_fitted when no is_fitted attribute."""
        extension = ConcreteExtension()
        model = Mock(spec=[])
        
        assert extension.check_if_model_fitted(model) is False


class TestExtensionInstantiateModelFromHPOClass:
    """Test instantiate_model_from_hpo_class method."""

    def test_instantiate_model_from_hpo_class(self):
        """Test instantiate_model_from_hpo_class returns model."""
        extension = ConcreteExtension()
        model = Mock()
        trace_iteration = Mock()
        
        instantiated = extension.instantiate_model_from_hpo_class(model, trace_iteration)
        
        assert instantiated is not None
        assert hasattr(instantiated, "params")

    def test_instantiate_model_from_hpo_class_with_trace(self):
        """Test instantiation with trace iteration data."""
        extension = ConcreteExtension()
        model = Mock()
        trace_iteration = {"param1": 10, "param2": 0.5}
        
        instantiated = extension.instantiate_model_from_hpo_class(model, trace_iteration)
        
        assert instantiated.params == trace_iteration


class TestExtensionAbstractMethods:
    """Test that Extension is abstract and requires implementation."""

    def test_cannot_instantiate_extension_directly(self):
        """Test that Extension cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Extension()

    def test_concrete_extension_can_be_instantiated(self):
        """Test that concrete implementation can be instantiated."""
        extension = ConcreteExtension()
        
        assert isinstance(extension, Extension)
        assert isinstance(extension, ConcreteExtension)


class TestExtensionEdgeCases:
    """Test edge cases for Extension."""

    def test_flow_to_model_with_none_flow(self):
        """Test flow_to_model behavior with None-like flow."""
        extension = ConcreteExtension()
        flow = Mock()
        flow.name = None
        
        model = extension.flow_to_model(flow)
        
        assert model.name is None

    def test_seed_model_with_none_seed(self):
        """Test seed_model with None seed."""
        extension = ConcreteExtension()
        model = Mock()
        model.random_state = 42
        
        result = extension.seed_model(model, None)
        
        assert result.random_state is None

    def test_seed_model_with_zero_seed(self):
        """Test seed_model with zero seed."""
        extension = ConcreteExtension()
        model = Mock()
        model.random_state = None
        
        result = extension.seed_model(model, 0)
        
        assert result.random_state == 0

    def test_seed_model_with_negative_seed(self):
        """Test seed_model with negative seed."""
        extension = ConcreteExtension()
        model = Mock()
        model.random_state = None
        
        result = extension.seed_model(model, -1)
        
        assert result.random_state == -1

    def test_obtain_parameter_values_empty_flow(self):
        """Test obtain_parameter_values with minimal flow."""
        extension = ConcreteExtension()
        flow = Mock(spec=[])
        
        params = extension.obtain_parameter_values(flow)
        
        # Should still return the default parameters
        assert isinstance(params, list)

    def test_create_setup_string_complex_model(self):
        """Test create_setup_string with complex model parameters."""
        extension = ConcreteExtension()
        model = Mock()
        model.params = {
            "nested": {"param1": 1, "param2": 2},
            "list_param": [1, 2, 3],
            "string_param": "value"
        }
        
        setup_str = extension.create_setup_string(model)
        
        assert isinstance(setup_str, str)


class TestExtensionIntegration:
    """Integration tests for Extension."""

    def test_full_workflow_model_to_flow_and_back(self):
        """Test converting model to flow and back to model."""
        extension = ConcreteExtension()
        
        # Create a model
        original_model = Mock()
        original_model.name = "OriginalModel"
        original_model.test_marker = True
        
        # Convert to flow
        flow = extension.model_to_flow(original_model)
        
        # Convert back to model
        reconstructed_model = extension.flow_to_model(flow)
        
        assert reconstructed_model.name == original_model.name

    def test_extension_with_different_model_types(self):
        """Test extension can handle different model types."""
        extension = ConcreteExtension()
        
        models = [
            Mock(test_marker=True, name="Model1"),
            Mock(test_marker=True, name="Model2"),
            Mock(test_marker=True, name="Model3"),
        ]
        
        for model in models:
            assert ConcreteExtension.can_handle_model(model) is True
            flow = extension.model_to_flow(model)
            assert flow.name == model.name

    def test_seeding_multiple_models(self):
        """Test seeding multiple models."""
        extension = ConcreteExtension()
        seeds = [0, 42, 123, 999]
        
        for seed in seeds:
            model = Mock()
            model.random_state = None
            
            seeded = extension.seed_model(model, seed)
            
            assert seeded.random_state == seed
