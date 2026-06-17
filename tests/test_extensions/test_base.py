# License: BSD 3-Clause

"""Test OpenML extension base classes and registry."""

import pytest
from collections import OrderedDict

from openml.exceptions import PyOpenMLError
from openml.extensions.base import (
    ModelSerializer,
    ModelExecutor,
)
from openml.extensions.registry import resolve_serializer, resolve_executor


class TestModelSerializer:
    """Test ModelSerializer abstract base class."""

    def test_is_abstract(self):
        """ModelSerializer should not be instantiable."""
        with pytest.raises(TypeError):
            ModelSerializer()  # noqa: B024

    class DummySerializer(ModelSerializer):
        @classmethod
        def can_handle_model(cls, model):
            return True

        def model_to_flow(self, model):
            return "dummy_flow"

        def flow_to_model(self, flow, initialize_with_defaults=False, strict_version=True):
            return "dummy_model"

        def get_version_information(self):
            return ["dummy>=0.1"]
        
        def obtain_parameter_values(self, flow, model=None):
            return []

    def test_concrete_implementation(self):
        serializer = self.DummySerializer()

        assert serializer.can_handle_model(object()) is True
        assert serializer.model_to_flow("model") == "dummy_flow"
        assert serializer.flow_to_model("flow") == "dummy_model"
        assert serializer.get_version_information() == ["dummy>=0.1"]


class TestModelExecutor:
    """Test ModelExecutor abstract base class."""

    def test_is_abstract(self):
        """ModelExecutor should not be instantiable."""
        with pytest.raises(TypeError):
            ModelExecutor()  # noqa: B024

    class DummyExecutor(ModelExecutor):
        @classmethod
        def can_handle_model(cls, model):
            return True
    
        def seed_model(self, model, seed):
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
            return (
                [],             # predictions
                None,            # probabilities
                OrderedDict(),   # user_defined_measures
                None,            # trace
            )

        def check_if_model_fitted(self, model):
            return False

        def instantiate_model_from_hpo_class(self, model, trace_iteration):
            return model

    def test_concrete_implementation(self):
        executor = self.DummyExecutor()

        assert executor.seed_model("model", 42) == "model"
        assert executor.check_if_model_fitted("model") is False
