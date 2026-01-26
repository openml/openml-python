# License: BSD 3-Clause

"""Test OpenML extension base classes and registry."""

import pytest
from collections import OrderedDict

from openml.exceptions import PyOpenMLError
from openml.extensions.base import (
    ModelSerializer,
    ModelExecutor,
    OpenMLAPIConnector,
)
from openml.extensions.registry import resolve_api_connector


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
        assert executor.obtain_parameter_values("flow") == []


class TestOpenMLAPIConnector:
    """Test OpenMLAPIConnector abstract base class."""

    def test_is_abstract(self):
        """OpenMLAPIConnector should not be instantiable."""
        with pytest.raises(TypeError):
            OpenMLAPIConnector()  # noqa: B024

    class DummySerializer:
        pass

    class DummyExecutor:
        pass

    class DummyConnector(OpenMLAPIConnector):
        def serializer(self):
            return TestOpenMLAPIConnector.DummySerializer()

        def executor(self):
            return TestOpenMLAPIConnector.DummyExecutor()

        @classmethod
        def supports(cls, estimator):
            return estimator == "supported"

    def test_concrete_implementation(self):
        connector = self.DummyConnector()

        assert isinstance(connector.serializer(), self.DummySerializer)
        assert isinstance(connector.executor(), self.DummyExecutor)
        assert self.DummyConnector.supports("supported") is True
        assert self.DummyConnector.supports("unsupported") is False

    def test_resolve_api_connector_success(self, monkeypatch):
        monkeypatch.setattr(
            "openml.extensions.registry.API_CONNECTOR_REGISTRY",
            [self.DummyConnector],
        )

        connector = resolve_api_connector("supported")
        assert isinstance(connector, self.DummyConnector)

    def test_resolve_api_connector_no_match(self, monkeypatch):
        monkeypatch.setattr(
            "openml.extensions.registry.API_CONNECTOR_REGISTRY",
            [],
        )

        with pytest.raises(PyOpenMLError, match="No OpenML API connector supports"):
            resolve_api_connector("anything")
