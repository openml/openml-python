from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..tasks.task import OpenMLTask
from ..flows.flow import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration


class Extension(ABC):

    """Defines the interface to connect machine learning libraries to OpenML-Python.
    """

    ################################################################################################
    # Abstract methods for flow serialization and de-serialization

    @abstractmethod
    def flow_to_model(self, flow: OpenMLFlow) -> Any:
        pass

    @abstractmethod
    def model_to_flow(self, model: Any) -> OpenMLFlow:
        pass

    @abstractmethod
    def flow_to_parameters(self, model: Any) -> List:
        pass

    ################################################################################################
    # Abstract methods for performing runs with extension modules

    @abstractmethod
    def is_estimator(self, model: Any) -> bool:
        pass

    @abstractmethod
    def seed_model(self, model: Any, seed: Optional[int]) -> Any:
        pass

    @abstractmethod
    def run_model_on_fold(
        self,
        model: Any,
        task: OpenMLTask,
        rep_no: int,
        fold_no: int,
        sample_no: int,
        can_measure_runtime: bool,
        add_local_measures: bool,
    ) -> Tuple:
        pass

    @abstractmethod
    def obtain_parameter_values(
        self,
        flow: OpenMLFlow,
        model: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts all parameter settings required for the flow from the model.
        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.
        """
        pass

    ################################################################################################
    # Abstract methods for hyperparameter optimization

    @abstractmethod
    def is_hpo_class(self, model: Any) -> bool:
        pass

    def assert_hpo_class(self, model: Any) -> None:
        if not self.is_hpo_class(model):
            raise AssertionError(
                "Flow model %s is not a hyperparameter optimization algorithm." % model
            )

    @abstractmethod
    def assert_hpo_class_has_trace(self, model: Any) -> None:
        pass

    @abstractmethod
    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: OpenMLTraceIteration,
    ) -> Any:
        pass

    @abstractmethod
    def obtain_arff_trace(
        self,
        model: Any,
        trace_content: List,
    ) -> OpenMLRunTrace:
        pass
