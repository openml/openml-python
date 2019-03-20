from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from openml import OpenMLTask, OpenMLFlow
from openml.runs.trace import OpenMLTraceIteration, OpenMLRunTrace


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
        extension: 'Extension',
    ) -> Tuple:
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
        extension: 'Extension',
        model: Any,
        trace_content: List,
    ) -> OpenMLRunTrace:
        pass
