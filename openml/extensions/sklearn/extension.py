from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import sklearn.base

from openml.extensions import Extension, register_extension
from openml.extensions.sklearn.flow_functions import (
    sklearn_to_flow,
    flow_to_sklearn,
    obtain_parameter_values,
    get_version_information,
    check_n_jobs,
    is_estimator,
    create_setup_string,
    is_sklearn_flow,
)
from openml.extensions.sklearn.run_functions import (
    seed_model,
    run_model_on_fold,
    is_hpo_class,
    assert_is_hpo_class,
    obtain_arff_trace,
)


if TYPE_CHECKING:
    from openml.flows import OpenMLFlow
    from openml.tasks.task import OpenMLTask
    from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration


class SklearnExtension(Extension):

    ################################################################################################
    # General setup

    @staticmethod
    def can_handle_flow(flow: 'OpenMLFlow') -> bool:
        return is_sklearn_flow(flow)

    @staticmethod
    def can_handle_model(model: Any) -> bool:
        return isinstance(model, sklearn.base.BaseEstimator)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow') -> Any:
        return flow_to_sklearn(flow)

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        return sklearn_to_flow(model)

    def flow_to_parameters(self, flow: Any) -> List:
        return obtain_parameter_values(flow)

    def get_version_information(self) -> List[str]:
        return get_version_information()

    def create_setup_string(self, model: Any) -> str:
        return create_setup_string(model)

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        return is_estimator(model)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        return seed_model(model, seed)

    def run_model_on_fold(
        self,
        model: Any,
        task: 'OpenMLTask',
        rep_no: int,
        fold_no: int,
        sample_no: int,
        can_measure_runtime: bool,
        add_local_measures: bool,
    ) -> Tuple:
        return run_model_on_fold(
            model=model,
            task=task,
            rep_no=rep_no,
            fold_no=fold_no,
            sample_no=sample_no,
            can_measure_runtime=can_measure_runtime,
            add_local_measures=add_local_measures
        )

    def obtain_parameter_values(
        self,
        flow: 'OpenMLFlow',
        model: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts all parameter settings required for the flow from the model.
        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.
        """
        return obtain_parameter_values(flow=flow, model=model)

    def will_model_train_parallel(self, model: Any) -> bool:
        """
        Returns True if the parameter settings of model are chosen s.t. the model
        will run on a single core (if so, openml-python can measure runtimes)
        """
        return check_n_jobs(model)

    ################################################################################################
    # Methods for hyperparameter optimization

    def is_hpo_class(self, model: Any) -> bool:
        return is_hpo_class(model)

    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: 'OpenMLTraceIteration',
    ) -> Any:
        assert_is_hpo_class(model)
        base_estimator = model.estimator
        base_estimator.set_params(**trace_iteration.get_parameters())
        return base_estimator

    def obtain_arff_trace(
        self,
        model: Any,
        trace_content: List,
    ) -> 'OpenMLRunTrace':
        return obtain_arff_trace(model, trace_content)


register_extension(SklearnExtension)

