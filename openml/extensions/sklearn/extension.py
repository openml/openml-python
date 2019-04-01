from collections import OrderedDict  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import sklearn.base

from openml.extensions import Extension, register_extension
from openml.extensions.sklearn.flow_functions import (
    sklearn_to_flow,
    flow_to_sklearn,
    obtain_parameter_values,
    get_version_information,
    is_estimator,
    create_setup_string,
    is_sklearn_flow,
    is_hpo_class,
)
from openml.extensions.sklearn.run_functions import (
    seed_model,
    run_model_on_fold,
    assert_is_hpo_class,
    obtain_arff_trace,
)


# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.flows import OpenMLFlow
    from openml.tasks.task import OpenMLTask
    from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration


class SklearnExtension(Extension):
    """Connect scikit-learn to OpenML-Python."""

    ################################################################################################
    # General setup

    @staticmethod
    def can_handle_flow(flow: 'OpenMLFlow') -> bool:
        """Check whether a given describes a scikit-learn estimator.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return is_sklearn_flow(flow)

    @staticmethod
    def can_handle_model(model: Any) -> bool:
        """Check whether a model is an instance of ``sklearn.base.BaseEstimator``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, sklearn.base.BaseEstimator)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow') -> Any:
        """Instantiate a scikit-learn model from the flow representation.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        Any
        """
        return flow_to_sklearn(flow)

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform a scikit-learn model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        return sklearn_to_flow(model)

    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Libraries listed are ``Python``, ``scikit-learn``, ``numpy`` and ``scipy``.

        Returns
        -------
        List
        """
        return get_version_information()

    def create_setup_string(self, model: Any) -> str:
        """Create a string which can be used to reinstantiate the given model.

        Parameters
        ----------
        model : Any

        Returns
        -------
        str
        """
        return create_setup_string(model)

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a scikit-learn estimator.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return is_estimator(model)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """Set the random state of all the unseeded components of a model and return the seeded
        model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Models that are already seeded will maintain the seed. In this case,
        only integer seeds are allowed (An exception is raised when a RandomState was used as
        seed).

        Parameters
        ----------
        model : sklearn model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """
        return seed_model(model, seed)

    def _run_model_on_fold(
        self,
        model: Any,
        task: 'OpenMLTask',
        rep_no: int,
        fold_no: int,
        sample_no: int,
        add_local_measures: bool,
    ) -> Tuple[List[List], List[List], 'OrderedDict[str, float]', Any]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are
        doing.

        Parameters
        ----------
        model : Any
            The UNTRAINED scikit-learn model to run. The model instance will be cloned and not
            altered.
        task : OpenMLTask
            The task to run the model on.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        sample_no : int
            In case of learning curves, the index of the subsample (0-based; in case of no
            learning curve, always 0)
        add_local_measures : bool
            Determines whether to calculate a set of measures (i.e., predictive accuracy)
            locally, to later verify server behaviour.

        Returns
        -------
        arff_datacontent : List[List]
            Arff representation (list of lists) of the predictions that were
            generated by this fold (required to populate predictions.arff)
        arff_tracecontent :  List[List]
            Arff representation (list of lists) of the trace data that was generated by
            this fold
            (will be used to populate trace.arff, leave it empty if the model did not
            perform any
            hyperparameter optimization).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        model : Any
            The model trained on this repeat,fold,subsample triplet. Will be used to generate trace
            information later on (in ``obtain_arff_trace``).
        """
        return run_model_on_fold(
            model=model,
            task=task,
            rep_no=rep_no,
            fold_no=fold_no,
            sample_no=sample_no,
            add_local_measures=add_local_measures
        )

    def obtain_parameter_values(
        self,
        flow: 'OpenMLFlow',
        model: Any = None,
    ) -> List[Dict[str, Any]]:
        """Extracts all parameter settings required for the flow from the model.

        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.

        Parameters
        ----------
        flow : OpenMLFlow
            OpenMLFlow object (containing flow ids, i.e., it has to be downloaded from the server)

        model: Any, optional (default=None)
            The model from which to obtain the parameter values. Must match the flow signature.
            If None, use the model specified in ``OpenMLFlow.model``.

        Returns
        -------
        list
            A list of dicts, where each dict has the following entries:
            - ``oml:name`` : str: The OpenML parameter name
            - ``oml:value`` : mixed: A representation of the parameter value
            - ``oml:component`` : int: flow id to which the parameter belongs
        """
        return obtain_parameter_values(flow=flow, model=model)

    ################################################################################################
    # Methods for hyperparameter optimization

    def is_hpo_class(self, model: Any) -> bool:
        """Check whether the model performs hyperparameter optimization.

        Used to check whether an optimization trace can be extracted from the model after
        running it.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return is_hpo_class(model)

    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: 'OpenMLTraceIteration',
    ) -> Any:
        """Instantiate a ``base_estimator`` which can be searched over by the hyperparameter
        optimization model.

        Parameters
        ----------
        model : Any
            A hyperparameter optimization model which defines the model to be instantiated.
        trace_iteration : OpenMLTraceIteration
            Describing the hyperparameter settings to instantiate.

        Returns
        -------
        Any
        """
        assert_is_hpo_class(model)
        base_estimator = model.estimator
        base_estimator.set_params(**trace_iteration.get_parameters())
        return base_estimator

    def obtain_arff_trace(
        self,
        model: Any,
        trace_content: List,
    ) -> 'OpenMLRunTrace':
        """Create arff trace object from a fitted model and the trace content obtained by
        repeatedly calling ``run_model_on_task``.

        Parameters
        ----------
        model : Any
            A fitted hyperparameter optimization model.

        trace_content : List[List]
            Trace content obtained by ``openml.runs.run_flow_on_task``.

        Returns
        -------
        OpenMLRunTrace
        """
        return obtain_arff_trace(model, trace_content)


register_extension(SklearnExtension)
