from abc import ABC, abstractmethod
from collections import OrderedDict  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.flows import OpenMLFlow
    from openml.tasks.task import OpenMLTask
    from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration


class Extension(ABC):

    """Defines the interface to connect machine learning libraries to OpenML-Python.

    See ``openml.extension.sklearn.extension`` for an implementation to bootstrap from.
    """

    ################################################################################################
    # General setup

    @classmethod
    @abstractmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        """Check whether a given flow can be handled by this extension.

        This is typically done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """

    @classmethod
    @abstractmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model flow can be handled by this extension.

        This is typically done by checking the type of the model, or the package it belongs to.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """

    ################################################################################################
    # Abstract methods for flow serialization and de-serialization

    @abstractmethod
    def flow_to_model(self, flow: 'OpenMLFlow') -> Any:
        """Instantiate a model from the flow representation.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        Any
        """

    @abstractmethod
    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform a model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """

    @abstractmethod
    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Returns
        -------
        List
        """

    @abstractmethod
    def create_setup_string(self, model: Any) -> str:
        """Create a string which can be used to reinstantiate the given model.

        Parameters
        ----------
        model : Any

        Returns
        -------
        str
        """

    ################################################################################################
    # Abstract methods for performing runs with extension modules

    @abstractmethod
    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is an estimator for the given extension.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """

    @abstractmethod
    def seed_model(self, model: Any, seed: Optional[int]) -> Any:
        """Set the seed of all the unseeded components of a model and return the seeded model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Parameters
        ----------
        model : Any
            The model to be seeded
        seed : int

        Returns
        -------
        model
        """

    @abstractmethod
    def _run_model_on_fold(
        self,
        model: Any,
        task: 'OpenMLTask',
        rep_no: int,
        fold_no: int,
        sample_no: int,
        add_local_measures: bool,
    ) -> Tuple[List[List], List[List], 'OrderedDict[str, float]', Any]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction information.

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
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
            Determines whether to calculate a set of measures (i.e., predictive accuracy) locally,
            to later verify server behaviour.

        Returns
        -------
        arff_datacontent : List[List]
            Arff representation (list of lists) of the predictions that were
            generated by this fold (required to populate predictions.arff)
        arff_tracecontent :  List[List]
            Arff representation (list of lists) of the trace data that was generated by this fold
            (will be used to populate trace.arff, leave it empty if the model did not perform any
            hyperparameter optimization).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        model : Any
            The model trained on this repeat,fold,subsample triple. Will be used to generate trace
            information later on (in ``obtain_arff_trace``).
        """

    @abstractmethod
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

    ################################################################################################
    # Abstract methods for hyperparameter optimization

    def is_hpo_class(self, model: Any) -> bool:
        """Check whether the model performs hyperparameter optimization.

        Used to check whether an optimization trace can be extracted from the model after running
        it.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """

    @abstractmethod
    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: 'OpenMLTraceIteration',
    ) -> Any:
        """Instantiate a base model which can be searched over by the hyperparameter optimization
        model.

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
        # TODO a trace belongs to a run and therefore a flow -> simplify this part of the interface!

    @abstractmethod
    def obtain_arff_trace(
        self,
        model: Any,
        trace_content: List[List],
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
