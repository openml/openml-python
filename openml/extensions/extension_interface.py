# License: BSD 3-Clause
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse

    from openml.flows import OpenMLFlow
    from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration  # F401
    from openml.tasks.task import OpenMLTask


class Extension(ABC):
    """Defines the interface to connect machine learning libraries to OpenML-Python.

    See ``openml.extension.sklearn.extension`` for an implementation to bootstrap from.
    """

    ################################################################################################
    # General setup

    @classmethod
    @abstractmethod
    def can_handle_flow(cls, flow: OpenMLFlow) -> bool:
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
    def flow_to_model(
        self,
        flow: OpenMLFlow,
        initialize_with_defaults: bool = False,  # noqa: FBT001, FBT002
        strict_version: bool = True,  # noqa: FBT002, FBT001
    ) -> Any:
        """Instantiate a model from the flow representation.

        Parameters
        ----------
        flow : OpenMLFlow

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        strict_version : bool, default=True
            Whether to fail if version requirements are not fulfilled.

        Returns
        -------
        Any
        """

    @abstractmethod
    def model_to_flow(self, model: Any) -> OpenMLFlow:
        """Transform a model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """

    @abstractmethod
    def get_version_information(self) -> list[str]:
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
    def seed_model(self, model: Any, seed: int | None) -> Any:
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
    def _run_model_on_fold(  # noqa: PLR0913
        self,
        model: Any,
        task: OpenMLTask,
        X_train: np.ndarray | scipy.sparse.spmatrix,
        rep_no: int,
        fold_no: int,
        y_train: np.ndarray | None = None,
        X_test: np.ndarray | scipy.sparse.spmatrix | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, OrderedDict[str, float], OpenMLRunTrace | None]:
        """Run a model on a repeat, fold, subsample triplet of the task.

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        :func:`openml.runs.run_flow_on_task`.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
        task : OpenMLTask
            The task to run the model on.
        X_train : array-like
            Training data for the given repetition and fold.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        y_train : Optional[np.ndarray] (default=None)
            Target attributes for supervised tasks. In case of classification, these are integer
            indices to the potential classes specified by dataset.
        X_test : Optional, array-like (default=None)
            Test attributes to test for generalization in supervised tasks.

        Returns
        -------
        predictions : np.ndarray
            Model predictions.
        probabilities :  Optional, np.ndarray
            Predicted probabilities (only applicable for supervised classification tasks).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        trace : Optional, OpenMLRunTrace
            Hyperparameter optimization trace (only applicable for supervised tasks with
            hyperparameter optimization).
        """

    @abstractmethod
    def obtain_parameter_values(
        self,
        flow: OpenMLFlow,
        model: Any = None,
    ) -> list[dict[str, Any]]:
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

    @abstractmethod
    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """

    ################################################################################################
    # Abstract methods for hyperparameter optimization

    @abstractmethod
    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: OpenMLTraceIteration,
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
