# License: BSD 3-Clause

"""Base class for estimator executors."""

from abc import ABC, abstractmethod
from typing import Any
from collections import OrderedDict

import numpy as np
import scipy.sparse

from openml.tasks.task import OpenMLTask
from openml.runs.trace import OpenMLRunTrace

class ModelExecutor(ABC):
    """Define runtime execution semantics for a specific API type."""

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
    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """