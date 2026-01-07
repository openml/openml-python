# License: BSD 3-Clause
from __future__ import annotations

import json
import logging
import time
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, cast

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.model_selection
import sklearn.pipeline

import openml
from openml.exceptions import PyOpenMLError
from openml.extensions.base import ModelExecutor
from openml.flows import OpenMLFlow
from openml.runs.trace import PREFIX, OpenMLRunTrace, OpenMLTraceIteration
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
)

if TYPE_CHECKING:
    import scipy.sparse

logger = logging.getLogger(__name__)

SKLEARN_PIPELINE_STRING_COMPONENTS = ("drop", "passthrough")
COMPONENT_REFERENCE = "component_reference"
COMPOSITION_STEP_CONSTANT = "composition_step_constant"


class SklearnExecutor(ModelExecutor):
    """Executor for Scikit-learn estimators."""

    def seed_model(self, model: Any, seed: int | None = None) -> Any:  # noqa: C901
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

        def _seed_current_object(current_value):
            if isinstance(current_value, int):  # acceptable behaviour
                return False

            if isinstance(current_value, np.random.RandomState):
                raise ValueError(
                    "Models initialized with a RandomState object are not "
                    "supported. Please seed with an integer. ",
                )

            if current_value is not None:
                raise ValueError(
                    "Models should be seeded with int or None (this should never happen). ",
                )

            return True

        rs = np.random.RandomState(seed)
        model_params = model.get_params()
        random_states = {}
        for param_name in sorted(model_params):
            if "random_state" in param_name:
                current_value = model_params[param_name]
                # important to draw the value at this point (and not in the if
                # statement) this way we guarantee that if a different set of
                # subflows is seeded, the same number of the random generator is
                # used
                new_value = rs.randint(0, 2**16)
                if _seed_current_object(current_value):
                    random_states[param_name] = new_value

            # Also seed CV objects!
            elif isinstance(model_params[param_name], sklearn.model_selection.BaseCrossValidator):
                if not hasattr(model_params[param_name], "random_state"):
                    continue

                current_value = model_params[param_name].random_state
                new_value = rs.randint(0, 2**16)
                if _seed_current_object(current_value):
                    model_params[param_name].random_state = new_value

        model.set_params(**random_states)
        return model

    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        try:
            # check if model is fitted
            check_is_fitted(model)

            # Creating random dummy data of arbitrary size
            dummy_data = np.random.uniform(size=(10, 3))  # noqa: NPY002
            # Using 'predict' instead of 'sklearn.utils.validation.check_is_fitted' for a more
            # robust check that works across sklearn versions and models. Internally, 'predict'
            # should call 'check_is_fitted' for every concerned attribute, thus offering a more
            # assured check than explicit calls to 'check_is_fitted'
            model.predict(dummy_data)
            # Will reach here if the model was fit on a dataset with 3 features
            return True
        except NotFittedError:  # needs to be the first exception to be caught
            # Model is not fitted, as is required
            return False
        except ValueError:
            # Will reach here if the model was fit on a dataset with more or less than 3 features
            return True

    def _run_model_on_fold(  # noqa: PLR0915, PLR0913, C901, PLR0912
        self,
        model: Any,
        task: OpenMLTask,
        X_train: np.ndarray | scipy.sparse.spmatrix | pd.DataFrame,
        rep_no: int,
        fold_no: int,
        y_train: np.ndarray | None = None,
        X_test: np.ndarray | scipy.sparse.spmatrix | pd.DataFrame | None = None,
    ) -> tuple[
        np.ndarray,
        pd.DataFrame | None,
        OrderedDict[str, float],
        OpenMLRunTrace | None,
    ]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Furthermore, it will measure run time measures in case multi-core behaviour allows this.
        * exact user cpu time will be measured if the number of cores is set (recursive throughout
        the model) exactly to 1
        * wall clock time will be measured if the number of cores is set (recursive throughout the
        model) to any given number (but not when it is set to -1)

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are doing.

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
        pred_y : np.ndarray
            Predictions on the training/test set, depending on the task type.
            For supervised tasks, predictions are on the test set.
            For unsupervised tasks, predictions are on the training set.
        proba_y : pd.DataFrame, optional
            Predicted probabilities for the test set.
            None, if task is not Classification or Learning Curve prediction.
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        trace : OpenMLRunTrace, optional
            arff trace object from a fitted model and the trace content obtained by
            repeatedly calling ``run_model_on_task``
        """

        def _prediction_to_probabilities(
            y: np.ndarray | list,
            model_classes: list[Any],
            class_labels: list[str] | None,
        ) -> pd.DataFrame:
            """Transforms predicted probabilities to match with OpenML class indices.

            Parameters
            ----------
            y : np.ndarray
                Predicted probabilities (possibly omitting classes if they were not present in the
                training data).
            model_classes : list
                List of classes known_predicted by the model, ordered by their index.
            class_labels : list
                List of classes as stored in the task object fetched from server.

            Returns
            -------
            pd.DataFrame
            """
            if class_labels is None:
                raise ValueError("The task has no class labels")

            if isinstance(y_train, np.ndarray) and isinstance(class_labels[0], str):
                # mapping (decoding) the predictions to the categories
                # creating a separate copy to not change the expected pred_y type
                y = [class_labels[pred] for pred in y]  # list or numpy array of predictions

            # model_classes: sklearn classifier mapping from original array id to
            # prediction index id
            if not isinstance(model_classes, list):
                raise ValueError("please convert model classes to list prior to calling this fn")

            # DataFrame allows more accurate mapping of classes as column names
            result = pd.DataFrame(
                0,
                index=np.arange(len(y)),
                columns=model_classes,
                dtype=np.float32,
            )
            for obs, prediction in enumerate(y):
                result.loc[obs, prediction] = 1.0
            return result

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError("argument y_train must not be of type None")
            if X_test is None:
                raise TypeError("argument X_test must not be of type None")

        model_copy = sklearn.base.clone(model, safe=True)
        # sanity check: prohibit users from optimizing n_jobs
        self._prevent_optimize_n_jobs(model_copy)
        # measures and stores runtimes
        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'
        try:
            # for measuring runtime. Only available since Python 3.3
            modelfit_start_cputime = time.process_time()
            modelfit_start_walltime = time.time()

            if isinstance(task, OpenMLSupervisedTask):
                model_copy.fit(X_train, y_train)  # type: ignore
            elif isinstance(task, OpenMLClusteringTask):
                model_copy.fit(X_train)  # type: ignore

            modelfit_dur_cputime = (time.process_time() - modelfit_start_cputime) * 1000
            modelfit_dur_walltime = (time.time() - modelfit_start_walltime) * 1000

            user_defined_measures["usercpu_time_millis_training"] = modelfit_dur_cputime
            refit_time = model_copy.refit_time_ * 1000 if hasattr(model_copy, "refit_time_") else 0  # type: ignore
            user_defined_measures["wall_clock_time_millis_training"] = modelfit_dur_walltime

        except AttributeError as e:
            # typically happens when training a regressor on classification task
            raise PyOpenMLError(str(e)) from e

        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            # search for model classes_ (might differ depending on modeltype)
            # first, pipelines are a special case (these don't have a classes_
            # object, but rather borrows it from the last step. We do this manually,
            # because of the BaseSearch check)
            if isinstance(model_copy, sklearn.pipeline.Pipeline):
                used_estimator = model_copy.steps[-1][-1]
            else:
                used_estimator = model_copy

            if self._is_hpo_class(used_estimator):
                model_classes = used_estimator.best_estimator_.classes_
            else:
                model_classes = used_estimator.classes_

            if not isinstance(model_classes, list):
                model_classes = model_classes.tolist()

            # to handle the case when dataset is numpy and categories are encoded
            # however the class labels stored in task are still categories
            if isinstance(y_train, np.ndarray) and isinstance(
                cast("List", task.class_labels)[0],
                str,
            ):
                model_classes = [cast("List[str]", task.class_labels)[i] for i in model_classes]

        modelpredict_start_cputime = time.process_time()
        modelpredict_start_walltime = time.time()

        # In supervised learning this returns the predictions for Y, in clustering
        # it returns the clusters
        if isinstance(task, OpenMLSupervisedTask):
            pred_y = model_copy.predict(X_test)
        elif isinstance(task, OpenMLClusteringTask):
            pred_y = model_copy.predict(X_train)
        else:
            raise ValueError(task)

        modelpredict_duration_cputime = (time.process_time() - modelpredict_start_cputime) * 1000
        user_defined_measures["usercpu_time_millis_testing"] = modelpredict_duration_cputime
        user_defined_measures["usercpu_time_millis"] = (
            modelfit_dur_cputime + modelpredict_duration_cputime
        )
        modelpredict_duration_walltime = (time.time() - modelpredict_start_walltime) * 1000
        user_defined_measures["wall_clock_time_millis_testing"] = modelpredict_duration_walltime
        user_defined_measures["wall_clock_time_millis"] = (
            modelfit_dur_walltime + modelpredict_duration_walltime + refit_time
        )

        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            try:
                proba_y = model_copy.predict_proba(X_test)
                proba_y = pd.DataFrame(proba_y, columns=model_classes)  # handles X_test as numpy
            except AttributeError:  # predict_proba is not available when probability=False
                proba_y = _prediction_to_probabilities(pred_y, model_classes, task.class_labels)

            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing
                    # at training time. By default, the classification targets
                    # are mapped to be zero-based indices to the actual classes.
                    # Therefore, the model_classes contain the correct indices to
                    # the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities
                    # for class 3 because the rest of the library expects that the
                    # probabilities are ordered the same way as the classes are ordered).
                    message = (
                        f"Estimator only predicted for {proba_y.shape[1]}/{len(task.class_labels)}"
                        " classes!"
                    )
                    warnings.warn(message, stacklevel=2)
                    openml.config.logger.warning(message)

                    for _i, col in enumerate(task.class_labels):
                        # adding missing columns with 0 probability
                        if col not in model_classes:
                            proba_y[col] = 0
                    # We re-order the columns to move possibly added missing columns into place.
                    proba_y = proba_y[task.class_labels]
            else:
                raise ValueError("The task has no class labels")

            if not np.all(set(proba_y.columns) == set(task.class_labels)):
                missing_cols = list(set(task.class_labels) - set(proba_y.columns))
                raise ValueError("Predicted probabilities missing for the columns: ", missing_cols)

        elif isinstance(task, (OpenMLRegressionTask, OpenMLClusteringTask)):
            proba_y = None
        else:
            raise TypeError(type(task))

        if self._is_hpo_class(model_copy):
            trace_data = self._extract_trace_data(model_copy, rep_no, fold_no)
            trace: OpenMLRunTrace | None = self._obtain_arff_trace(
                model_copy,
                trace_data,
            )
        else:
            trace = None

        return pred_y, proba_y, user_defined_measures, trace

    def obtain_parameter_values(  # noqa: C901, PLR0915
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
        openml.flows.functions._check_flow_for_server_id(flow)

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(  # noqa: PLR0915, PLR0912, C901
            _flow,
            _flow_dict,
            component_model,
            _main_call=False,  # noqa: FBT002
            main_id=None,
        ):
            def is_subcomponent_specification(values):
                # checks whether the current value can be a specification of
                # subcomponents, as for example the value for steps parameter
                # (in Pipeline) or transformers parameter (in
                # ColumnTransformer).
                return (
                    # Specification requires list/tuple of list/tuple with
                    # at least length 2.
                    isinstance(values, (tuple, list))
                    and all(isinstance(item, (tuple, list)) and len(item) > 1 for item in values)
                    # And each component needs to be a flow or interpretable string
                    and all(
                        isinstance(item[1], openml.flows.OpenMLFlow)
                        or (
                            isinstance(item[1], str)
                            and item[1] in SKLEARN_PIPELINE_STRING_COMPONENTS
                        )
                        for item in values
                    )
                )

            # _flow is openml flow object, _param dict maps from flow name to flow
            # id for the main call, the param dict can be overridden (useful for
            # unit tests / sentinels) this way, for flows without subflows we do
            # not have to rely on _flow_dict
            exp_parameters = set(_flow.parameters)
            if (
                isinstance(component_model, str)
                and component_model in SKLEARN_PIPELINE_STRING_COMPONENTS
            ):
                model_parameters = set()
            else:
                model_parameters = set(component_model.get_params(deep=False))
            if len(exp_parameters.symmetric_difference(model_parameters)) != 0:
                flow_params = sorted(exp_parameters)
                model_params = sorted(model_parameters)
                raise ValueError(
                    "Parameters of the model do not match the "
                    "parameters expected by the "
                    "flow:\nexpected flow parameters: "
                    f"{flow_params}\nmodel parameters: {model_params}",
                )
            exp_components = set(_flow.components)
            if (
                isinstance(component_model, str)
                and component_model in SKLEARN_PIPELINE_STRING_COMPONENTS
            ):
                model_components = set()
            else:
                _ = set(component_model.get_params(deep=False))
                model_components = {
                    mp
                    for mp in component_model.get_params(deep=True)
                    if "__" not in mp and mp not in _
                }
            if len(exp_components.symmetric_difference(model_components)) != 0:
                is_problem = True
                if len(exp_components - model_components) > 0:
                    # If an expected component is not returned as a component by get_params(),
                    # this means that it is also a parameter -> we need to check that this is
                    # actually the case
                    difference = exp_components - model_components
                    component_in_model_parameters = []
                    for component in difference:
                        if component in model_parameters:
                            component_in_model_parameters.append(True)
                        else:
                            component_in_model_parameters.append(False)
                    is_problem = not all(component_in_model_parameters)
                if is_problem:
                    flow_components = sorted(exp_components)
                    model_components = sorted(model_components)
                    raise ValueError(
                        "Subcomponents of the model do not match the "
                        "parameters expected by the "
                        "flow:\nexpected flow subcomponents: "
                        f"{flow_components}\nmodel subcomponents: {model_components}",
                    )

            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current["oml:name"] = _param_name

                current_param_values = self.model_to_flow(component_model.get_params()[_param_name])

                # Try to filter out components (a.k.a. subflows) which are
                # handled further down in the code (by recursively calling
                # this function)!
                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                if is_subcomponent_specification(current_param_values):
                    # complex parameter value, with subcomponents
                    parsed_values = []
                    for subcomponent in current_param_values:
                        # scikit-learn stores usually tuples in the form
                        # (name (str), subcomponent (mixed), argument
                        # (mixed)). OpenML replaces the subcomponent by an
                        # OpenMLFlow object.
                        if len(subcomponent) < 2 or len(subcomponent) > 3:
                            raise ValueError("Component reference should be size {2,3}. ")

                        subcomponent_identifier = subcomponent[0]
                        subcomponent_flow = subcomponent[1]
                        if not isinstance(subcomponent_identifier, str):
                            raise TypeError(
                                "Subcomponent identifier should be of type string, "
                                f"but is {type(subcomponent_identifier)}",
                            )
                        if not isinstance(subcomponent_flow, (openml.flows.OpenMLFlow, str)):
                            if (
                                isinstance(subcomponent_flow, str)
                                and subcomponent_flow in SKLEARN_PIPELINE_STRING_COMPONENTS
                            ):
                                pass
                            else:
                                raise TypeError(
                                    "Subcomponent flow should be of type flow, but is"
                                    f" {type(subcomponent_flow)}",
                                )

                        current = {
                            "oml-python:serialized_object": COMPONENT_REFERENCE,
                            "value": {
                                "key": subcomponent_identifier,
                                "step_name": subcomponent_identifier,
                            },
                        }
                        if len(subcomponent) == 3:
                            if not isinstance(subcomponent[2], list) and not isinstance(
                                subcomponent[2],
                                OrderedDict,
                            ):
                                raise TypeError(
                                    "Subcomponent argument should be list or OrderedDict",
                                )
                            current["value"]["argument_1"] = subcomponent[2]
                        parsed_values.append(current)
                    parsed_values = json.dumps(parsed_values)
                else:
                    # vanilla parameter value
                    parsed_values = json.dumps(current_param_values)

                _current["oml:value"] = parsed_values
                if _main_call:
                    _current["oml:component"] = main_id
                else:
                    _current["oml:component"] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = component_model.get_params()[_identifier]
                _params.extend(
                    extract_parameters(
                        _flow.components[_identifier],
                        _flow_dict,
                        subcomponent_model,
                    ),
                )
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        return extract_parameters(flow, flow_dict, model, _main_call=True, main_id=flow.flow_id)

    def _openml_param_name_to_sklearn(
        self,
        openml_parameter: openml.setups.OpenMLParameter,
        flow: OpenMLFlow,
    ) -> str:
        """
        Converts the name of an OpenMLParameter into the sklean name, given a flow.

        Parameters
        ----------
        openml_parameter: OpenMLParameter
            The parameter under consideration

        flow: OpenMLFlow
            The flow that provides context.

        Returns
        -------
        sklearn_parameter_name: str
            The name the parameter will have once used in scikit-learn
        """
        if not isinstance(openml_parameter, openml.setups.OpenMLParameter):
            raise ValueError("openml_parameter should be an instance of OpenMLParameter")
        if not isinstance(flow, OpenMLFlow):
            raise ValueError("flow should be an instance of OpenMLFlow")

        flow_structure = flow.get_structure("name")
        if openml_parameter.flow_name not in flow_structure:
            raise ValueError("Obtained OpenMLParameter and OpenMLFlow do not correspond. ")
        name = openml_parameter.flow_name  # for PEP8
        return "__".join(flow_structure[name] + [openml_parameter.parameter_name])

    ################################################################################################
    # Methods for hyperparameter optimization

    def _is_hpo_class(self, model: Any) -> bool:
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
        return isinstance(model, sklearn.model_selection._search.BaseSearchCV)

    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: OpenMLTraceIteration,
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
        if not self._is_hpo_class(model):
            raise AssertionError(
                f"Flow model {model} is not an instance of"
                " sklearn.model_selection._search.BaseSearchCV",
            )
        base_estimator = model.estimator
        base_estimator.set_params(**trace_iteration.get_parameters())
        return base_estimator

    def _extract_trace_data(self, model, rep_no, fold_no):
        """Extracts data from a machine learning model's cross-validation results
        and creates an ARFF (Attribute-Relation File Format) trace.

        Parameters
        ----------
        model : Any
            A fitted hyperparameter optimization model.
        rep_no : int
            The repetition number.
        fold_no : int
            The fold number.

        Returns
        -------
        A list of ARFF tracecontent.
        """
        arff_tracecontent = []
        for itt_no in range(len(model.cv_results_["mean_test_score"])):
            # we use the string values for True and False, as it is defined in
            # this way by the OpenML server
            selected = "false"
            if itt_no == model.best_index_:
                selected = "true"
            test_score = model.cv_results_["mean_test_score"][itt_no]
            arff_line = [rep_no, fold_no, itt_no, test_score, selected]
            for key in model.cv_results_:
                if key.startswith("param_"):
                    value = model.cv_results_[key][itt_no]
                    # Built-in serializer does not convert all numpy types,
                    # these methods convert them to built-in types instead.
                    if isinstance(value, np.generic):
                        # For scalars it actually returns scalars, not a list
                        value = value.tolist()
                    serialized_value = json.dumps(value) if value is not np.ma.masked else np.nan
                    arff_line.append(serialized_value)
            arff_tracecontent.append(arff_line)
        return arff_tracecontent

    def _obtain_arff_trace(
        self,
        model: Any,
        trace_content: list,
    ) -> OpenMLRunTrace:
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
        if not self._is_hpo_class(model):
            raise AssertionError(
                f"Flow model {model} is not an instance of "
                "sklearn.model_selection._search.BaseSearchCV",
            )
        if not hasattr(model, "cv_results_"):
            raise ValueError("model should contain `cv_results_`")

        # attributes that will be in trace arff, regardless of the model
        trace_attributes = [
            ("repeat", "NUMERIC"),
            ("fold", "NUMERIC"),
            ("iteration", "NUMERIC"),
            ("evaluation", "NUMERIC"),
            ("selected", ["true", "false"]),
        ]

        # model dependent attributes for trace arff
        for key in model.cv_results_:
            if key.startswith("param_"):
                # supported types should include all types, including bool,
                # int float
                supported_basic_types = (bool, int, float, str)
                for param_value in model.cv_results_[key]:
                    if isinstance(param_value, np.generic):
                        param_value = param_value.tolist()  # noqa: PLW2901
                    if (
                        isinstance(param_value, supported_basic_types)
                        or param_value is None
                        or param_value is np.ma.masked
                    ):
                        # basic string values
                        type = "STRING"  # noqa: A001
                    elif isinstance(param_value, (list, tuple)) and all(
                        isinstance(i, int) for i in param_value
                    ):
                        # list of integers (usually for selecting features)
                        # hyperparameter layer_sizes of MLPClassifier
                        type = "STRING"  # noqa: A001
                    else:
                        raise TypeError(f"Unsupported param type in param grid: {key}")

                # renamed the attribute param to parameter, as this is a required
                # OpenML convention - this also guards against name collisions
                # with the required trace attributes
                attribute = (PREFIX + key[6:], type)  # type: ignore
                trace_attributes.append(attribute)

        return OpenMLRunTrace.generate(
            trace_attributes,
            trace_content,
        )
