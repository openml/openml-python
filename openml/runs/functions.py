# License: BSD 3-Clause
from __future__ import annotations

import itertools
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal

import numpy as np
import pandas as pd
import sklearn.metrics
import xmltodict
from joblib.parallel import Parallel, delayed

import openml
import openml._api_calls
import openml.utils
from openml import config
from openml.exceptions import (
    OpenMLCacheException,
    OpenMLRunsExistError,
    OpenMLServerException,
    PyOpenMLError,
)
from openml.extensions import get_extension_by_model
from openml.flows import OpenMLFlow, flow_exists, get_flow
from openml.flows.flow import _copy_server_fields
from openml.setups import initialize_model, setup_exists
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
    TaskType,
    get_task,
)

from .run import OpenMLRun
from .trace import OpenMLRunTrace

# Avoid import cycles: https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from openml.config import _Config
    from openml.extensions.extension_interface import Extension

# get_dict is in run.py to avoid circular imports

RUNS_CACHE_DIR_NAME = "runs"
ERROR_CODE = 512


# TODO(eddiebergman): Could potentially overload this but
# it seems very big to do so
def run_model_on_task(  # noqa: PLR0913
    model: Any,
    task: int | str | OpenMLTask,
    avoid_duplicate_runs: bool = True,  # noqa: FBT001, FBT002
    flow_tags: list[str] | None = None,
    seed: int | None = None,
    add_local_measures: bool = True,  # noqa: FBT001, FBT002
    upload_flow: bool = False,  # noqa: FBT001, FBT002
    return_flow: bool = False,  # noqa: FBT001, FBT002
    dataset_format: Literal["array", "dataframe"] = "dataframe",
    n_jobs: int | None = None,
) -> OpenMLRun | tuple[OpenMLRun, OpenMLFlow]:
    """Run the model on the dataset defined by the task.

    Parameters
    ----------
    model : sklearn model
        A model which has a function fit(X,Y) and predict(X),
        all supervised estimators of scikit learn follow this definition of a model.
    task : OpenMLTask or int or str
        Task to perform or Task id.
        This may be a model instead if the first argument is an OpenMLTask.
    avoid_duplicate_runs : bool, optional (default=True)
        If True, the run will throw an error if the setup/task combination is already present on
        the server. This feature requires an internet connection.
    flow_tags : List[str], optional (default=None)
        A list of tags that the flow should have at creation.
    seed: int, optional (default=None)
        Models that are not seeded will get this seed.
    add_local_measures : bool, optional (default=True)
        Determines whether to calculate a set of evaluation measures locally,
        to later verify server behaviour.
    upload_flow : bool (default=False)
        If True, upload the flow to OpenML if it does not exist yet.
        If False, do not upload the flow to OpenML.
    return_flow : bool (default=False)
        If True, returns the OpenMLFlow generated from the model in addition to the OpenMLRun.
    dataset_format : str (default='dataframe')
        If 'array', the dataset is passed to the model as a numpy array.
        If 'dataframe', the dataset is passed to the model as a pandas dataframe.
    n_jobs : int (default=None)
        The number of processes/threads to distribute the evaluation asynchronously.
        If `None` or `1`, then the evaluation is treated as synchronous and processed sequentially.
        If `-1`, then the job uses as many cores available.

    Returns
    -------
    run : OpenMLRun
        Result of the run.
    flow : OpenMLFlow (optional, only if `return_flow` is True).
        Flow generated from the model.
    """
    if avoid_duplicate_runs and not config.apikey:
        warnings.warn(
            "avoid_duplicate_runs is set to True, but no API key is set. "
            "Please set your API key in the OpenML configuration file, see"
            "https://openml.github.io/openml-python/main/examples/20_basic/introduction_tutorial"
            ".html#authentication for more information on authentication.",
            RuntimeWarning,
            stacklevel=2,
        )

    # TODO: At some point in the future do not allow for arguments in old order (6-2018).
    # Flexibility currently still allowed due to code-snippet in OpenML100 paper (3-2019).
    # When removing this please also remove the method `is_estimator` from the extension
    # interface as it is only used here (MF, 3-2019)
    if isinstance(model, (int, str, OpenMLTask)):
        warnings.warn(
            "The old argument order (task, model) is deprecated and "
            "will not be supported in the future. Please use the "
            "order (model, task).",
            DeprecationWarning,
            stacklevel=2,
        )
        task, model = model, task

    extension = get_extension_by_model(model, raise_if_no_extension=True)
    if extension is None:
        # This should never happen and is only here to please mypy will be gone soon once the
        # whole function is removed
        raise TypeError(extension)

    flow = extension.model_to_flow(model)

    def get_task_and_type_conversion(_task: int | str | OpenMLTask) -> OpenMLTask:
        """Retrieve an OpenMLTask object from either an integer or string ID,
        or directly from an OpenMLTask object.

        Parameters
        ----------
        _task : Union[int, str, OpenMLTask]
            The task ID or the OpenMLTask object.

        Returns
        -------
        OpenMLTask
            The OpenMLTask object.
        """
        if isinstance(_task, (int, str)):
            return get_task(int(_task))  # type: ignore

        return _task

    task = get_task_and_type_conversion(task)

    run = run_flow_on_task(
        task=task,
        flow=flow,
        avoid_duplicate_runs=avoid_duplicate_runs,
        flow_tags=flow_tags,
        seed=seed,
        add_local_measures=add_local_measures,
        upload_flow=upload_flow,
        dataset_format=dataset_format,
        n_jobs=n_jobs,
    )
    if return_flow:
        return run, flow
    return run


def run_flow_on_task(  # noqa: C901, PLR0912, PLR0915, PLR0913
    flow: OpenMLFlow,
    task: OpenMLTask,
    avoid_duplicate_runs: bool = True,  # noqa: FBT002, FBT001
    flow_tags: list[str] | None = None,
    seed: int | None = None,
    add_local_measures: bool = True,  # noqa: FBT001, FBT002
    upload_flow: bool = False,  # noqa: FBT001, FBT002
    dataset_format: Literal["array", "dataframe"] = "dataframe",
    n_jobs: int | None = None,
) -> OpenMLRun:
    """Run the model provided by the flow on the dataset defined by task.

    Takes the flow and repeat information into account.
    The Flow may optionally be published.

    Parameters
    ----------
    flow : OpenMLFlow
        A flow wraps a machine learning model together with relevant information.
        The model has a function fit(X,Y) and predict(X),
        all supervised estimators of scikit learn follow this definition of a model.
    task : OpenMLTask
        Task to perform. This may be an OpenMLFlow instead if the first argument is an OpenMLTask.
    avoid_duplicate_runs : bool, optional (default=True)
        If True, the run will throw an error if the setup/task combination is already present on
        the server. This feature requires an internet connection.
    flow_tags : List[str], optional (default=None)
        A list of tags that the flow should have at creation.
    seed: int, optional (default=None)
        Models that are not seeded will get this seed.
    add_local_measures : bool, optional (default=True)
        Determines whether to calculate a set of evaluation measures locally,
        to later verify server behaviour.
    upload_flow : bool (default=False)
        If True, upload the flow to OpenML if it does not exist yet.
        If False, do not upload the flow to OpenML.
    dataset_format : str (default='dataframe')
        If 'array', the dataset is passed to the model as a numpy array.
        If 'dataframe', the dataset is passed to the model as a pandas dataframe.
    n_jobs : int (default=None)
        The number of processes/threads to distribute the evaluation asynchronously.
        If `None` or `1`, then the evaluation is treated as synchronous and processed sequentially.
        If `-1`, then the job uses as many cores available.

    Returns
    -------
    run : OpenMLRun
        Result of the run.
    """
    if flow_tags is not None and not isinstance(flow_tags, list):
        raise ValueError("flow_tags should be a list")

    # TODO: At some point in the future do not allow for arguments in old order (changed 6-2018).
    # Flexibility currently still allowed due to code-snippet in OpenML100 paper (3-2019).
    if isinstance(flow, OpenMLTask) and isinstance(task, OpenMLFlow):
        # We want to allow either order of argument (to avoid confusion).
        warnings.warn(
            "The old argument order (Flow, model) is deprecated and "
            "will not be supported in the future. Please use the "
            "order (model, Flow).",
            DeprecationWarning,
            stacklevel=2,
        )
        task, flow = flow, task

    if task.task_id is None:
        raise ValueError("The task should be published at OpenML")

    if flow.model is None:
        flow.model = flow.extension.flow_to_model(flow)

    flow.model = flow.extension.seed_model(flow.model, seed=seed)

    # We only need to sync with the server right now if we want to upload the flow,
    # or ensure no duplicate runs exist. Otherwise it can be synced at upload time.
    flow_id = None
    if upload_flow or avoid_duplicate_runs:
        flow_id = flow_exists(flow.name, flow.external_version)
        if isinstance(flow.flow_id, int) and flow_id != flow.flow_id:
            if flow_id is not False:
                raise PyOpenMLError(
                    "Local flow_id does not match server flow_id: "
                    f"'{flow.flow_id}' vs '{flow_id}'",
                )
            raise PyOpenMLError(
                "Flow does not exist on the server, but 'flow.flow_id' is not None."
            )
        if upload_flow and flow_id is False:
            flow.publish()
            flow_id = flow.flow_id
        elif flow_id:
            flow_from_server = get_flow(flow_id)
            _copy_server_fields(flow_from_server, flow)
            if avoid_duplicate_runs:
                flow_from_server.model = flow.model
                setup_id = setup_exists(flow_from_server)
                ids = run_exists(task.task_id, setup_id)
                if ids:
                    error_message = (
                        "One or more runs of this setup were already performed on the task."
                    )
                    raise OpenMLRunsExistError(ids, error_message)
        else:
            # Flow does not exist on server and we do not want to upload it.
            # No sync with the server happens.
            flow_id = None

    dataset = task.get_dataset()

    run_environment = flow.extension.get_version_information()
    tags = ["openml-python", run_environment[1]]

    if flow.extension.check_if_model_fitted(flow.model):
        warnings.warn(
            "The model is already fitted!"
            " This might cause inconsistency in comparison of results.",
            RuntimeWarning,
            stacklevel=2,
        )

    # execute the run
    res = _run_task_get_arffcontent(
        model=flow.model,
        task=task,
        extension=flow.extension,
        add_local_measures=add_local_measures,
        dataset_format=dataset_format,
        n_jobs=n_jobs,
    )

    data_content, trace, fold_evaluations, sample_evaluations = res
    fields = [*run_environment, time.strftime("%c"), "Created by run_flow_on_task"]
    generated_description = "\n".join(fields)
    run = OpenMLRun(
        task_id=task.task_id,
        flow_id=flow_id,
        dataset_id=dataset.dataset_id,
        model=flow.model,
        flow_name=flow.name,
        tags=tags,
        trace=trace,
        data_content=data_content,
        flow=flow,
        setup_string=flow.extension.create_setup_string(flow.model),
        description_text=generated_description,
    )

    if (upload_flow or avoid_duplicate_runs) and flow.flow_id is not None:
        # We only extract the parameter settings if a sync happened with the server.
        # I.e. when the flow was uploaded or we found it in the avoid_duplicate check.
        # Otherwise, we will do this at upload time.
        run.parameter_settings = flow.extension.obtain_parameter_values(flow)

    # now we need to attach the detailed evaluations
    if task.task_type_id == TaskType.LEARNING_CURVE:
        run.sample_evaluations = sample_evaluations
    else:
        run.fold_evaluations = fold_evaluations

    if flow_id:
        message = f"Executed Task {task.task_id} with Flow id:{run.flow_id}"
    else:
        message = f"Executed Task {task.task_id} on local Flow with name {flow.name}."
    config.logger.info(message)

    return run


def get_run_trace(run_id: int) -> OpenMLRunTrace:
    """
    Get the optimization trace object for a given run id.

    Parameters
    ----------
    run_id : int

    Returns
    -------
    openml.runs.OpenMLTrace
    """
    trace_xml = openml._api_calls._perform_api_call("run/trace/%d" % run_id, "get")
    return OpenMLRunTrace.trace_from_xml(trace_xml)


def initialize_model_from_run(run_id: int, *, strict_version: bool = True) -> Any:
    """
    Initialized a model based on a run_id (i.e., using the exact
    same parameter settings)

    Parameters
    ----------
    run_id : int
        The Openml run_id
    strict_version: bool (default=True)
        See `flow_to_model` strict_version.

    Returns
    -------
    model
    """
    run = get_run(run_id)
    # TODO(eddiebergman): I imagine this is None if it's not published,
    # might need to raise an explicit error for that
    assert run.setup_id is not None
    return initialize_model(setup_id=run.setup_id, strict_version=strict_version)


def initialize_model_from_trace(
    run_id: int,
    repeat: int,
    fold: int,
    iteration: int | None = None,
) -> Any:
    """
    Initialize a model based on the parameters that were set
    by an optimization procedure (i.e., using the exact same
    parameter settings)

    Parameters
    ----------
    run_id : int
        The Openml run_id. Should contain a trace file,
        otherwise a OpenMLServerException is raised

    repeat : int
        The repeat nr (column in trace file)

    fold : int
        The fold nr (column in trace file)

    iteration : int
        The iteration nr (column in trace file). If None, the
        best (selected) iteration will be searched (slow),
        according to the selection criteria implemented in
        OpenMLRunTrace.get_selected_iteration

    Returns
    -------
    model
    """
    run = get_run(run_id)
    # TODO(eddiebergman): I imagine this is None if it's not published,
    # might need to raise an explicit error for that
    assert run.flow_id is not None

    flow = get_flow(run.flow_id)
    run_trace = get_run_trace(run_id)

    if iteration is None:
        iteration = run_trace.get_selected_iteration(repeat, fold)

    request = (repeat, fold, iteration)
    if request not in run_trace.trace_iterations:
        raise ValueError("Combination repeat, fold, iteration not available")
    current = run_trace.trace_iterations[(repeat, fold, iteration)]

    search_model = initialize_model_from_run(run_id)
    return flow.extension.instantiate_model_from_hpo_class(search_model, current)


def run_exists(task_id: int, setup_id: int) -> set[int]:
    """Checks whether a task/setup combination is already present on the
    server.

    Parameters
    ----------
    task_id : int

    setup_id : int

    Returns
    -------
        Set run ids for runs where flow setup_id was run on task_id. Empty
        set if it wasn't run yet.
    """
    if setup_id <= 0:
        # openml setups are in range 1-inf
        return set()

    try:
        result = list_runs(task=[task_id], setup=[setup_id], output_format="dataframe")
        assert isinstance(result, pd.DataFrame)  # TODO(eddiebergman): Remove once #1299
        return set() if result.empty else set(result["run_id"])
    except OpenMLServerException as exception:
        # error code implies no results. The run does not exist yet
        if exception.code != ERROR_CODE:
            raise exception
        return set()


def _run_task_get_arffcontent(  # noqa: PLR0915, PLR0912, PLR0913, C901
    *,
    model: Any,
    task: OpenMLTask,
    extension: Extension,
    add_local_measures: bool,
    dataset_format: Literal["array", "dataframe"],
    n_jobs: int | None = None,
) -> tuple[
    list[list],
    OpenMLRunTrace | None,
    OrderedDict[str, OrderedDict],
    OrderedDict[str, OrderedDict],
]:
    """Runs the hyperparameter optimization on the given task
    and returns the arfftrace content.

    Parameters
    ----------
    model : Any
        The model that is to be evalauted.
    task : OpenMLTask
        The OpenMLTask to evaluate.
    extension : Extension
        The OpenML extension object.
    add_local_measures : bool
        Whether to compute additional local evaluation measures.
    dataset_format : str
        The format in which to download the dataset.
    n_jobs : int
        Number of jobs to run in parallel.
        If None, use 1 core by default. If -1, use all available cores.

    Returns
    -------
    Tuple[List[List], Optional[OpenMLRunTrace],
        OrderedDict[str, OrderedDict], OrderedDict[str, OrderedDict]]
    A tuple containing the arfftrace content,
    the OpenML run trace, the global and local evaluation measures.
    """
    arff_datacontent = []  # type: list[list]
    traces = []  # type: list[OpenMLRunTrace]
    # stores fold-based evaluation measures. In case of a sample based task,
    # this information is multiple times overwritten, but due to the ordering
    # of tne loops, eventually it contains the information based on the full
    # dataset size
    user_defined_measures_per_fold = OrderedDict()  # type: 'OrderedDict[str, OrderedDict]'
    # stores sample-based evaluation measures (sublevel of fold-based)
    # will also be filled on a non sample-based task, but the information
    # is the same as the fold-based measures, and disregarded in that case
    user_defined_measures_per_sample = OrderedDict()  # type: 'OrderedDict[str, OrderedDict]'

    # TODO use different iterator to only provide a single iterator (less
    # methods, less maintenance, less confusion)
    num_reps, num_folds, num_samples = task.get_split_dimensions()

    jobs = []
    for n_fit, (rep_no, fold_no, sample_no) in enumerate(
        itertools.product(
            range(num_reps),
            range(num_folds),
            range(num_samples),
        ),
        start=1,
    ):
        jobs.append((n_fit, rep_no, fold_no, sample_no))

    # The forked child process may not copy the configuration state of OpenML from the parent.
    # Current configuration setup needs to be copied and passed to the child processes.
    _config = config.get_config_as_dict()
    # Execute runs in parallel
    # assuming the same number of tasks as workers (n_jobs), the total compute time for this
    # statement will be similar to the slowest run
    # TODO(eddiebergman): Simplify this
    job_rvals: list[
        tuple[
            np.ndarray,
            pd.DataFrame | None,
            np.ndarray,
            pd.DataFrame | None,
            OpenMLRunTrace | None,
            OrderedDict[str, float],
        ],
    ]
    job_rvals = Parallel(verbose=0, n_jobs=n_jobs)(  # type: ignore
        delayed(_run_task_get_arffcontent_parallel_helper)(
            extension=extension,
            fold_no=fold_no,
            model=model,
            rep_no=rep_no,
            sample_no=sample_no,
            task=task,
            dataset_format=dataset_format,
            configuration=_config,
        )
        for _n_fit, rep_no, fold_no, sample_no in jobs
    )  # job_rvals contain the output of all the runs with one-to-one correspondence with `jobs`

    for n_fit, rep_no, fold_no, sample_no in jobs:
        pred_y, proba_y, test_indices, test_y, inner_trace, user_defined_measures_fold = job_rvals[
            n_fit - 1
        ]

        if inner_trace is not None:
            traces.append(inner_trace)

        # add client-side calculated metrics. These is used on the server as
        # consistency check, only useful for supervised tasks
        def _calculate_local_measure(  # type: ignore
            sklearn_fn,
            openml_name,
            _test_y=test_y,
            _pred_y=pred_y,
            _user_defined_measures_fold=user_defined_measures_fold,
        ):
            _user_defined_measures_fold[openml_name] = sklearn_fn(_test_y, _pred_y)

        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            assert test_y is not None
            assert proba_y is not None

            for i, tst_idx in enumerate(test_indices):
                if task.class_labels is not None:
                    prediction = (
                        task.class_labels[pred_y[i]]
                        if isinstance(pred_y[i], (int, np.integer))
                        else pred_y[i]
                    )
                    if isinstance(test_y, pd.Series):
                        truth = (
                            task.class_labels[test_y.iloc[i]]
                            if isinstance(test_y.iloc[i], int)
                            else test_y.iloc[i]
                        )
                    else:
                        truth = (
                            task.class_labels[test_y[i]]
                            if isinstance(test_y[i], (int, np.integer))
                            else test_y[i]
                        )
                    pred_prob = proba_y.iloc[i] if isinstance(proba_y, pd.DataFrame) else proba_y[i]

                    arff_line = format_prediction(
                        task=task,
                        repeat=rep_no,
                        fold=fold_no,
                        sample=sample_no,
                        index=tst_idx,
                        prediction=prediction,
                        truth=truth,
                        proba=dict(zip(task.class_labels, pred_prob)),
                    )
                else:
                    raise ValueError("The task has no class labels")

                arff_datacontent.append(arff_line)

            if add_local_measures:
                _calculate_local_measure(
                    sklearn.metrics.accuracy_score,
                    "predictive_accuracy",
                )

        elif isinstance(task, OpenMLRegressionTask):
            assert test_y is not None
            for i, _ in enumerate(test_indices):
                truth = test_y.iloc[i] if isinstance(test_y, pd.Series) else test_y[i]
                arff_line = format_prediction(
                    task=task,
                    repeat=rep_no,
                    fold=fold_no,
                    index=test_indices[i],
                    prediction=pred_y[i],
                    truth=truth,
                )

                arff_datacontent.append(arff_line)

            if add_local_measures:
                _calculate_local_measure(
                    sklearn.metrics.mean_absolute_error,
                    "mean_absolute_error",
                )

        elif isinstance(task, OpenMLClusteringTask):
            for i, _ in enumerate(test_indices):
                arff_line = [test_indices[i], pred_y[i]]  # row_id, cluster ID
                arff_datacontent.append(arff_line)

        else:
            raise TypeError(type(task))

        for measure in user_defined_measures_fold:
            if measure not in user_defined_measures_per_fold:
                user_defined_measures_per_fold[measure] = OrderedDict()
            if rep_no not in user_defined_measures_per_fold[measure]:
                user_defined_measures_per_fold[measure][rep_no] = OrderedDict()

            if measure not in user_defined_measures_per_sample:
                user_defined_measures_per_sample[measure] = OrderedDict()
            if rep_no not in user_defined_measures_per_sample[measure]:
                user_defined_measures_per_sample[measure][rep_no] = OrderedDict()
            if fold_no not in user_defined_measures_per_sample[measure][rep_no]:
                user_defined_measures_per_sample[measure][rep_no][fold_no] = OrderedDict()

            user_defined_measures_per_fold[measure][rep_no][fold_no] = user_defined_measures_fold[
                measure
            ]
            user_defined_measures_per_sample[measure][rep_no][fold_no][sample_no] = (
                user_defined_measures_fold[measure]
            )

    trace: OpenMLRunTrace | None = None
    if len(traces) > 0:
        if len(traces) != len(jobs):
            raise ValueError(
                f"Did not find enough traces (expected {len(jobs)}, found {len(traces)})",
            )

        trace = OpenMLRunTrace.merge_traces(traces)

    return (
        arff_datacontent,
        trace,
        user_defined_measures_per_fold,
        user_defined_measures_per_sample,
    )


def _run_task_get_arffcontent_parallel_helper(  # noqa: PLR0913
    extension: Extension,
    fold_no: int,
    model: Any,
    rep_no: int,
    sample_no: int,
    task: OpenMLTask,
    dataset_format: Literal["array", "dataframe"],
    configuration: _Config | None = None,
) -> tuple[
    np.ndarray,
    pd.DataFrame | None,
    np.ndarray,
    pd.DataFrame | None,
    OpenMLRunTrace | None,
    OrderedDict[str, float],
]:
    """Helper function that runs a single model on a single task fold sample.

    Parameters
    ----------
    extension : Extension
        An OpenML extension instance.
    fold_no : int
        The fold number to be run.
    model : Any
        The model that is to be evaluated.
    rep_no : int
        Repetition number to be run.
    sample_no : int
        Sample number to be run.
    task : OpenMLTask
        The task object from OpenML.
    dataset_format : str
        The dataset format to be used.
    configuration : _Config
        Hyperparameters to configure the model.

    Returns
    -------
    Tuple[np.ndarray, Optional[pd.DataFrame], np.ndarray, Optional[pd.DataFrame],
           Optional[OpenMLRunTrace], OrderedDict[str, float]]
    A tuple containing the predictions, probability estimates (if applicable),
    actual target values, actual target value probabilities (if applicable),
    the trace object of the OpenML run (if applicable),
    and a dictionary of local measures for this particular fold.
    """
    # Sets up the OpenML instantiated in the child process to match that of the parent's
    # if configuration=None, loads the default
    config._setup(configuration)

    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=rep_no,
        fold=fold_no,
        sample=sample_no,
    )

    if isinstance(task, OpenMLSupervisedTask):
        x, y = task.get_X_and_y(dataset_format=dataset_format)
        if isinstance(x, pd.DataFrame):
            assert isinstance(y, (pd.Series, pd.DataFrame))
            train_x = x.iloc[train_indices]
            train_y = y.iloc[train_indices]
            test_x = x.iloc[test_indices]
            test_y = y.iloc[test_indices]
        else:
            # TODO(eddiebergman): Complains spmatrix doesn't support __getitem__ for typing
            assert y is not None
            train_x = x[train_indices]  # type: ignore
            train_y = y[train_indices]
            test_x = x[test_indices]  # type: ignore
            test_y = y[test_indices]
    elif isinstance(task, OpenMLClusteringTask):
        x = task.get_X(dataset_format=dataset_format)
        # TODO(eddiebergman): Complains spmatrix doesn't support __getitem__ for typing
        train_x = x.iloc[train_indices] if isinstance(x, pd.DataFrame) else x[train_indices]  # type: ignore
        train_y = None
        test_x = None
        test_y = None
    else:
        raise NotImplementedError(task.task_type)

    config.logger.info(
        f"Going to run model {model!s} on "
        f"dataset {openml.datasets.get_dataset(task.dataset_id).name} "
        f"for repeat {rep_no} fold {fold_no} sample {sample_no}"
    )
    (
        pred_y,
        proba_y,
        user_defined_measures_fold,
        trace,
    ) = extension._run_model_on_fold(
        model=model,
        task=task,
        X_train=train_x,
        # TODO(eddiebergman): Likely should not be ignored
        y_train=train_y,  # type: ignore
        rep_no=rep_no,
        fold_no=fold_no,
        X_test=test_x,
    )
    return pred_y, proba_y, test_indices, test_y, trace, user_defined_measures_fold  # type: ignore


def get_runs(run_ids: list[int]) -> list[OpenMLRun]:
    """Gets all runs in run_ids list.

    Parameters
    ----------
    run_ids : list of ints

    Returns
    -------
    runs : list of OpenMLRun
        List of runs corresponding to IDs, fetched from the server.
    """
    runs = []
    for run_id in run_ids:
        runs.append(get_run(run_id))
    return runs


@openml.utils.thread_safe_if_oslo_installed
def get_run(run_id: int, ignore_cache: bool = False) -> OpenMLRun:  # noqa: FBT002, FBT001
    """Gets run corresponding to run_id.

    Parameters
    ----------
    run_id : int

    ignore_cache : bool
        Whether to ignore the cache. If ``true`` this will download and overwrite the run xml
        even if the requested run is already cached.

    ignore_cache

    Returns
    -------
    run : OpenMLRun
        Run corresponding to ID, fetched from the server.
    """
    run_dir = Path(openml.utils._create_cache_directory_for_id(RUNS_CACHE_DIR_NAME, run_id))
    run_file = run_dir / "description.xml"

    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not ignore_cache:
            return _get_cached_run(run_id)

        raise OpenMLCacheException(message="dummy")

    except OpenMLCacheException:
        run_xml = openml._api_calls._perform_api_call("run/%d" % run_id, "get")
        with run_file.open("w", encoding="utf8") as fh:
            fh.write(run_xml)

    return _create_run_from_xml(run_xml)


def _create_run_from_xml(xml: str, from_server: bool = True) -> OpenMLRun:  # noqa: PLR0915, PLR0912, C901, FBT001, FBT002
    """Create a run object from xml returned from server.

    Parameters
    ----------
    xml : string
        XML describing a run.

    from_server : bool, optional (default=True)
        If True, an AttributeError is raised if any of the fields required by the server is not
        present in the xml. If False, those absent fields will be treated as None.

    Returns
    -------
    run : OpenMLRun
        New run object representing run_xml.
    """

    def obtain_field(xml_obj, fieldname, from_server, cast=None):  # type: ignore
        # this function can be used to check whether a field is present in an
        # object. if it is not present, either returns None or throws an error
        # (this is usually done if the xml comes from the server)
        if fieldname in xml_obj:
            if cast is not None:
                return cast(xml_obj[fieldname])
            return xml_obj[fieldname]

        if not from_server:
            return None

        raise AttributeError("Run XML does not contain required (server) " "field: ", fieldname)

    run = xmltodict.parse(xml, force_list=["oml:file", "oml:evaluation", "oml:parameter_setting"])[
        "oml:run"
    ]
    run_id = obtain_field(run, "oml:run_id", from_server, cast=int)
    uploader = obtain_field(run, "oml:uploader", from_server, cast=int)
    uploader_name = obtain_field(run, "oml:uploader_name", from_server)
    task_id = int(run["oml:task_id"])
    task_type = obtain_field(run, "oml:task_type", from_server)

    # even with the server requirement this field may be empty.
    task_evaluation_measure = run.get("oml:task_evaluation_measure", None)

    if not from_server and run["oml:flow_id"] is None:
        # This can happen for a locally stored run of which the flow is not yet published.
        flow_id = None
        parameters = None
    else:
        flow_id = obtain_field(run, "oml:flow_id", from_server, cast=int)
        # parameters are only properly formatted once the flow is established on the server.
        # thus they are also not stored for runs with local flows.
        parameters = []
        if "oml:parameter_setting" in run:
            obtained_parameter_settings = run["oml:parameter_setting"]
            for parameter_dict in obtained_parameter_settings:
                current_parameter = {
                    "oml:name": parameter_dict["oml:name"],
                    "oml:value": parameter_dict["oml:value"],
                }
                if "oml:component" in parameter_dict:
                    current_parameter["oml:component"] = parameter_dict["oml:component"]
                parameters.append(current_parameter)

    flow_name = obtain_field(run, "oml:flow_name", from_server)
    setup_id = obtain_field(run, "oml:setup_id", from_server, cast=int)
    setup_string = obtain_field(run, "oml:setup_string", from_server)
    # run_details is currently not sent by the server, so we need to retrieve it safely.
    # whenever that's resolved, we can enforce it being present (OpenML#1087)
    run_details = obtain_field(run, "oml:run_details", from_server=False)

    if "oml:input_data" in run:
        dataset_id = int(run["oml:input_data"]["oml:dataset"]["oml:did"])
    elif not from_server:
        dataset_id = None
    else:
        # fetching the task to obtain dataset_id
        t = openml.tasks.get_task(task_id, download_data=False)
        if not hasattr(t, "dataset_id"):
            raise ValueError(
                f"Unable to fetch dataset_id from the task({task_id}) linked to run({run_id})",
            )
        dataset_id = t.dataset_id

    files: dict[str, int] = {}
    evaluations: dict[str, float | Any] = {}
    fold_evaluations: dict[str, dict[int, dict[int, float | Any]]] = {}
    sample_evaluations: dict[str, dict[int, dict[int, dict[int, float | Any]]]] = {}
    if "oml:output_data" not in run:
        if from_server:
            raise ValueError("Run does not contain output_data " "(OpenML server error?)")
        predictions_url = None
    else:
        output_data = run["oml:output_data"]
        predictions_url = None
        if "oml:file" in output_data:
            # multiple files, the normal case
            for file_dict in output_data["oml:file"]:
                files[file_dict["oml:name"]] = int(file_dict["oml:file_id"])
                if file_dict["oml:name"] == "predictions":
                    predictions_url = file_dict["oml:url"]
        if "oml:evaluation" in output_data:
            # in normal cases there should be evaluations, but in case there
            # was an error these could be absent
            for evaluation_dict in output_data["oml:evaluation"]:
                key = evaluation_dict["oml:name"]
                if "oml:value" in evaluation_dict:
                    value = float(evaluation_dict["oml:value"])
                elif "oml:array_data" in evaluation_dict:
                    value = evaluation_dict["oml:array_data"]
                else:
                    raise ValueError(
                        'Could not find keys "value" or '
                        f'"array_data" in {evaluation_dict.keys()!s}',
                    )
                if (
                    "@repeat" in evaluation_dict
                    and "@fold" in evaluation_dict
                    and "@sample" in evaluation_dict
                ):
                    repeat = int(evaluation_dict["@repeat"])
                    fold = int(evaluation_dict["@fold"])
                    sample = int(evaluation_dict["@sample"])
                    if key not in sample_evaluations:
                        sample_evaluations[key] = {}
                    if repeat not in sample_evaluations[key]:
                        sample_evaluations[key][repeat] = {}
                    if fold not in sample_evaluations[key][repeat]:
                        sample_evaluations[key][repeat][fold] = {}
                    sample_evaluations[key][repeat][fold][sample] = value
                elif "@repeat" in evaluation_dict and "@fold" in evaluation_dict:
                    repeat = int(evaluation_dict["@repeat"])
                    fold = int(evaluation_dict["@fold"])
                    if key not in fold_evaluations:
                        fold_evaluations[key] = {}
                    if repeat not in fold_evaluations[key]:
                        fold_evaluations[key][repeat] = {}
                    fold_evaluations[key][repeat][fold] = value
                else:
                    evaluations[key] = value

    if "description" not in files and from_server is True:
        raise ValueError("No description file for run %d in run " "description XML" % run_id)

    if "predictions" not in files and from_server is True:
        task = openml.tasks.get_task(task_id)
        if task.task_type_id == TaskType.SUBGROUP_DISCOVERY:
            raise NotImplementedError("Subgroup discovery tasks are not yet supported.")

        # JvR: actually, I am not sure whether this error should be raised.
        # a run can consist without predictions. But for now let's keep it
        # Matthias: yes, it should stay as long as we do not really handle
        # this stuff
        raise ValueError("No prediction files for run %d in run description XML" % run_id)

    tags = openml.utils.extract_xml_tags("oml:tag", run)

    return OpenMLRun(
        run_id=run_id,
        uploader=uploader,
        uploader_name=uploader_name,
        task_id=task_id,
        task_type=task_type,
        task_evaluation_measure=task_evaluation_measure,
        flow_id=flow_id,
        flow_name=flow_name,
        setup_id=setup_id,
        setup_string=setup_string,
        parameter_settings=parameters,
        dataset_id=dataset_id,
        output_files=files,
        # Make sure default values are used where needed to keep run objects identical
        evaluations=evaluations or None,
        fold_evaluations=fold_evaluations or None,
        sample_evaluations=sample_evaluations or None,
        tags=tags,
        predictions_url=predictions_url,
        run_details=run_details,
    )


def _get_cached_run(run_id: int) -> OpenMLRun:
    """Load a run from the cache."""
    run_cache_dir = openml.utils._create_cache_directory_for_id(RUNS_CACHE_DIR_NAME, run_id)
    run_file = run_cache_dir / "description.xml"
    try:
        with run_file.open(encoding="utf8") as fh:
            return _create_run_from_xml(xml=fh.read())
    except OSError as e:
        raise OpenMLCacheException(f"Run file for run id {run_id} not cached") from e


# TODO(eddiebergman): Could overload, likely too large an annoying to do
# nvm, will be deprecated in 0.15
def list_runs(  # noqa: PLR0913
    offset: int | None = None,
    size: int | None = None,
    id: list | None = None,  # noqa: A002
    task: list[int] | None = None,
    setup: list | None = None,
    flow: list | None = None,
    uploader: list | None = None,
    tag: str | None = None,
    study: int | None = None,
    display_errors: bool = False,  # noqa: FBT001, FBT002
    output_format: Literal["dict", "dataframe"] = "dict",
    **kwargs: Any,
) -> dict | pd.DataFrame:
    """
    List all runs matching all of the given filters.
    (Supports large amount of results)

    Parameters
    ----------
    offset : int, optional
        the number of runs to skip, starting from the first
    size : int, optional
        the maximum number of runs to show

    id : list, optional

    task : list, optional

    setup: list, optional

    flow : list, optional

    uploader : list, optional

    tag : str, optional

    study : int, optional

    display_errors : bool, optional (default=None)
        Whether to list runs which have an error (for example a missing
        prediction file).

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    kwargs : dict, optional
        Legal filter operators: task_type.

    Returns
    -------
    dict of dicts, or dataframe
    """
    if output_format not in ["dataframe", "dict"]:
        raise ValueError("Invalid output format selected. Only 'dict' or 'dataframe' applicable.")

    # TODO: [0.15]
    if output_format == "dict":
        msg = (
            "Support for `output_format` of 'dict' will be removed in 0.15 "
            "and pandas dataframes will be returned instead. To ensure your code "
            "will continue to work, use `output_format`='dataframe'."
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)

    # TODO(eddiebergman): Do we really need this runtime type validation?
    if id is not None and (not isinstance(id, list)):
        raise TypeError("id must be of type list.")
    if task is not None and (not isinstance(task, list)):
        raise TypeError("task must be of type list.")
    if setup is not None and (not isinstance(setup, list)):
        raise TypeError("setup must be of type list.")
    if flow is not None and (not isinstance(flow, list)):
        raise TypeError("flow must be of type list.")
    if uploader is not None and (not isinstance(uploader, list)):
        raise TypeError("uploader must be of type list.")

    return openml.utils._list_all(  # type: ignore
        list_output_format=output_format,  # type: ignore
        listing_call=_list_runs,
        offset=offset,
        size=size,
        id=id,
        task=task,
        setup=setup,
        flow=flow,
        uploader=uploader,
        tag=tag,
        study=study,
        display_errors=display_errors,
        **kwargs,
    )


def _list_runs(  # noqa: PLR0913
    id: list | None = None,  # noqa: A002
    task: list | None = None,
    setup: list | None = None,
    flow: list | None = None,
    uploader: list | None = None,
    study: int | None = None,
    display_errors: bool = False,  # noqa: FBT002, FBT001
    output_format: Literal["dict", "dataframe"] = "dict",
    **kwargs: Any,
) -> dict | pd.DataFrame:
    """
    Perform API call `/run/list/{filters}'
    <https://www.openml.org/api_docs/#!/run/get_run_list_filters>`

    Parameters
    ----------
    The arguments that are lists are separated from the single value
    ones which are put into the kwargs.
    display_errors is also separated from the kwargs since it has a
    default value.

    id : list, optional

    task : list, optional

    setup: list, optional

    flow : list, optional

    uploader : list, optional

    study : int, optional

    display_errors : bool, optional (default=None)
        Whether to list runs which have an error (for example a missing
        prediction file).

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    kwargs : dict, optional
        Legal filter operators: task_type.

    Returns
    -------
    dict, or dataframe
        List of found runs.
    """
    api_call = "run/list"
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += f"/{operator}/{value}"
    if id is not None:
        api_call += "/run/{}".format(",".join([str(int(i)) for i in id]))
    if task is not None:
        api_call += "/task/{}".format(",".join([str(int(i)) for i in task]))
    if setup is not None:
        api_call += "/setup/{}".format(",".join([str(int(i)) for i in setup]))
    if flow is not None:
        api_call += "/flow/{}".format(",".join([str(int(i)) for i in flow]))
    if uploader is not None:
        api_call += "/uploader/{}".format(",".join([str(int(i)) for i in uploader]))
    if study is not None:
        api_call += "/study/%d" % study
    if display_errors:
        api_call += "/show_errors/true"
    return __list_runs(api_call=api_call, output_format=output_format)


def __list_runs(
    api_call: str, output_format: Literal["dict", "dataframe"] = "dict"
) -> dict | pd.DataFrame:
    """Helper function to parse API calls which are lists of runs"""
    xml_string = openml._api_calls._perform_api_call(api_call, "get")
    runs_dict = xmltodict.parse(xml_string, force_list=("oml:run",))
    # Minimalistic check if the XML is useful
    if "oml:runs" not in runs_dict:
        raise ValueError(f'Error in return XML, does not contain "oml:runs": {runs_dict}')

    if "@xmlns:oml" not in runs_dict["oml:runs"]:
        raise ValueError(
            f'Error in return XML, does not contain "oml:runs"/@xmlns:oml: {runs_dict}'
        )

    if runs_dict["oml:runs"]["@xmlns:oml"] != "http://openml.org/openml":
        raise ValueError(
            "Error in return XML, value of  "
            '"oml:runs"/@xmlns:oml is not '
            f'"http://openml.org/openml": {runs_dict}',
        )

    assert isinstance(runs_dict["oml:runs"]["oml:run"], list), type(runs_dict["oml:runs"])

    runs = {
        int(r["oml:run_id"]): {
            "run_id": int(r["oml:run_id"]),
            "task_id": int(r["oml:task_id"]),
            "setup_id": int(r["oml:setup_id"]),
            "flow_id": int(r["oml:flow_id"]),
            "uploader": int(r["oml:uploader"]),
            "task_type": TaskType(int(r["oml:task_type_id"])),
            "upload_time": str(r["oml:upload_time"]),
            "error_message": str((r["oml:error_message"]) or ""),
        }
        for r in runs_dict["oml:runs"]["oml:run"]
    }

    if output_format == "dataframe":
        runs = pd.DataFrame.from_dict(runs, orient="index")

    return runs


def format_prediction(  # noqa: PLR0913
    task: OpenMLSupervisedTask,
    repeat: int,
    fold: int,
    index: int,
    prediction: str | int | float,
    truth: str | int | float,
    sample: int | None = None,
    proba: dict[str, float] | None = None,
) -> list[str | int | float]:
    """Format the predictions in the specific order as required for the run results.

    Parameters
    ----------
    task: OpenMLSupervisedTask
        Task for which to format the predictions.
    repeat: int
        From which repeat this predictions is made.
    fold: int
        From which fold this prediction is made.
    index: int
        For which index this prediction is made.
    prediction: str, int or float
        The predicted class label or value.
    truth: str, int or float
        The true class label or value.
    sample: int, optional (default=None)
        From which sample set this prediction is made.
        Required only for LearningCurve tasks.
    proba: Dict[str, float], optional (default=None)
        For classification tasks only.
        A mapping from each class label to their predicted probability.
        The dictionary should contain an entry for each of the `task.class_labels`.
        E.g.: {"Iris-Setosa": 0.2, "Iris-Versicolor": 0.7, "Iris-Virginica": 0.1}

    Returns
    -------
    A list with elements for the prediction results of a run.

    The returned order of the elements is (if available):
        [repeat, fold, sample, index, prediction, truth, *probabilities]

    This order follows the R Client API.
    """
    if isinstance(task, OpenMLClassificationTask):
        if proba is None:
            raise ValueError("`proba` is required for classification task")
        if task.class_labels is None:
            raise ValueError("The classification task must have class labels set")
        if not set(task.class_labels) == set(proba):
            raise ValueError("Each class should have a predicted probability")
        if sample is None:
            if isinstance(task, OpenMLLearningCurveTask):
                raise ValueError("`sample` can not be none for LearningCurveTask")

            sample = 0
        probabilities = [proba[c] for c in task.class_labels]
        return [repeat, fold, sample, index, prediction, truth, *probabilities]

    if isinstance(task, OpenMLRegressionTask):
        return [repeat, fold, index, prediction, truth]

    raise NotImplementedError(f"Formatting for {type(task)} is not supported.")


def delete_run(run_id: int) -> bool:
    """Delete run with id `run_id` from the OpenML server.

    You can only delete runs which you uploaded.

    Parameters
    ----------
    run_id : int
        OpenML id of the run

    Returns
    -------
    bool
        True if the deletion was successful. False otherwise.
    """
    return openml.utils._delete_entity("run", run_id)
