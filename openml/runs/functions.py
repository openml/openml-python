from collections import defaultdict
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import sklearn.pipeline
import six
import xmltodict
import sklearn.metrics

import openml
import openml.utils
from ..exceptions import PyOpenMLError
from .. import config
from ..flows import sklearn_to_flow, get_flow, flow_exists, _check_n_jobs, \
    _copy_server_fields
from ..setups import setup_exists, initialize_model
from ..exceptions import OpenMLCacheException, OpenMLServerException
from .._api_calls import _perform_api_call
from .run import OpenMLRun, _get_version_information
from .trace import OpenMLRunTrace, OpenMLTraceIteration


# _get_version_info, _get_dict and _create_setup_string are in run.py to avoid
# circular imports


def run_model_on_task(task, model, avoid_duplicate_runs=True, flow_tags=None,
                      seed=None):
    """See ``run_flow_on_task for a documentation."""

    flow = sklearn_to_flow(model)

    return run_flow_on_task(task=task, flow=flow,
                            avoid_duplicate_runs=avoid_duplicate_runs,
                            flow_tags=flow_tags, seed=seed)


def run_flow_on_task(task, flow, avoid_duplicate_runs=True, flow_tags=None,
                     seed=None):
    """Run the model provided by the flow on the dataset defined by task.

    Takes the flow and repeat information into account. In case a flow is not
    yet published, it is published after executing the run (requires
    internet connection).

    Parameters
    ----------
    task : OpenMLTask
        Task to perform.
    model : sklearn model
        A model which has a function fit(X,Y) and predict(X),
        all supervised estimators of scikit learn follow this definition of a model [1]
        [1](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)
    avoid_duplicate_runs : bool
        If this flag is set to True, the run will throw an error if the
        setup/task combination is already present on the server. Works only
        if the flow is already published on the server. This feature requires an
        internet connection.
    flow_tags : list(str)
        A list of tags that the flow should have at creation.
    seed: int
        Models that are not seeded will get this seed.

    Returns
    -------
    run : OpenMLRun
        Result of the run.
    """
    if flow_tags is not None and not isinstance(flow_tags, list):
        raise ValueError("flow_tags should be list")

    flow.model = _get_seeded_model(flow.model, seed=seed)

    # skips the run if it already exists and the user opts for this in the config file.
    # also, if the flow is not present on the server, the check is not needed.
    flow_id = flow_exists(flow.name, flow.external_version)
    if avoid_duplicate_runs and flow_id:
        flow_from_server = get_flow(flow_id)
        setup_id = setup_exists(flow_from_server, flow.model)
        ids = _run_exists(task.task_id, setup_id)
        if ids:
            raise PyOpenMLError("Run already exists in server. Run id(s): %s" %str(ids))
        _copy_server_fields(flow_from_server, flow)

    dataset = task.get_dataset()

    class_labels = task.class_labels
    if class_labels is None:
        raise ValueError('The task has no class labels. This method currently '
                         'only works for tasks with class labels.')

    run_environment = _get_version_information()
    tags = ['openml-python', run_environment[1]]

    # execute the run
    res = _run_task_get_arffcontent(flow.model, task, class_labels)

    if flow.flow_id is None:
        _publish_flow_if_necessary(flow)

    run = OpenMLRun(task_id=task.task_id, flow_id=flow.flow_id,
                    dataset_id=dataset.dataset_id, model=flow.model, tags=tags)
    run.parameter_settings = OpenMLRun._parse_parameters(flow)

    run.data_content, run.trace_content, run.trace_attributes, fold_evaluations, sample_evaluations = res
    # now we need to attach the detailed evaluations
    if task.task_type_id == 3:
        run.sample_evaluations = sample_evaluations
    else:
        run.fold_evaluations = fold_evaluations

    config.logger.info('Executed Task %d with Flow id: %d' % (task.task_id, run.flow_id))

    return run


def _publish_flow_if_necessary(flow):
    # try publishing the flow if one has to assume it doesn't exist yet. It
    # might fail because it already exists, then the flow is currently not
    # reused

        try:
            flow.publish()
        except OpenMLServerException as e:
            if e.message == "flow already exists":
                flow_id = openml.flows.flow_exists(flow.name,
                                                   flow.external_version)
                server_flow = get_flow(flow_id)
                openml.flows.flow._copy_server_fields(server_flow, flow)
                openml.flows.assert_flows_equal(flow, server_flow,
                                                ignore_parameter_values=True)
            else:
                raise e


def get_run_trace(run_id):
    """Get the optimization trace object for a given run id.

     Parameters
     ----------
     run_id : int

     Returns
     -------
     openml.runs.OpenMLTrace
    """

    trace_xml = _perform_api_call('run/trace/%d' % run_id)
    run_trace = _create_trace_from_description(trace_xml)
    return run_trace


def initialize_model_from_run(run_id):
    '''
    Initialized a model based on a run_id (i.e., using the exact
    same parameter settings)

    Parameters
        ----------
        run_id : int
            The Openml run_id

        Returns
        -------
        model : sklearn model
            the scikitlearn model with all parameters initailized
    '''
    run = get_run(run_id)
    return initialize_model(run.setup_id)


def initialize_model_from_trace(run_id, repeat, fold, iteration=None):
    '''
    Initialize a model based on the parameters that were set
    by an optimization procedure (i.e., using the exact same
    parameter settings)

    Parameters
    ----------
    run_id : int
        The Openml run_id. Should contain a trace file, 
        otherwise a OpenMLServerException is raised

    repeat: int
        The repeat nr (column in trace file)

    fold: int
        The fold nr (column in trace file)

    iteration: int
        The iteration nr (column in trace file). If None, the
        best (selected) iteration will be searched (slow), 
        according to the selection criteria implemented in
        OpenMLRunTrace.get_selected_iteration

    Returns
    -------
    model : sklearn model
        the scikit-learn model with all parameters initailized
    '''
    run_trace = get_run_trace(run_id)

    if iteration is None:
        iteration = run_trace.get_selected_iteration(repeat, fold)

    request = (repeat, fold, iteration)
    if request not in run_trace.trace_iterations:
        raise ValueError('Combination repeat, fold, iteration not availavle')
    current = run_trace.trace_iterations[(repeat, fold, iteration)]

    search_model = initialize_model_from_run(run_id)
    if not isinstance(search_model, sklearn.model_selection._search.BaseSearchCV):
        raise ValueError('Deserialized flow not instance of ' \
                         'sklearn.model_selection._search.BaseSearchCV')
    base_estimator = search_model.estimator
    base_estimator.set_params(**current.get_parameters())
    return base_estimator


def _run_exists(task_id, setup_id):
    '''
    Checks whether a task/setup combination is already present on the server.

    :param task_id: int
    :param setup_id: int
    :return: List of run ids iff these already exists on the server, False otherwise
    '''
    if setup_id <= 0:
        # openml setups are in range 1-inf
        return False

    try:
        result = list_runs(task=[task_id], setup=[setup_id])
        if len(result) > 0:
            return set(result.keys())
        else:
            return False
    except OpenMLServerException as exception:
        # error code 512 implies no results. This means the run does not exist yet
        assert(exception.code == 512)
        return False


def _get_seeded_model(model, seed=None):
    '''Sets all the non-seeded components of a model with a seed.
       Models that are already seeded will maintain the seed. In
       this case, only integer seeds are allowed (An exception
       is thrown when a RandomState was used as seed)

        Parameters
        ----------
        model : sklearn model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        model : sklearn model
            a version of the model where all (sub)components have
            a seed
    '''

    def _seed_current_object(current_value):
        if isinstance(current_value, int):  # acceptable behaviour
            return False
        elif isinstance(current_value, np.random.RandomState):
            raise ValueError(
                'Models initialized with a RandomState object are not supported. Please seed with an integer. ')
        elif current_value is not None:
            raise ValueError(
                'Models should be seeded with int or None (this should never happen). ')
        else:
            return True

    rs = np.random.RandomState(seed)
    model_params = model.get_params()
    random_states = {}
    for param_name in sorted(model_params):
        if 'random_state' in param_name:
            currentValue = model_params[param_name]
            # important to draw the value at this point (and not in the if statement)
            # this way we guarantee that if a different set of subflows is seeded,
            # the same number of the random generator is used
            newValue = rs.randint(0, 2**16)
            if _seed_current_object(currentValue):
                random_states[param_name] = newValue

        # Also seed CV objects!
        elif isinstance(model_params[param_name],
                        sklearn.model_selection.BaseCrossValidator):
            if not hasattr(model_params[param_name], 'random_state'):
                continue

            currentValue = model_params[param_name].random_state
            newValue = rs.randint(0, 2 ** 16)
            if _seed_current_object(currentValue):
                model_params[param_name].random_state = newValue

    model.set_params(**random_states)
    return model


def _prediction_to_row(rep_no, fold_no, sample_no, row_id, correct_label,
                       predicted_label, predicted_probabilities, class_labels,
                       model_classes_mapping):
    """Util function that turns probability estimates of a classifier for a given
        instance into the right arff format to upload to openml.

        Parameters
        ----------
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        sample_no : int
            In case of learning curves, the index of the subsample (0-based; in case of no learning curve, always 0)
        row_id : int
            row id in the initial dataset
        correct_label : str
            original label of the instance
        predicted_label : str
            the label that was predicted
        predicted_probabilities : array (size=num_classes)
            probabilities per class
        class_labels : array (size=num_classes)
        model_classes_mapping : list
            A list of classes the model produced.
            Obtained by BaseEstimator.classes_

        Returns
        -------
        arff_line : list
            representation of the current prediction in OpenML format
        """
    if not isinstance(rep_no, (int, np.integer)): raise ValueError('rep_no should be int')
    if not isinstance(fold_no, (int, np.integer)): raise ValueError('fold_no should be int')
    if not isinstance(sample_no, (int, np.integer)): raise ValueError('sample_no should be int')
    if not isinstance(row_id, (int, np.integer)): raise ValueError('row_id should be int')
    if not len(predicted_probabilities) == len(model_classes_mapping):
        raise ValueError('len(predicted_probabilities) != len(class_labels)')

    arff_line = [rep_no, fold_no, sample_no, row_id]
    for class_label_idx in range(len(class_labels)):
        if class_label_idx in model_classes_mapping:
            index = np.where(model_classes_mapping == class_label_idx)[0][0]  # TODO: WHY IS THIS 2D???
            arff_line.append(predicted_probabilities[index])
        else:
            arff_line.append(0.0)

    arff_line.append(class_labels[predicted_label])
    arff_line.append(correct_label)
    return arff_line


# JvR: why is class labels a parameter? could be removed and taken from task object, right?
def _run_task_get_arffcontent(model, task, class_labels):

    def _prediction_to_probabilities(y, model_classes):
        # y: list or numpy array of predictions
        # model_classes: sklearn classifier mapping from original array id to prediction index id
        if not isinstance(model_classes, list):
            raise ValueError('please convert model classes to list prior to calling this fn')
        result = np.zeros((len(y), len(model_classes)), dtype=np.float32)
        for obs, prediction_idx in enumerate(y):
            array_idx = model_classes.index(prediction_idx)
            result[obs][array_idx] = 1.0
        return result

    X, Y = task.get_X_and_y()
    arff_datacontent = []
    arff_tracecontent = []
    # stores fold-based evaluation measures. In case of a sample based task,
    # this information is multiple times overwritten, but due to the ordering
    # of tne loops, eventually it contains the information based on the full
    # dataset size
    user_defined_measures_fold = defaultdict(lambda: defaultdict(dict))
    # stores sample-based evaluation measures (sublevel of fold-based)
    # will also be filled on a non sample-based task, but the information
    # is the same as the fold-based measures, and disregarded in that case
    user_defined_measures_sample = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # sys.version_info returns a tuple, the following line compares the entry of tuples
    # https://docs.python.org/3.6/reference/expressions.html#value-comparisons
    can_measure_runtime = sys.version_info[:2] >= (3, 3) and _check_n_jobs(model)
    # TODO use different iterator to only provide a single iterator (less
    # methods, less maintenance, less confusion)
    num_reps, num_folds, num_samples = task.get_split_dimensions()

    for rep_no in range(num_reps):
        for fold_no in range(num_folds):
            for sample_no in range(num_samples):
                model_fold = sklearn.base.clone(model, safe=True)
                train_indices, test_indices = task.get_train_test_split_indices(repeat=rep_no,
                                                                                fold=fold_no,
                                                                                sample=sample_no)
                trainX = X[train_indices]
                trainY = Y[train_indices]
                testX = X[test_indices]
                testY = Y[test_indices]

                try:
                    # for measuring runtime. Only available since Python 3.3
                    if can_measure_runtime:
                        modelfit_starttime = time.process_time()
                    model_fold.fit(trainX, trainY)

                    if can_measure_runtime:
                        modelfit_duration = (time.process_time() - modelfit_starttime) * 1000
                        user_defined_measures_sample['usercpu_time_millis_training'][rep_no][fold_no][sample_no] = modelfit_duration
                        user_defined_measures_fold['usercpu_time_millis_training'][rep_no][fold_no] = modelfit_duration
                except AttributeError as e:
                    # typically happens when training a regressor on classification task
                    raise PyOpenMLError(str(e))

                # extract trace, if applicable
                if isinstance(model_fold, sklearn.model_selection._search.BaseSearchCV):
                    arff_tracecontent.extend(_extract_arfftrace(model_fold, rep_no, fold_no))

                # search for model classes_ (might differ depending on modeltype)
                # first, pipelines are a special case (these don't have a classes_
                # object, but rather borrows it from the last step. We do this manually,
                # because of the BaseSearch check)
                if isinstance(model_fold, sklearn.pipeline.Pipeline):
                    used_estimator = model_fold.steps[-1][-1]
                else:
                    used_estimator = model_fold

                if isinstance(used_estimator, sklearn.model_selection._search.BaseSearchCV):
                    model_classes = used_estimator.best_estimator_.classes_
                else:
                    model_classes = used_estimator.classes_

                if can_measure_runtime:
                    modelpredict_starttime = time.process_time()

                PredY = model_fold.predict(testX)
                try:
                    ProbaY = model_fold.predict_proba(testX)
                except AttributeError:
                    ProbaY = _prediction_to_probabilities(PredY, list(model_classes))

                # add client-side calculated metrics. These might be used on the server as consistency check
                def _calculate_local_measure(sklearn_fn, openml_name):
                    user_defined_measures_fold[openml_name][rep_no][fold_no] = \
                        sklearn_fn(testY, PredY)
                    user_defined_measures_sample[openml_name][rep_no][fold_no][sample_no] = \
                        sklearn_fn(testY, PredY)

                _calculate_local_measure(sklearn.metrics.accuracy_score, 'predictive_accuracy')

                if can_measure_runtime:
                    modelpredict_duration = (time.process_time() - modelpredict_starttime) * 1000
                    user_defined_measures_fold['usercpu_time_millis_testing'][rep_no][fold_no] = modelpredict_duration
                    user_defined_measures_fold['usercpu_time_millis'][rep_no][fold_no] = modelfit_duration + modelpredict_duration
                    user_defined_measures_sample['usercpu_time_millis_testing'][rep_no][fold_no][sample_no] = modelpredict_duration
                    user_defined_measures_sample['usercpu_time_millis'][rep_no][fold_no][sample_no] = modelfit_duration + modelpredict_duration

                if ProbaY.shape[1] != len(class_labels):
                    warnings.warn("Repeat %d Fold %d: estimator only predicted for %d/%d classes!" %(rep_no, fold_no, ProbaY.shape[1], len(class_labels)))

                for i in range(0, len(test_indices)):
                    arff_line = _prediction_to_row(rep_no, fold_no, sample_no,
                                                   test_indices[i], class_labels[testY[i]],
                                                   PredY[i], ProbaY[i], class_labels, model_classes)
                    arff_datacontent.append(arff_line)

    if isinstance(model_fold, sklearn.model_selection._search.BaseSearchCV):
        # arff_tracecontent is already set
        arff_trace_attributes = _extract_arfftrace_attributes(model_fold)
    else:
        arff_tracecontent = None
        arff_trace_attributes = None

    return arff_datacontent, \
           arff_tracecontent, \
           arff_trace_attributes, \
           user_defined_measures_fold, \
           user_defined_measures_sample


def _extract_arfftrace(model, rep_no, fold_no):
    if not isinstance(model, sklearn.model_selection._search.BaseSearchCV):
        raise ValueError('model should be instance of'\
                         ' sklearn.model_selection._search.BaseSearchCV')
    if not hasattr(model, 'cv_results_'):
        raise ValueError('model should contain `cv_results_`')

    arff_tracecontent = []
    for itt_no in range(0, len(model.cv_results_['mean_test_score'])):
        # we use the string values for True and False, as it is defined in this way by the OpenML server
        selected = 'false'
        if itt_no == model.best_index_:
            selected = 'true'
        test_score = model.cv_results_['mean_test_score'][itt_no]
        arff_line = [rep_no, fold_no, itt_no, test_score, selected]
        for key in model.cv_results_:
            if key.startswith('param_'):
                serialized_value = json.dumps(model.cv_results_[key][itt_no])
                arff_line.append(serialized_value)
        arff_tracecontent.append(arff_line)
    return arff_tracecontent


def _extract_arfftrace_attributes(model):
    if not isinstance(model, sklearn.model_selection._search.BaseSearchCV):
        raise ValueError('model should be instance of'\
                         ' sklearn.model_selection._search.BaseSearchCV')
    if not hasattr(model, 'cv_results_'):
        raise ValueError('model should contain `cv_results_`')

    # attributes that will be in trace arff, regardless of the model
    trace_attributes = [('repeat', 'NUMERIC'),
                        ('fold', 'NUMERIC'),
                        ('iteration', 'NUMERIC'),
                        ('evaluation', 'NUMERIC'),
                        ('selected', ['true', 'false'])]

    # model dependent attributes for trace arff
    for key in model.cv_results_:
        if key.startswith('param_'):
            # supported types should include all types, including bool, int float
            supported_types = (bool, int, float, six.string_types)
            if all(isinstance(i, supported_types) or i is None for i in model.cv_results_[key]):
                type = 'STRING'
            else:
                raise TypeError('Unsupported param type in param grid')

            # we renamed the attribute param to parameter, as this is a required
            # OpenML convention
            attribute = ("parameter_" + key[6:], type)
            trace_attributes.append(attribute)
    return trace_attributes


def get_runs(run_ids):
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


def get_run(run_id):
    """Gets run corresponding to run_id.

    Parameters
    ----------
    run_id : int

    Returns
    -------
    run : OpenMLRun
        Run corresponding to ID, fetched from the server.
    """
    run_file = os.path.join(config.get_cache_directory(), "runs",
                            "run_%d.xml" % run_id)

    try:
        return _get_cached_run(run_id)

    except (OpenMLCacheException):
        run_xml = _perform_api_call("run/%d" % run_id)
        with io.open(run_file, "w", encoding='utf8') as fh:
            fh.write(run_xml)

    run = _create_run_from_xml(run_xml)

    return run


def _create_run_from_xml(xml):
    """Create a run object from xml returned from server.

    Parameters
    ----------
    run_xml : string
        XML describing a run.

    Returns
    -------
    run : OpenMLRun
        New run object representing run_xml.
    """
    run = xmltodict.parse(xml)["oml:run"]
    run_id = int(run['oml:run_id'])
    uploader = int(run['oml:uploader'])
    uploader_name = run['oml:uploader_name']
    task_id = int(run['oml:task_id'])
    task_type = run['oml:task_type']
    if 'oml:task_evaluation_measure' in run:
        task_evaluation_measure = run['oml:task_evaluation_measure']
    else:
        task_evaluation_measure = None

    flow_id = int(run['oml:flow_id'])
    flow_name = run['oml:flow_name']
    setup_id = int(run['oml:setup_id'])
    setup_string = run['oml:setup_string']

    parameters = dict()
    if 'oml:parameter_settings' in run:
        parameter_settings = run['oml:parameter_settings']
        for parameter_dict in parameter_settings:
            key = parameter_dict['oml:name']
            value = parameter_dict['oml:value']
            parameters[key] = value

    dataset_id = int(run['oml:input_data']['oml:dataset']['oml:did'])

    files = dict()
    evaluations = dict()
    fold_evaluations = defaultdict(lambda: defaultdict(dict))
    sample_evaluations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    if 'oml:output_data' not in run:
        raise ValueError('Run does not contain output_data (OpenML server error?)')
    else:
        if isinstance(run['oml:output_data']['oml:file'], dict):
            # only one result.. probably due to an upload error
            file_dict = run['oml:output_data']['oml:file']
            files[file_dict['oml:name']] = int(file_dict['oml:file_id'])
        elif isinstance(run['oml:output_data']['oml:file'], list):
            # multiple files, the normal case
            for file_dict in run['oml:output_data']['oml:file']:
                files[file_dict['oml:name']] = int(file_dict['oml:file_id'])
        else:
            raise TypeError(type(run['oml:output_data']['oml:file']))

        if 'oml:evaluation' in run['oml:output_data']:
            # in normal cases there should be evaluations, but in case there
            # was an error these could be absent
            for evaluation_dict in run['oml:output_data']['oml:evaluation']:
                key = evaluation_dict['oml:name']
                if 'oml:value' in evaluation_dict:
                    value = float(evaluation_dict['oml:value'])
                elif 'oml:array_data' in evaluation_dict:
                    value = evaluation_dict['oml:array_data']
                else:
                    raise ValueError('Could not find keys "value" or "array_data" '
                                     'in %s' % str(evaluation_dict.keys()))
                if '@repeat' in evaluation_dict and '@fold' in evaluation_dict and '@sample' in evaluation_dict:
                    repeat = int(evaluation_dict['@repeat'])
                    fold = int(evaluation_dict['@fold'])
                    sample = int(evaluation_dict['@sample'])
                    repeat_dict = sample_evaluations[key]
                    fold_dict = repeat_dict[repeat]
                    sample_dict = fold_dict[fold]
                    sample_dict[sample] = value
                elif '@repeat' in evaluation_dict and '@fold' in evaluation_dict:
                    repeat = int(evaluation_dict['@repeat'])
                    fold = int(evaluation_dict['@fold'])
                    repeat_dict = fold_evaluations[key]
                    fold_dict = repeat_dict[repeat]
                    fold_dict[fold] = value
                else:
                    evaluations[key] = value

    if 'description' not in files:
        raise ValueError('No description file for run %d in run '
                         'description XML' % run_id)

    if 'predictions' not in files:
        task = openml.tasks.get_task(task_id)
        if task.task_type_id == 8:
            raise NotImplementedError(
                'Subgroup discovery tasks are not yet supported.'
            )
        else:
            # JvR: actually, I am not sure whether this error should be raised.
            # a run can consist without predictions. But for now let's keep it
            # Matthias: yes, it should stay as long as we do not really handle
            # this stuff
            raise ValueError('No prediction files for run %d in run '
                             'description XML' % run_id)

    tags = openml.utils.extract_xml_tags('oml:tag', run)

    return OpenMLRun(run_id=run_id, uploader=uploader,
                     uploader_name=uploader_name, task_id=task_id,
                     task_type=task_type,
                     task_evaluation_measure=task_evaluation_measure,
                     flow_id=flow_id, flow_name=flow_name,
                     setup_id=setup_id, setup_string=setup_string,
                     parameter_settings=parameters,
                     dataset_id=dataset_id, output_files=files,
                     evaluations=evaluations,
                     fold_evaluations=fold_evaluations,
                     sample_evaluations=sample_evaluations,
                     tags=tags)


def _create_trace_from_description(xml):
    result_dict = xmltodict.parse(xml, force_list=('oml:trace_iteration',))['oml:trace']

    run_id = result_dict['oml:run_id']
    trace = dict()

    if 'oml:trace_iteration' not in result_dict:
        raise ValueError('Run does not contain valid trace. ')

    assert type(result_dict['oml:trace_iteration']) == list, \
        type(result_dict['oml:trace_iteration'])

    for itt in result_dict['oml:trace_iteration']:
        repeat = int(itt['oml:repeat'])
        fold = int(itt['oml:fold'])
        iteration = int(itt['oml:iteration'])
        setup_string = json.loads(itt['oml:setup_string'])
        evaluation = float(itt['oml:evaluation'])

        selectedValue = itt['oml:selected']
        if selectedValue == 'true':
            selected = True
        elif selectedValue == 'false':
            selected = False
        else:
            raise ValueError('expected {"true", "false"} value for '\
                             'selected field, received: %s' %selectedValue)

        current = OpenMLTraceIteration(repeat, fold, iteration,
                                        setup_string, evaluation,
                                        selected)
        trace[(repeat, fold, iteration)] = current

    return OpenMLRunTrace(run_id, trace)


def _create_trace_from_arff(arff_obj):
    """
    Creates a trace file from arff obj (for example, generated by a local run)

    Parameters
    ----------
    arff_obj : dict
        LIAC arff obj, dict containing attributes, relation, data and description

    Returns
    -------
    run : OpenMLRunTrace
        Object containing None for run id and a dict containing the trace iterations
    """
    trace = dict()
    attribute_idx = {att[0]: idx for idx, att in enumerate(arff_obj['attributes'])}
    for required_attribute in ['repeat', 'fold', 'iteration', 'evaluation', 'selected']:
        if required_attribute not in attribute_idx:
            raise ValueError('arff misses required attribute: %s' %required_attribute)

    for itt in arff_obj['data']:
        repeat = int(itt[attribute_idx['repeat']])
        fold = int(itt[attribute_idx['fold']])
        iteration = int(itt[attribute_idx['iteration']])
        evaluation = float(itt[attribute_idx['evaluation']])
        selectedValue = itt[attribute_idx['selected']]
        if selectedValue == 'true':
            selected = True
        elif selectedValue == 'false':
            selected = False
        else:
            raise ValueError('expected {"true", "false"} value for selected field, received: %s' % selectedValue)

        # TODO: if someone needs it, he can use the parameter
        # fields to revive the setup_string as well
        # However, this is usually done by the OpenML server
        # and if we are going to duplicate this functionality
        # it needs proper testing

        current = OpenMLTraceIteration(repeat, fold, iteration, None, evaluation, selected)
        trace[(repeat, fold, iteration)] = current

    return OpenMLRunTrace(None, trace)


def _get_cached_run(run_id):
    """Load a run from the cache."""
    cache_dir = config.get_cache_directory()
    run_cache_dir = os.path.join(cache_dir, "runs")
    try:
        run_file = os.path.join(run_cache_dir,
                                "run_%d.xml" % int(run_id))
        with io.open(run_file, encoding='utf8') as fh:
            run = _create_run_from_xml(xml=fh.read())
        return run

    except (OSError, IOError):
        raise OpenMLCacheException("Run file for run id %d not "
                                   "cached" % run_id)


def list_runs(offset=None, size=None, id=None, task=None, setup=None,
              flow=None, uploader=None, tag=None, display_errors=False):
    """List all runs matching all of the given filters.

    Perform API call `/run/list/{filters} <https://www.openml.org/api_docs/#!/run/get_run_list_filters>`_

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

    display_errors : bool, optional (default=None)
        Whether to list runs which have an error (for example a missing
        prediction file).
    Returns
    -------
    list
        List of found runs.
    """

    api_call = "run/list"
    if offset is not None:
        api_call += "/offset/%d" % int(offset)
    if size is not None:
       api_call += "/limit/%d" % int(size)
    if id is not None:
        api_call += "/run/%s" % ','.join([str(int(i)) for i in id])
    if task is not None:
        api_call += "/task/%s" % ','.join([str(int(i)) for i in task])
    if setup is not None:
        api_call += "/setup/%s" % ','.join([str(int(i)) for i in setup])
    if flow is not None:
        api_call += "/flow/%s" % ','.join([str(int(i)) for i in flow])
    if uploader is not None:
        api_call += "/uploader/%s" % ','.join([str(int(i)) for i in uploader])
    if tag is not None:
        api_call += "/tag/%s" % tag
    if display_errors:
        api_call += "/show_errors/true"

    return _list_runs(api_call)


def _list_runs(api_call):
    """Helper function to parse API calls which are lists of runs"""

    xml_string = _perform_api_call(api_call)

    runs_dict = xmltodict.parse(xml_string, force_list=('oml:run',))
    # Minimalistic check if the XML is useful
    if 'oml:runs' not in runs_dict:
        raise ValueError('Error in return XML, does not contain "oml:runs": %s'
                         % str(runs_dict))
    elif '@xmlns:oml' not in runs_dict['oml:runs']:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:runs"/@xmlns:oml: %s'
                         % str(runs_dict))
    elif runs_dict['oml:runs']['@xmlns:oml'] != 'http://openml.org/openml':
        raise ValueError('Error in return XML, value of  '
                         '"oml:runs"/@xmlns:oml is not '
                         '"http://openml.org/openml": %s'
                         % str(runs_dict))

    assert type(runs_dict['oml:runs']['oml:run']) == list, \
        type(runs_dict['oml:runs'])

    runs = dict()
    for run_ in runs_dict['oml:runs']['oml:run']:
        run_id = int(run_['oml:run_id'])
        run = {'run_id': run_id,
               'task_id': int(run_['oml:task_id']),
               'setup_id': int(run_['oml:setup_id']),
               'flow_id': int(run_['oml:flow_id']),
               'uploader': int(run_['oml:uploader'])}

        runs[run_id] = run

    return runs
