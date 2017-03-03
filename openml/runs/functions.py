from collections import defaultdict
import io
import os
import xmltodict
from sklearn.model_selection._search import BaseSearchCV

from .. import config
from ..flows import sklearn_to_flow
from ..exceptions import OpenMLCacheException
from ..util import URLError
from ..tasks.functions import _create_task_from_xml
from .._api_calls import _perform_api_call
from .run import OpenMLRun


# _get_version_info, _get_dict and _create_setup_string are in run.py to avoid
# circular imports



def run_task(task, model):
    """Performs a CV run on the dataset of the given task, using the split.

    Parameters
    ----------
    task : OpenMLTask
        Task to perform.
    model : sklearn model
        a model which has a function fit(X,Y) and predict(X),
        all supervised estimators of scikit learn follow this definition of a model [1]
        [1](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)


    Returns
    -------
    run : OpenMLRun
        Result of the run.
    """
    # TODO move this into its onwn module. While it somehow belongs here, it
    # adds quite a lot of functionality which is better suited in other places!
    # TODO why doesn't this accept a flow as input? - this would make this more flexible!

    dataset = task.get_dataset()
    X, Y = dataset.get_data(target=task.target_name)

    class_labels = task.class_labels
    if class_labels is None:
        raise ValueError('The task has no class labels. This method currently '
                         'only works for tasks with class labels.')

    # execute the run
    run = OpenMLRun(task_id=task.task_id, flow_id=None, dataset_id=dataset.dataset_id, model=model)
    run.data_content, run.trace_content = _run_task_get_arffcontent(model, task, class_labels)

    # now generate the flow
    flow = sklearn_to_flow(model)
    flow_id = flow._ensure_flow_exists()
    if flow_id < 0:
        print("No flow")
        return 0, 2
    config.logger.info(flow_id)

    # attach the flow to the run
    run.flow_id = flow_id

    return run


def _run_task_get_arffcontent(model, task, class_labels):
    X, Y = task.get_X_and_y()
    arff_datacontent = []
    arff_tracecontent = []

    rep_no = 0
    # TODO use different iterator to only provide a single iterator (less
    # methods, less maintenance, less confusion)
    for rep in task.iterate_repeats():
        fold_no = 0
        for fold in rep:
            train_indices, test_indices = fold
            trainX = X[train_indices]
            trainY = Y[train_indices]
            testX = X[test_indices]
            testY = Y[test_indices]

            model.fit(trainX, trainY)
            if isinstance(model, BaseSearchCV):
                _add_results_to_arfftrace(arff_tracecontent, fold_no, model,
                                          rep_no)

            ProbaY = model.predict_proba(testX)
            PredY = model.predict(testX)

            for i in range(0, len(test_indices)):
                arff_line = [rep_no, fold_no, test_indices[i]]
                arff_line.extend(ProbaY[i])
                arff_line.append(class_labels[PredY[i]])
                arff_line.append(class_labels[testY[i]])
                arff_datacontent.append(arff_line)

            fold_no = fold_no + 1
        rep_no = rep_no + 1

    if not isinstance(model, BaseSearchCV):
        arff_tracecontent = None

    return arff_datacontent, arff_tracecontent


def _add_results_to_arfftrace(arff_tracecontent, fold_no, model, rep_no):
    for itt_no in range(0, len(model.cv_results_['mean_test_score'])):
        # we use the string values for True and False, as it is defined in this way by the OpenML server
        selected = 'false'
        if itt_no == model.best_index_:
            selected = 'true'
        test_score = model.cv_results_['mean_test_score'][itt_no]
        arff_line = [rep_no, fold_no, itt_no, test_score, selected]
        for key in model.cv_results_:
            if key.startswith("param_"):
                arff_line.append(str(model.cv_results_[key][itt_no]))
        arff_tracecontent.append(arff_line)


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
        try:
            run_xml = _perform_api_call("run/%d" % run_id)
        except (URLError, UnicodeEncodeError) as e:
            # TODO logger.debug
            print(e)
            raise e

        with io.open(run_file, "w", encoding='utf8') as fh:
            fh.write(run_xml)

    try:
        run = _create_run_from_xml(run_xml)
    except Exception as e:
        # TODO logger.debug
        print("Run ID", run_id)
        raise e

    with io.open(run_file, "w", encoding='utf8') as fh:
        fh.write(run_xml)

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

    predictions_url = None
    for file_dict in run['oml:output_data']['oml:file']:
        if file_dict['oml:name'] == 'predictions':
            predictions_url = file_dict['oml:url']
    if predictions_url is None:
        raise ValueError('No URL to download predictions for run %d in run '
                         'description XML' % run_id)
    evaluations = dict()
    detailed_evaluations = defaultdict(lambda: defaultdict(dict))
    evaluation_flows = dict()
    for evaluation_dict in run['oml:output_data']['oml:evaluation']:
        key = evaluation_dict['oml:name']
        if 'oml:value' in evaluation_dict:
            value = float(evaluation_dict['oml:value'])
        elif 'oml:array_data' in evaluation_dict:
            value = evaluation_dict['oml:array_data']
        else:
            raise ValueError('Could not find keys "value" or "array_data" '
                             'in %s' % str(evaluation_dict.keys()))

        if '@repeat' in evaluation_dict and '@fold' in evaluation_dict:
            repeat = int(evaluation_dict['@repeat'])
            fold = int(evaluation_dict['@fold'])
            repeat_dict = detailed_evaluations[key]
            fold_dict = repeat_dict[repeat]
            fold_dict[fold] = value
        else:
            evaluations[key] = value
            evaluation_flows[key] = flow_id

        evaluation_flows[key] = flow_id

    return OpenMLRun(run_id=run_id, uploader=uploader,
                     uploader_name=uploader_name, task_id=task_id,
                     task_type=task_type,
                     task_evaluation_measure=task_evaluation_measure,
                     flow_id=flow_id, flow_name=flow_name,
                     setup_id=setup_id, setup_string=setup_string,
                     parameter_settings=parameters,
                     dataset_id=dataset_id, predictions_url=predictions_url,
                     evaluations=evaluations,
                     detailed_evaluations=detailed_evaluations)


def _get_cached_run(run_id):
    """Load a run from the cache."""
    cache_dir = config.get_cache_directory()
    run_cache_dir = os.path.join(cache_dir, "runs")
    try:
        run_file = os.path.join(run_cache_dir,
                                "run_%d.xml" % int(run_id))
        with io.open(run_file, encoding='utf8') as fh:
            run = _create_task_from_xml(xml=fh.read())
        return run

    except (OSError, IOError):
        raise OpenMLCacheException("Run file for run id %d not "
                                   "cached" % run_id)


def list_runs(offset=None, size=None, id=None, task=None,
              flow=None, uploader=None, tag=None):
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

    flow : list, optional

    uploader : list, optional

    tag : str, optional

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
    if flow is not None:
        api_call += "/flow/%s" % ','.join([str(int(i)) for i in flow])
    if uploader is not None:
        api_call += "/uploader/%s" % ','.join([str(int(i)) for i in uploader])
    if tag is not None:
        api_call += "/tag/%s" % tag

    return _list_runs(api_call)


def _list_runs(api_call):
    """Helper function to parse API calls which are lists of runs"""

    xml_string = _perform_api_call(api_call)

    runs_dict = xmltodict.parse(xml_string)
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

    if isinstance(runs_dict['oml:runs']['oml:run'], list):
        runs_list = runs_dict['oml:runs']['oml:run']
    elif isinstance(runs_dict['oml:runs']['oml:run'], dict):
        runs_list = [runs_dict['oml:runs']['oml:run']]
    else:
        raise TypeError()

    runs = dict()
    for run_ in runs_list:
        run_id = int(run_['oml:run_id'])
        run = {'run_id': run_id,
               'task_id': int(run_['oml:task_id']),
               'setup_id': int(run_['oml:setup_id']),
               'flow_id': int(run_['oml:flow_id']),
               'uploader': int(run_['oml:uploader'])}

        runs[run_id] = run

    return runs
