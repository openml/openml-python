from collections import defaultdict
import os

import sklearn
import xmltodict

from .._api_calls import _perform_api_call
from .. import config
from ..flows import OpenMLFlow
from .run import OpenMLRun
from ..exceptions import OpenMLCacheException
from ..util import URLError


def run_task(task, model, dependencies=None):
    """Performs a CV run on the dataset of the given task, using the split.

    Parameters
    ----------
    task : OpenMLTask
        Task to perform.
    model : sklearn model or OpenMLFlow
        a model which has a function fit(X,Y), predict(X) and predict_proba(X,
        most supervised estimators of scikit learn follow this definition of a
        model [1]
        [1](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)
    dependencies : str
        Dependencies are passed to flow generation when a model is passed.

    Returns
    -------
    run : OpenMLRun
        Result of the run.
    """
    # TODO test this function and it's subfunctions

    if not isinstance(model, OpenMLFlow):
        # Try to extract the source code, name, description, external version
        # etc from the package in order to usefully create the flow!
        external_version = 'sklearn_' + sklearn.__version__
        flow = OpenMLFlow(model=model, external_version=external_version,
                          description='Automatically generated.',
                          dependencies=dependencies)
        # Triggers generating the name which is needed for
        # _ensure_flow_exists() below!
        flow._generate_flow_xml()
    else:
        flow = model

    flow_id = flow._ensure_flow_exists()
    if flow_id < 0:
        raise ValueError('Trying to run a task with an unregistered flow. '
                         'Register the flow first.')

    class_labels = task.class_labels
    if class_labels is None:
        raise ValueError('The task has no class labels. This method currently '
                         'only works for tasks with class labels.')

    run = OpenMLRun(task.task_id, flow_id, task.dataset_id)
    run.data_content = _run_task_get_arffcontent(model, task, class_labels)

    # The model will not be uploaded at the moment, but used to get the
    # hyperparameter values when uploading the run
    X, Y = task.get_X_and_Y()
    run.model = model.fit(X, Y)

    return run


def _run_task_get_arffcontent(model, task, class_labels):
    X, Y = task.get_X_and_Y()
    arff_datacontent = []
    rep_no = 0
    # TODO use different iterator to only provide a single iterator (less
    # methods, less maintenance, less confusion)
    for rep in task.iterate_repeats():
        fold_no = 0
        for fold in rep:
            # TODO Put into its own function for testability!
            train_indices, test_indices = fold
            trainX = X[train_indices]
            trainY = Y[train_indices]
            testX = X[test_indices]
            testY = Y[test_indices]

            model.fit(trainX, trainY)
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

    return arff_datacontent


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
            return_code, run_xml = _perform_api_call("run/%d" % run_id)
        except (URLError, UnicodeEncodeError) as e:
            # TODO logger.debug
            print(e)
            raise e

        with open(run_file, "w") as fh:
            fh.write(run_xml)

    try:
        run = _create_run_from_xml(run_xml)
    except Exception as e:
        # TODO logger.debug
        print("Run ID", run_id)
        raise e

    with open(run_file, "w") as fh:
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
    task_evaluation_measure = run['oml:task_evaluation_measure']
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
        flow_id = int(evaluation_dict['oml:flow_id'])
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
    for cache_dir in [config.get_cache_directory(),
                      config.get_private_directory()]:
        run_cache_dir = os.path.join(cache_dir, "runs")
        try:
            run_file = os.path.join(run_cache_dir,
                                    "run_%d.xml" % int(run_id))
            with open(run_file) as fh:
                run = _create_run_from_xml(xml=fh.read())
            return run

        except (OSError, IOError):
            continue

    raise OpenMLCacheException("Run file for run id %d not "
                               "cached" % run_id)


def list_runs_by_filters(id=None, task=None, flow=None,
                         uploader=None):
    """List all runs matching all of the given filters.

    Perform API call `/run/list/{filters} <http://www.openml.org/api_docs/#!/run/get_run_list_filters>`_

    Parameters
    ----------
    id : int or list

    task : int or list

    flow : int or list

    uploader : int or list

    Returns
    -------
    list
        List of found runs.
    """

    value = []
    by = []

    if id is not None:
        value.append(id)
        by.append('run')
    if task is not None:
        value.append(task)
        by.append('task')
    if flow is not None:
        value.append(flow)
        by.append('flow')
    if uploader is not None:
        value.append(uploader)
        by.append('uploader')

    if len(value) == 0:
        raise ValueError('At least one argument out of task, flow, uploader '
                         'must have a different value than None')

    api_call = "run/list"
    for id_, by_ in zip(value, by):
        if isinstance(id_, list):
            for i in range(len(id_)):
                # Type checking to avoid bad calls to the server
                id_[i] = str(int(id_[i]))
            id_ = ','.join(id_)
        else:
            # Only type checking here
            id_ = int(id_)

        if by_ is None:
            raise ValueError("Argument 'by' must not contain None!")
        api_call = "%s/%s/%s" % (api_call, by_, id_)

    return _list_runs(api_call)


def list_runs_by_tag(tag):
    """List runs by tag.

    Perform API call `/run/list/tag/{tag} <http://www.openml.org/api_docs/#!/run/get_run_list_tag_tag>`_

    Parameters
    ----------
    tag : str

    Returns
    -------
    list
        List of found runs.
    """
    return _list_runs_by(tag, 'tag')


def list_runs(run_ids):
    """List runs by their ID.

    Perform API call `/run/list/run/{ids} <http://www.openml.org/api_docs/#!/run/get_run_list_run_ids>`_

    Parameters
    ----------
    run_id : int or list

    Returns
    -------
    list
        List of found runs.
    """
    return _list_runs_by(run_ids, 'run')


def list_runs_by_task(task_id):
    """List runs by task.

    Perform API call `/run/list/task/{ids} <http://www.openml.org/api_docs/#!/run/get_run_list_task_ids>`_

    Parameters
    ----------
    task_id : int or list

    Returns
    -------
    list
        List of found runs.
    """
    return _list_runs_by(task_id, 'task')


def list_runs_by_flow(flow_id):
    """List runs by flow.

    Perform API call `/run/list/flow/{ids} <http://www.openml.org/api_docs/#!/run/get_run_list_flow_ids>`_

    Parameters
    ----------
    flow_id : int or list

    Returns
    -------
    list
        List of found runs.
    """
    return _list_runs_by(flow_id, 'flow')


def list_runs_by_uploader(uploader_id):
    """List runs by uploader.

    Perform API call `/run/list/uploader/{ids} <http://www.openml.org/api_docs/#!/run/get_run_list_uploader_ids>`_

    Parameters
    ----------
    uploader_id : int or list

    Returns
    -------
    list
        List of found runs.
    """
    return _list_runs_by(uploader_id, 'uploader')


def _list_runs_by(id_, by):
    """Helper function to create API call strings.

    Helper for the following api calls:

    * http://www.openml.org/api_docs/#!/run/get_run_list_task_ids
    * http://www.openml.org/api_docs/#!/run/get_run_list_run_ids
    * http://www.openml.org/api_docs/#!/run/get_run_list_tag_tag
    * http://www.openml.org/api_docs/#!/run/get_run_list_uploader_ids
    * http://www.openml.org/api_docs/#!/run/get_run_list_flow_ids

    All of these allow either an integer as ID or a list of integers. Their
    name follows the convention run/list/{by}/{id}

    Parameters
    ----------
    id_ : int or list

    by : str

    Returns
    -------
    list
        List of found runs.

    """

    if isinstance(id_, list):
        for i in range(len(id_)):
            # Type checking to avoid bad calls to the server
            id_[i] = str(int(id_[i]))
        id_ = ','.join(id_)
    elif by == 'tag':
        pass
    else:
        id_ = int(id_)

    api_call = "run/list"
    if by is not None:
        api_call += "/%s" % by
    api_call = "%s/%s" % (api_call, id_)
    return _list_runs(api_call)


def _list_runs(api_call):
    """Helper function to parse API calls which are lists of runs"""

    return_code, xml_string = _perform_api_call(api_call)

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

    runs = []
    for run_ in runs_list:
        run = {'run_id': int(run_['oml:run_id']),
               'task_id': int(run_['oml:task_id']),
               'setup_id': int(run_['oml:setup_id']),
               'flow_id': int(run_['oml:flow_id']),
               'uploader': int(run_['oml:uploader'])}

        runs.append(run)
    runs.sort(key=lambda t: t['run_id'])

    return runs
