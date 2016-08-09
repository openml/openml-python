import time
import arff
import xmltodict
from collections import OrderedDict, defaultdict
import sys
import os

from .. import config
from ..flows import OpenMLFlow
from ..exceptions import OpenMLCacheException
from ..util import URLError
from ..tasks import get_task
from ..tasks.task_functions import _create_task_from_xml
from .._api_calls import _perform_api_call


class OpenMLRun(object):
    """OpenML Run: result of running a model on an openml dataset.

    Parameters
    ----------
    FIXME

    """
    def __init__(self, task_id, flow_id, setup_string, dataset_id, files=None,
                 setup_id=None, tags=None, run_id=None, uploader=None,
                 uploader_name=None, evaluations=None,
                 detailed_evaluations=None, data_content=None,
                 model=None, task_type=None, task_evaluation_measure=None,
                 flow_name=None, parameter_settings=None, predictions_url=None):
        self.run_id = run_id
        self.uploader = uploader
        self.uploader_name = uploader_name
        self.task_id = task_id
        self.task_type = task_type
        self.task_evaluation_measure = task_evaluation_measure
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.setup_id = setup_id
        self.setup_string = setup_string
        self.parameter_settings = parameter_settings
        self.dataset_id = dataset_id
        self.predictions_url = predictions_url
        self.evaluations = evaluations
        self.detailed_evaluations = detailed_evaluations
        self.data_content = data_content
        self.model = model

    def _generate_arff(self):
        """Generates an arff for upload to server.

        Returns
        -------
        arf_dict : dictionary
            Dictionary representation of an ARFF data format containing
            predictions and confidences.
        """
        run_environment = (_get_version_information() +
                           [time.strftime("%c")] + ['Created by run_task()'])
        task = get_task(self.task_id)
        class_labels = task.class_labels

        arff_dict = {}
        arff_dict['attributes'] = [('repeat', 'NUMERIC'),  # lowercase 'numeric' gives an error
                                   ('fold', 'NUMERIC'),
                                   ('row_id', 'NUMERIC')] + \
            [('confidence.' + class_labels[i], 'NUMERIC') for i in range(len(class_labels))] +\
            [('prediction', class_labels),
             ('correct', class_labels)]
        arff_dict['data'] = self.data_content
        arff_dict['description'] = "\n".join(run_environment)
        arff_dict['relation'] = 'openml_task_' + str(task.task_id) + '_predictions'
        return arff_dict

    def publish(self):
        """Publish a run to the OpenML server.

        Uploads the results of a run to OpenML.
        """
        predictions = arff.dumps(self._generate_arff())
        description_xml = self._create_description_xml()
        data = {'predictions': ("predictions.csv", predictions),
                'description': ("description.xml", description_xml)}
        return_code, return_value = _perform_api_call(
            "/run/", file_elements=data)
        return return_code, return_value

    def _create_description_xml(self):
        """Create xml representation of run for upload.

        Returns
        -------
        xml_string : string
            XML description of run.
        """
        run_environment = _get_version_information()
        setup_string = ''  # " ".join(sys.argv);

        parameter_settings = self.model.get_params()
        # as a tag, it must be of the form ([a-zA-Z0-9_\-\.])+
        # so we format time from 'mm/dd/yy hh:mm:ss' to 'mm-dd-yy_hh.mm.ss'
        well_formatted_time = time.strftime("%c").replace(
            ' ', '_').replace('/', '-').replace(':', '.')
        tags = run_environment + [well_formatted_time] + ['run_task'] + \
            [self.model.__module__ + "." + self.model.__class__.__name__]
        description = _to_dict(
            self.task_id, self.flow_id, setup_string, parameter_settings, tags)
        description_xml = xmltodict.unparse(description, pretty=True)
        return description_xml


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
    flow = OpenMLFlow(model=model)
    flow_id = flow._ensure_flow_exists()
    if(flow_id < 0):
        print("No flow")
        return 0, 2
    print(flow_id)

    #runname = "t" + str(task.task_id) + "_" + str(model)
    arff_datacontent = []

    dataset = task.get_dataset()
    X, Y = dataset.get_data(target=task.target_feature)

    class_labels = task.class_labels
    if class_labels is None:
        raise ValueError('The task has no class labels. This method currently '
                         'only works for tasks with class labels.')
    setup_string = _create_setup_string(model)

    run = OpenMLRun(task.task_id, flow_id, setup_string, dataset.id)

    train_times = []

    rep_no = 0
    for rep in task.iterate_repeats():
        fold_no = 0
        for fold in rep:
            train_indices, test_indices = fold
            trainX = X[train_indices]
            trainY = Y[train_indices]
            testX = X[test_indices]
            testY = Y[test_indices]

            start_time = time.time()
            model.fit(trainX, trainY)
            ProbaY = model.predict_proba(testX)
            PredY = model.predict(testX)
            end_time = time.time()

            train_times.append(end_time - start_time)

            for i in range(0, len(test_indices)):
                arff_line = [rep_no, fold_no, test_indices[i],
                             class_labels[PredY[i]], class_labels[testY[i]]]
                arff_line[3:3] = ProbaY[i]
                arff_datacontent.append(arff_line)

            fold_no = fold_no + 1
        rep_no = rep_no + 1

    run.data_content = arff_datacontent
    run.model = model.fit(X, Y)
    return run


def _to_dict(taskid, flow_id, setup_string, parameter_settings, tags):
    """ Creates a dictionary corresponding to the desired xml desired by openML

    Parameters
    ----------
    taskid : int
        the identifier of the task
    setup_string : string
        a CLI string which can invoke the learning with the correct parameter settings
    parameter_settings : array of dicts
        each dict containing keys name, value and component, one per parameter setting
    tags : array of strings
        information that give a description of the run, must conform to
        regex ``([a-zA-Z0-9_\-\.])+``

    Returns
    -------
    result : an array with version information of the above packages
    """
    description = OrderedDict()
    description['oml:run'] = OrderedDict()
    description['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    description['oml:run']['oml:task_id'] = taskid
    description['oml:run']['oml:flow_id'] = flow_id

    params = []
    for k, v in parameter_settings.items():
        param_dict = OrderedDict()
        param_dict['oml:name'] = k
        param_dict['oml:value'] = ('None' if v is None else v)
        params.append(param_dict)

    description['oml:run']['oml:parameter_setting'] = params
    description['oml:run']['oml:tag'] = tags  # Tags describing the run
    #description['oml:run']['oml:output_data'] = 0;
    # all data that was output of this run, which can be evaluation scores
    # (though those are also calculated serverside)
    # must be of special data type
    return description


def _create_setup_string(model):
    """Create a string representing the model"""
    run_environment = " ".join(_get_version_information())
    # fixme str(model) might contain (...)
    return run_environment + " " + str(model)


# This can possibly be done by a package such as pyxb, but I could not get
# it to work properly.
def _get_version_information():
    """Gets versions of python, sklearn, numpy and scipy, returns them in an array,

    Returns
    -------
    result : an array with version information of the above packages
    """
    import sklearn
    import scipy
    import numpy

    major, minor, micro, _, _ = sys.version_info
    python_version = 'Python_{}.'.format(
        ".".join([str(major), str(minor), str(micro)]))
    sklearn_version = 'Sklearn_{}.'.format(sklearn.__version__)
    numpy_version = 'NumPy_{}.'.format(numpy.__version__)
    scipy_version = 'SciPy_{}.'.format(scipy.__version__)

    return [python_version, sklearn_version, numpy_version, scipy_version]


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
    run_file = os.path.join(config.get_cache_directory(), "runs", "run_%d.xml" % run_id)

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
    for cache_dir in [config.get_cache_directory(), config.get_private_directory()]:
        run_cache_dir = os.path.join(cache_dir, "runs")
        try:
            run_file = os.path.join(run_cache_dir,
                                    "run_%d.xml" % int(run_id))
            with open(run_file) as fh:
                run = _create_task_from_xml(xml=fh.read())
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
