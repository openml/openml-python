from collections import OrderedDict
import io
import re
import os

from oslo_concurrency import lockutils
import xmltodict

from ..exceptions import OpenMLCacheException
from ..datasets import get_dataset
from .task import (
    OpenMLClassificationTask,
    OpenMLClusteringTask,
    OpenMLLearningCurveTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask
)
import openml.utils
import openml._api_calls

TASKS_CACHE_DIR_NAME = 'tasks'

def _get_cached_tasks():
    """Return a dict of all the tasks which are cached locally.
    Returns
    -------
    tasks : OrderedDict
        A dict of all the cached tasks. Each task is an instance of
        OpenMLTask.
    """
    tasks = OrderedDict()

    task_cache_dir = openml.utils._create_cache_directory(TASKS_CACHE_DIR_NAME)
    directory_content = os.listdir(task_cache_dir)
    directory_content.sort()
    # Find all dataset ids for which we have downloaded the dataset
    # description

    for filename in directory_content:
        if not re.match(r"[0-9]*", filename):
            continue

        tid = int(filename)
        tasks[tid] = _get_cached_task(tid)

    return tasks



def _get_cached_task(tid):
    """Return a cached task based on the given id.

    Parameters
    ----------
    tid : int
        Id of the task.

    Returns
    -------
    OpenMLTask
    """
    tid_cache_dir = openml.utils._create_cache_directory_for_id(
        TASKS_CACHE_DIR_NAME,
        tid
    )

    try:
        with io.open(os.path.join(tid_cache_dir, "task.xml"), encoding='utf8') as fh:
            return _create_task_from_xml(fh.read())
    except (OSError, IOError):
        openml.utils._remove_cache_dir_for_id(TASKS_CACHE_DIR_NAME, tid_cache_dir)
        raise OpenMLCacheException("Task file for tid %d not "
                                   "cached" % tid)


def _get_estimation_procedure_list():
    """Return a list of all estimation procedures which are on OpenML.
    Returns
    -------
    procedures : list
        A list of all estimation procedures. Every procedure is represented by
        a dictionary containing the following information: id, task type id,
        name, type, repeats, folds, stratified.
    """

    xml_string = openml._api_calls._perform_api_call("estimationprocedure/list")
    procs_dict = xmltodict.parse(xml_string)
    # Minimalistic check if the XML is useful
    if 'oml:estimationprocedures' not in procs_dict:
        raise ValueError('Error in return XML, does not contain tag '
                         'oml:estimationprocedures.')
    elif '@xmlns:oml' not in procs_dict['oml:estimationprocedures']:
        raise ValueError('Error in return XML, does not contain tag '
                         '@xmlns:oml as a child of oml:estimationprocedures.')
    elif procs_dict['oml:estimationprocedures']['@xmlns:oml'] != \
            'http://openml.org/openml':
        raise ValueError('Error in return XML, value of '
                         'oml:estimationprocedures/@xmlns:oml is not '
                         'http://openml.org/openml, but %s' %
                         str(procs_dict['oml:estimationprocedures']['@xmlns:oml']))

    procs = []
    for proc_ in procs_dict['oml:estimationprocedures']['oml:estimationprocedure']:
        procs.append(
            {
                'id': int(proc_['oml:id']),
                'task_type_id': int(proc_['oml:ttid']),
                'name': proc_['oml:name'],
                'type': proc_['oml:type'],
            }
        )

    return procs


def list_tasks(task_type_id=None, offset=None, size=None, tag=None, **kwargs):
    """
    Return a number of tasks having the given tag and task_type_id
    Parameters
    ----------
    Filter task_type_id is separated from the other filters because
    it is used as task_type_id in the task description, but it is named
    type when used as a filter in list tasks call.
    task_type_id : int, optional
        ID of the task type as detailed
        `here <https://www.openml.org/search?type=task_type>`_.
        - Supervised classification: 1
        - Supervised regression: 2
        - Learning curve: 3
        - Supervised data stream classification: 4
        - Clustering: 5
        - Machine Learning Challenge: 6
        - Survival Analysis: 7
        - Subgroup Discovery: 8
    offset : int, optional
        the number of tasks to skip, starting from the first
    size : int, optional
        the maximum number of tasks to show
    tag : str, optional
        the tag to include
    kwargs: dict, optional
        Legal filter operators: data_tag, status, data_id, data_name, number_instances, number_features,
        number_classes, number_missing_values.
    Returns
    -------
    dict
        All tasks having the given task_type_id and the give tag. Every task is
        represented by a dictionary containing the following information:
        task id, dataset id, task_type and status. If qualities are calculated
        for the associated dataset, some of these are also returned.
    """
    return openml.utils._list_all(_list_tasks, task_type_id=task_type_id, offset=offset, size=size, tag=tag, **kwargs)


def _list_tasks(task_type_id=None, **kwargs):
    """
    Perform the api call to return a number of tasks having the given filters.
    Parameters
    ----------
    Filter task_type_id is separated from the other filters because
    it is used as task_type_id in the task description, but it is named
    type when used as a filter in list tasks call.
    task_type_id : int, optional
        ID of the task type as detailed
        `here <https://www.openml.org/search?type=task_type>`_.
        - Supervised classification: 1
        - Supervised regression: 2
        - Learning curve: 3
        - Supervised data stream classification: 4
        - Clustering: 5
        - Machine Learning Challenge: 6
        - Survival Analysis: 7
        - Subgroup Discovery: 8
    kwargs: dict, optional
        Legal filter operators: tag, task_id (list), data_tag, status, limit,
        offset, data_id, data_name, number_instances, number_features,
        number_classes, number_missing_values.
    Returns
    -------
    dict
    """
    api_call = "task/list"
    if task_type_id is not None:
        api_call += "/type/%d" % int(task_type_id)
    if kwargs is not None:
        for operator, value in kwargs.items():
            if operator == 'task_id':
                value = ','.join([str(int(i)) for i in value])
            api_call += "/%s/%s" % (operator, value)
    return __list_tasks(api_call)


def __list_tasks(api_call):

    xml_string = openml._api_calls._perform_api_call(api_call)
    tasks_dict = xmltodict.parse(xml_string, force_list=('oml:task', 'oml:input'))
    # Minimalistic check if the XML is useful
    if 'oml:tasks' not in tasks_dict:
        raise ValueError('Error in return XML, does not contain "oml:runs": %s'
                         % str(tasks_dict))
    elif '@xmlns:oml' not in tasks_dict['oml:tasks']:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:runs"/@xmlns:oml: %s'
                         % str(tasks_dict))
    elif tasks_dict['oml:tasks']['@xmlns:oml'] != 'http://openml.org/openml':
        raise ValueError('Error in return XML, value of  '
                         '"oml:runs"/@xmlns:oml is not '
                         '"http://openml.org/openml": %s'
                         % str(tasks_dict))

    assert type(tasks_dict['oml:tasks']['oml:task']) == list, \
        type(tasks_dict['oml:tasks'])

    tasks = dict()
    procs = _get_estimation_procedure_list()
    proc_dict = dict((x['id'], x) for x in procs)

    for task_ in tasks_dict['oml:tasks']['oml:task']:
        tid = None
        try:
            tid = int(task_['oml:task_id'])
            task = {'tid': tid,
                    'ttid': int(task_['oml:task_type_id']),
                    'did': int(task_['oml:did']),
                    'name': task_['oml:name'],
                    'task_type': task_['oml:task_type'],
                    'status': task_['oml:status']}

            # Other task inputs
            for input in task_.get('oml:input', list()):
                if input['@name'] == 'estimation_procedure':
                    task[input['@name']] = proc_dict[int(input['#text'])]['name']
                else:
                    value = input.get('#text')
                    task[input['@name']] = value

            # The number of qualities can range from 0 to infinity
            for quality in task_.get('oml:quality', list()):
                if '#text' not in quality:
                    quality_value = 0.0
                else:
                    quality['#text'] = float(quality['#text'])
                    if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                        quality['#text'] = int(quality['#text'])
                    quality_value = quality['#text']
                task[quality['@name']] = quality_value
            tasks[tid] = task
        except KeyError as e:
            if tid is not None:
                raise KeyError(
                    "Invalid xml for task %d: %s\nFrom %s" % (
                        tid, e, task_
                    )
                )
            else:
                raise KeyError('Could not find key %s in %s!' % (e, task_))

    return tasks


def get_tasks(task_ids):
    """Download tasks.
    This function iterates :meth:`openml.tasks.get_task`.
    Parameters
    ----------
    task_ids : iterable
        Integers representing task ids.
    Returns
    -------
    list
    """
    tasks = []
    for task_id in task_ids:
        tasks.append(get_task(task_id))
    return tasks


def get_task(task_id):
    """Download the OpenML task for a given task ID.
    Parameters
    ----------
    task_id : int
        The OpenML task id.
    """
    task_id = int(task_id)

    with lockutils.external_lock(
            name='task.functions.get_task:%d' % task_id,
            lock_path=openml.utils._create_lockfiles_dir(),
    ):
        tid_cache_dir = openml.utils._create_cache_directory_for_id(
            TASKS_CACHE_DIR_NAME, task_id,
        )

        try:
            task = _get_task_description(task_id)
            dataset = get_dataset(task.dataset_id)
            # Clustering tasks do not have class labels
            # and do not offer download_split
            if isinstance(task, OpenMLSupervisedTask):
                task.download_split()
                if isinstance(task, OpenMLClassificationTask):
                    task.class_labels = \
                        dataset.retrieve_class_labels(task.target_name)
        except Exception as e:
            openml.utils._remove_cache_dir_for_id(
                TASKS_CACHE_DIR_NAME,
                tid_cache_dir,
            )
            raise e

    return task


def _get_task_description(task_id):

    try:
        return _get_cached_task(task_id)
    except OpenMLCacheException:
        xml_file = os.path.join(
            openml.utils._create_cache_directory_for_id(
                TASKS_CACHE_DIR_NAME,
                task_id,
            ),
            "task.xml",
        )
        task_xml = openml._api_calls._perform_api_call("task/%d" % task_id)

        with io.open(xml_file, "w", encoding='utf8') as fh:
            fh.write(task_xml)
        return _create_task_from_xml(task_xml)


def _create_task_from_xml(xml):
    """Create a task given a xml string.

    Parameters
    ----------
    xml : string
        Task xml representation.

    Returns
    -------
    OpenMLTask
    """
    dic = xmltodict.parse(xml)["oml:task"]
    estimation_parameters = dict()
    inputs = dict()
    # Due to the unordered structure we obtain, we first have to extract
    # the possible keys of oml:input; dic["oml:input"] is a list of
    # OrderedDicts

    # Check if there is a list of inputs
    if isinstance(dic["oml:input"], list):
        for input_ in dic["oml:input"]:
            name = input_["@name"]
            inputs[name] = input_
    # Single input case
    elif isinstance(dic["oml:input"], dict):
        name = dic["oml:input"]["@name"]
        inputs[name] = dic["oml:input"]

    evaluation_measures = None
    if 'evaluation_measures' in inputs:
        evaluation_measures = inputs["evaluation_measures"][
            "oml:evaluation_measures"]["oml:evaluation_measure"]

    task_type = dic["oml:task_type"]
    common_kwargs = {
        'task_id': dic["oml:task_id"],
        'task_type': task_type,
        'task_type_id': dic["oml:task_type_id"],
        'data_set_id': inputs["source_data"][
            "oml:data_set"]["oml:data_set_id"],
        'evaluation_measure': evaluation_measures,
    }
    if task_type in (
        "Supervised Classification",
        "Supervised Regression",
        "Learning Curve"
    ):
        # Convert some more parameters
        for parameter in \
                inputs["estimation_procedure"]["oml:estimation_procedure"][
                    "oml:parameter"]:
            name = parameter["@name"]
            text = parameter.get("#text", "")
            estimation_parameters[name] = text

        common_kwargs['estimation_procedure_type'] = inputs[
            "estimation_procedure"][
            "oml:estimation_procedure"]["oml:type"]
        common_kwargs['estimation_parameters'] = estimation_parameters
        common_kwargs['target_name'] = inputs[
                "source_data"]["oml:data_set"]["oml:target_feature"]
        common_kwargs['data_splits_url'] = inputs["estimation_procedure"][
                "oml:estimation_procedure"]["oml:data_splits_url"]

    cls = {
        "Supervised Classification": OpenMLClassificationTask,
        "Supervised Regression": OpenMLRegressionTask,
        "Clustering": OpenMLClusteringTask,
        "Learning Curve": OpenMLLearningCurveTask,
    }.get(task_type)
    if cls is None:
        raise NotImplementedError('Task type %s not supported.')
    return cls(**common_kwargs)
