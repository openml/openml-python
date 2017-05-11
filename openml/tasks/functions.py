import io
import os
import re
from collections import OrderedDict
import xmltodict

from ..exceptions import OpenMLCacheException
from .. import datasets
from .task import OpenMLTask, _create_task_cache_dir
from .. import config
from .._api_calls import _perform_api_call


def _get_cached_tasks():
    tasks = OrderedDict()
    cache_dir = config.get_cache_directory()

    task_cache_dir = os.path.join(cache_dir, "tasks")
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
    cache_dir = config.get_cache_directory()
    task_cache_dir = os.path.join(cache_dir, "tasks")
    task_file = os.path.join(task_cache_dir, str(tid), "task.xml")

    try:
        with io.open(task_file, encoding='utf8') as fh:
            task = _create_task_from_xml(xml=fh.read())
        return task
    except (OSError, IOError):
        raise OpenMLCacheException("Task file for tid %d not "
                                   "cached" % tid)


def _get_estimation_procedure_list():
    """Return a list of all estimation procedures which are on OpenML.

    Returns
    -------
    procedures : list
        A list of all estimation procedures. Every procedure is represented by a
        dictionary containing the following information: id,
        task type id, name, type, repeats, folds, stratified.
    """

    xml_string = _perform_api_call("estimationprocedure/list")
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
        proc = {'id': int(proc_['oml:id']),
                'task_type_id': int(proc_['oml:ttid']),
                'name': proc_['oml:name'],
                'type': proc_['oml:type']}

        procs.append(proc)

    return procs


def list_tasks(task_type_id=None, offset=None, size=None, tag=None):
    """Return a number of tasks having the given tag and task_type_id

    Parameters
    ----------
    task_type_id : int, optional
        ID of the task type as detailed
        `here <https://www.openml.org/search?type=task_type>`_.
    offset : int, optional
        the number of tasks to skip, starting from the first
    size : int, optional
        the maximum number of tasks to show
    tag : str, optional
        the tag to include

    Returns
    -------
    dict
        All tasks having the given task_type_id and the give tag. Every task is
        represented by a dictionary containing the following information:
        task id, dataset id, task_type and status. If qualities are calculated
        for the associated dataset, some of these are also returned.
    """
    api_call = "task/list"
    if task_type_id is not None:
        api_call += "/type/%d" % int(task_type_id)

    if offset is not None:
        api_call += "/offset/%d" % int(offset)

    if size is not None:
        api_call += "/limit/%d" % int(size)

    if tag is not None:
        api_call += "/tag/%s" % tag

    return _list_tasks(api_call)


def _list_tasks(api_call):
    xml_string = _perform_api_call(api_call)
    with open('/tmp/list_tasks.xml', 'w') as fh:
        fh.write(xml_string)
    tasks_dict = xmltodict.parse(xml_string)
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

    try:
        tasks = dict()
        procs = _get_estimation_procedure_list()
        proc_dict = dict((x['id'], x) for x in procs)
        for task_ in tasks_dict['oml:tasks']['oml:task']:
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
                quality['#text'] = float(quality['#text'])
                if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                    quality['#text'] = int(quality['#text'])
                task[quality['@name']] = quality['#text']
            tasks[tid] = task
    except KeyError as e:
        raise KeyError("Invalid xml for task: %s" % e)

    return tasks


def get_task(task_id):
    """Download the OpenML task for a given task ID.

    Parameters
    ----------
    task_id : int
        The OpenML task id.
    """
    try:
        task_id = int(task_id)
    except:
        raise ValueError("Task ID is neither an Integer nor can be "
                         "cast to an Integer.")

    xml_file = os.path.join(_create_task_cache_dir(task_id),
                            "task.xml")

    try:
        with io.open(xml_file, encoding='utf8') as fh:
            task = _create_task_from_xml(fh.read())

    except (OSError, IOError):
        task_xml = _perform_api_call("task/%d" % task_id)

        with io.open(xml_file, "w", encoding='utf8') as fh:
            fh.write(task_xml)

        task = _create_task_from_xml(task_xml)

    # TODO extract this to a function
    task.download_split()
    dataset = datasets.get_dataset(task.dataset_id)

    # TODO look into either adding the class labels to task xml, or other
    # way of reading it.
    class_labels = dataset.retrieve_class_labels(task.target_name)
    task.class_labels = class_labels
    return task


def _create_task_from_xml(xml):
    dic = xmltodict.parse(xml)["oml:task"]

    estimation_parameters = dict()
    inputs = dict()
    # Due to the unordered structure we obtain, we first have to extract
    # the possible keys of oml:input; dic["oml:input"] is a list of
    # OrderedDicts
    for input_ in dic["oml:input"]:
        name = input_["@name"]
        inputs[name] = input_

    evaluation_measures = None
    if 'evaluation_measures' in inputs:
        evaluation_measures = inputs["evaluation_measures"]["oml:evaluation_measures"]["oml:evaluation_measure"]


    # Convert some more parameters
    for parameter in \
            inputs["estimation_procedure"]["oml:estimation_procedure"][
                "oml:parameter"]:
        name = parameter["@name"]
        text = parameter.get("#text", "")
        estimation_parameters[name] = text

    return OpenMLTask(
        dic["oml:task_id"], dic['oml:task_type_id'], dic["oml:task_type"],
        inputs["source_data"]["oml:data_set"]["oml:data_set_id"],
        inputs["source_data"]["oml:data_set"]["oml:target_feature"],
        inputs["estimation_procedure"]["oml:estimation_procedure"][
            "oml:type"],
        inputs["estimation_procedure"]["oml:estimation_procedure"][
            "oml:data_splits_url"], estimation_parameters,
        evaluation_measures, None)
