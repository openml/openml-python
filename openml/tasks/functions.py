import os
import re
from collections import OrderedDict
import xmltodict

from ..util import URLError
from ..exceptions import OpenMLCacheException
from .. import datasets
from .task import OpenMLTask, _create_task_cache_dir
from .. import config
from .._api_calls import _perform_api_call


def _get_cached_tasks():
    tasks = OrderedDict()
    for cache_dir in [config.get_cache_directory(), config.get_private_directory()]:

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
    for cache_dir in [config.get_cache_directory(), config.get_private_directory()]:
        task_cache_dir = os.path.join(cache_dir, "tasks")
        task_file = os.path.join(task_cache_dir, str(tid), "task.xml")

        try:
            with open(task_file) as fh:
                task = _create_task_from_xml(xml=fh.read())
            return task
        except (OSError, IOError):
            continue

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

    return_code, xml_string = _perform_api_call(
        "estimationprocedure/list")
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


def list_tasks_by_type(task_type_id):
    """Return a list of all tasks for a given tasks type which are on OpenML.

    Parameters
    ----------
    task_type_id : int
        ID of the task type as detailed
        `here <http://www.openml.org/search?type=task_type>`_.

    Returns
    -------
    list
        A list of all tasks of the given task type. Every task is represented by
        a dictionary containing the following information: task id,
        dataset id, task_type and status. If qualities are calculated for
        the associated dataset, some of these are also returned.
    """
    try:
        task_type_id = int(task_type_id)
    except:
        raise ValueError("Task Type ID is neither an Integer nor can be "
                         "cast to an Integer.")
    return _list_tasks("task/list/type/%d" % task_type_id)


def list_tasks_by_tag(tag):
    """Return all tasks having the given tag

    Parameters
    ----------
    tag : str

    Returns
    -------
    list
        A list of all tasks having a give tag. Every task is represented by
        a dictionary containing the following information: task id,
        dataset id, task_type and status. If qualities are calculated for
        the associated dataset, some of these are also returned.
    """
    return _list_tasks("task/list/tag/%s" % tag)


def list_tasks():
    """Return a list of all tasks which are on OpenML.

    Returns
    -------
    list
        A list of all tasks. Every task is represented by a
        dictionary containing the following information: task id,
        dataset id, task_type and status. If qualities are calculated for
        the associated dataset, some of these are also returned.
    """
    return _list_tasks('task/list')


def _list_tasks(api_call):
    return_code, xml_string = _perform_api_call(api_call)
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
        tasks = []
        procs = _get_estimation_procedure_list()
        proc_dict = dict((x['id'], x) for x in procs)
        for task_ in tasks_dict['oml:tasks']['oml:task']:
            task = {'tid': int(task_['oml:task_id']),
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
            tasks.append(task)
    except KeyError as e:
        raise KeyError("Invalid xml for task: %s" % e)

    tasks.sort(key=lambda t: t['tid'])

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
        with open(xml_file) as fh:
            task = _create_task_from_xml(fh.read())
    except (OSError, IOError):

        try:
            return_code, task_xml = _perform_api_call(
                "task/%d" % task_id)
        except (URLError, UnicodeEncodeError) as e:
            print(e)
            raise e

        with open(xml_file, "w") as fh:
            fh.write(task_xml)

        task = _create_task_from_xml(task_xml)

    # TODO extract this to a function
    task.download_split()
    dataset = datasets.get_dataset(task.dataset_id)

    # TODO look into either adding the class labels to task xml, or other
    # way of reading it.
    class_labels = dataset._retrieve_class_labels(task.target_feature)
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

    # Convert some more parameters
    for parameter in \
            inputs["estimation_procedure"]["oml:estimation_procedure"][
                "oml:parameter"]:
        name = parameter["@name"]
        text = parameter.get("#text", "")
        estimation_parameters[name] = text

    return OpenMLTask(
        dic["oml:task_id"], dic["oml:task_type"],
        inputs["source_data"]["oml:data_set"]["oml:data_set_id"],
        inputs["source_data"]["oml:data_set"]["oml:target_feature"],
        inputs["estimation_procedure"]["oml:estimation_procedure"][
            "oml:type"],
        inputs["estimation_procedure"]["oml:estimation_procedure"][
            "oml:data_splits_url"], estimation_parameters,
        inputs["evaluation_measures"]["oml:evaluation_measures"][
            "oml:evaluation_measure"], None)
