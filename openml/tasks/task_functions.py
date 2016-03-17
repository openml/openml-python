import os
import re
from collections import OrderedDict
import xmltodict

from ..util import URLError
from ..exceptions import OpenMLCacheException
from .. import datasets
from .task import OpenMLTask, _create_task_cache_dir


def get_cached_tasks(api_connector):
    tasks = OrderedDict()
    for task_cache_dir in [api_connector.task_cache_dir,
                           api_connector._private_directory_tasks]:

        directory_content = os.listdir(task_cache_dir)
        directory_content.sort()

        # Find all dataset ids for which we have downloaded the dataset
        # description

        for filename in directory_content:
            match = re.match(r"(tid)_([0-9]*)\.xml", filename)
            if match:
                tid = match.group(2)
                tid = int(tid)

                tasks[tid] = api_connector.get_cached_task(tid)

    return tasks


def get_cached_task(api_connector, tid):
    for task_cache_dir in [api_connector.task_cache_dir,
                           api_connector._private_directory_tasks]:
        task_file = os.path.join(task_cache_dir,
                                 "tid_%d.xml" % int(tid))

        try:
            with open(task_file) as fh:
                task = _create_task_from_xml(api_connector, xml=fh.read())
            return task
        except (OSError, IOError):
            continue

    raise OpenMLCacheException("Task file for tid %d not "
                               "cached" % tid)


def get_estimation_procedure_list(api_connector):
    """Return a list of all estimation procedures which are on OpenML.

    Returns
    -------
    procedures : list
        A list of all estimation procedures. Every procedure is represented by a
        dictionary containing the following information: id,
        task type id, name, type, repeats, folds, stratified.
    """

    return_code, xml_string = api_connector._perform_api_call(
        "estimationprocedure/list")
    procs_dict = xmltodict.parse(xml_string)
    # Minimalistic check if the XML is useful
    assert procs_dict['oml:estimationprocedures']['@xmlns:oml'] == \
        'http://openml.org/openml'
    assert type(procs_dict['oml:estimationprocedures']['oml:estimationprocedure']) == list

    procs = []
    for proc_ in procs_dict['oml:estimationprocedures']['oml:estimationprocedure']:
        proc = {'id': int(proc_['oml:id']),
                'task_type_id': int(proc_['oml:ttid']),
                'name': proc_['oml:name'],
                'type': proc_['oml:type']}

        procs.append(proc)

    return procs


def list_tasks(api_connector, task_type_id=1):
    """Return a list of all tasks which are on OpenML.

    Parameters
    ----------
    task_type_id : int
        ID of the task type as detailed
        `here <http://openml.org/api/?f=openml.task.types>`_.

    Returns
    -------
    tasks : list
        A list of all tasks. Every task is represented by a
        dictionary containing the following information: task id,
        dataset id, task_type and status. If qualities are calculated for
        the associated dataset, some of these are also returned.
    """
    try:
        task_type_id = int(task_type_id)
    except:
        raise ValueError("Task Type ID is neither an Integer nor can be "
                         "cast to an Integer.")

    return_code, xml_string = api_connector._perform_api_call(
        "task/list/type/%d" % task_type_id)
    tasks_dict = xmltodict.parse(xml_string)
    # Minimalistic check if the XML is useful
    assert tasks_dict['oml:tasks']['@xmlns:oml'] == \
        'http://openml.org/openml'
    assert type(tasks_dict['oml:tasks']['oml:task']) == list

    tasks = []
    procs = get_estimation_procedure_list(api_connector)
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
                task[input['@name']] = input['#text']

        task[input['@name']] = input['#text']

        # The number of qualities can range from 0 to infinity
        for quality in task_.get('oml:quality', list()):
            quality['#text'] = float(quality['#text'])
            if abs(int(quality['#text']) - quality['#text']) < 0.0000001:
                quality['#text'] = int(quality['#text'])
            task[quality['@name']] = quality['#text']

        tasks.append(task)
    tasks.sort(key=lambda t: t['tid'])

    return tasks


def get_task(api_connector, task_id):
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

    xml_file = os.path.join(_create_task_cache_dir(api_connector, task_id),
                            "task.xml")

    try:
        with open(xml_file) as fh:
            task = _create_task_from_xml(api_connector, fh.read())
    except (OSError, IOError):

        try:
            return_code, task_xml = api_connector._perform_api_call(
                "task/%d" % task_id)
        except (URLError, UnicodeEncodeError) as e:
            print(e)
            raise e

        # Cache the xml task file
        if os.path.exists(xml_file):
            with open(xml_file) as fh:
                local_xml = fh.read()

            if task_xml != local_xml:
                raise ValueError("Task description of task %d cached at %s "
                                 "has changed." % (task_id, xml_file))

        else:
            with open(xml_file, "w") as fh:
                fh.write(task_xml)

        task = _create_task_from_xml(api_connector, task_xml)

    task.download_split()
    dataset = datasets.get_dataset(api_connector, task.dataset_id)

    # TODO look into either adding the class labels to task xml, or other
    # way of reading it.
    class_labels = dataset.retrieve_class_labels()
    task.class_labels = class_labels
    return task


def _create_task_from_xml(api_connector, xml):
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
            "oml:evaluation_measure"], None, api_connector)
