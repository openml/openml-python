import xmltodict

from openml.study import OpenMLStudy
from .._api_calls import _perform_api_call

def _multitag_to_list(result_dict, tag):
    if isinstance(result_dict[tag], list):
        return result_dict[tag]
    elif isinstance(result_dict[tag], dict):
        return [result_dict[tag]]
    else:
        raise TypeError()


def get_study(study_id):
    '''
    Retrieves all relevant information of an OpenML study from the server
    Note that some of the (data, tasks, flows, setups) fields can be empty
    (depending on information on the server)
    '''
    xml_string = _perform_api_call("study/%d" %(study_id))
    result_dict = xmltodict.parse(xml_string)['oml:study']
    id = int(result_dict['oml:id'])
    name = result_dict['oml:name']
    description = result_dict['oml:description']
    creation_date = result_dict['oml:creation_date']
    creator = result_dict['oml:creator']
    tags = []
    for tag in _multitag_to_list(result_dict, 'oml:tag'):
        tags.append({'name': tag['oml:name'],
                     'window_start': tag['oml:window_start'],
                     'write_access': tag['oml:write_access']})

    datasets = None
    tasks = None
    flows = None
    setups = None

    if 'oml:data' in result_dict:
        datasets = [int(x) for x in result_dict['oml:data']['oml:data_id']]

    if 'oml:tasks' in result_dict:
        tasks = [int(x) for x in result_dict['oml:tasks']['oml:task_id']]

    if 'oml:flows' in result_dict:
        flows = [int(x) for x in result_dict['oml:flows']['oml:flow_id']]

    if 'oml:setups' in result_dict:
        setups = [int(x) for x in result_dict['oml:setups']['oml:setup_id']]

    study = OpenMLStudy(id, name, description, creation_date, creator, tags,
                        datasets, tasks, flows, setups)
    return study