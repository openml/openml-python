import xmltodict

from openml.study import OpenMLStudy
import openml._api_calls


def _multitag_to_list(result_dict, tag):
    if isinstance(result_dict[tag], list):
        return result_dict[tag]
    elif isinstance(result_dict[tag], dict):
        return [result_dict[tag]]
    else:
        raise TypeError()


def get_study(study_id, type=None):
    '''
    Retrieves all relevant information of an OpenML study from the server
    Note that some of the (data, tasks, flows, setups) fields can be empty
    (depending on information on the server)
    '''
    call_suffix = "study/%s" %str(study_id)
    if type is not None:
        call_suffix += "/" + type
    xml_string = openml._api_calls._perform_api_call(call_suffix)
    result_dict = xmltodict.parse(xml_string)['oml:study']
    id = int(result_dict['oml:id'])
    name = result_dict['oml:name']
    description = result_dict['oml:description']
    creation_date = result_dict['oml:creation_date']
    creator = result_dict['oml:creator']
    tags = []
    for tag in _multitag_to_list(result_dict, 'oml:tag'):
        current_tag = {'name': tag['oml:name'],
                       'write_access': tag['oml:write_access']}
        if 'oml:window_start' in tag:
            current_tag['window_start'] = tag['oml:window_start']
        tags.append(current_tag)

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
