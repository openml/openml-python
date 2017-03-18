import openml
import xmltodict

from collections import OrderedDict

def setup_exists(downloaded_flow, sklearn_model):
    '''
    Checks whether a flow / hyperparameter configuration already exists on the server

    :param downloaded_flow:
        the openml flow object (should be downloaded from server.
        Otherwise also give flow id parameter)
    :param sklearn_model: obvious
    :param flow_id: int
    :return: int setup id iff exists, False otherwise
    '''

    # sadly, this api call relies on a run object
    openml_param_settings = openml.runs.OpenMLRun._parse_parameters(sklearn_model, downloaded_flow)
    description = xmltodict.unparse(_to_dict(downloaded_flow.flow_id, openml_param_settings), pretty=True)
    file_elements = {'description': ('description.arff',description)}

    result = openml._api_calls._perform_api_call('/setup/exists/',
                                                 file_elements = file_elements)
    result_dict = xmltodict.parse(result)
    if 'oml:id' in result_dict['oml:setup_exists']:
        return int(result_dict['oml:setup_exists']['oml:id'])
    else:
        return False


def _to_dict(flow_id, openml_parameter_settings):
    xml = OrderedDict()
    xml['oml:run'] = OrderedDict()
    xml['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    xml['oml:run']['oml:flow_id'] = flow_id
    xml['oml:run']['oml:parameter_setting'] = openml_parameter_settings

    return xml