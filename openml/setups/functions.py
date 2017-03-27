import openml
import xmltodict

from collections import OrderedDict

def setup_exists(downloaded_flow, sklearn_model):
    '''
    Checks whether a flow / hyperparameter configuration already exists on the server

    Parameter
    ---------

    downloaded_flow : flow
        the openml flow object (should be downloaded from server.
        Otherwise also give flow id parameter)
    sklearn_model : BaseEstimator
        The base estimator that was used to create the flow. Will
         be used to extract parameter settings from.

    Returns
    -------
    setup_id : int s
        setup id iff exists, False otherwise
    '''

    # sadly, this api call relies on a run object
    openml_param_settings = openml.runs.OpenMLRun._parse_parameters(sklearn_model, downloaded_flow)
    description = xmltodict.unparse(_to_dict(downloaded_flow.flow_id, openml_param_settings), pretty=True)
    file_elements = {'description': ('description.arff',description)}

    result = openml._api_calls._perform_api_call('/setup/exists/',
                                                 file_elements = file_elements)
    result_dict = xmltodict.parse(result)
    setup_id = int(result_dict['oml:setup_exists']['oml:id'])
    if setup_id > 0:
        return setup_id
    else:
        return False;


def _to_dict(flow_id, openml_parameter_settings):
    xml = OrderedDict()
    xml['oml:run'] = OrderedDict()
    xml['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    xml['oml:run']['oml:flow_id'] = flow_id
    xml['oml:run']['oml:parameter_setting'] = openml_parameter_settings

    return xml