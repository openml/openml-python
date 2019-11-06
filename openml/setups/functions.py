# License: BSD 3-Clause

from collections import OrderedDict
import io
import os
from typing import Any, Union, List, Dict, Optional

import xmltodict
import pandas as pd

import openml
from .. import config
from .setup import OpenMLSetup, OpenMLParameter
from openml.flows import flow_exists
import openml.exceptions
import openml.utils


def setup_exists(flow) -> int:
    """
    Checks whether a hyperparameter configuration already exists on the server.

    Parameters
    ----------
    flow : flow
        The openml flow object. Should have flow id present for the main flow
        and all subflows (i.e., it should be downloaded from the server by
        means of flow.get, and not instantiated locally)

    Returns
    -------
    setup_id : int
        setup id iff exists, False otherwise
    """
    # sadly, this api call relies on a run object
    openml.flows.functions._check_flow_for_server_id(flow)
    if flow.model is None:
        raise ValueError('Flow should have model field set with the actual model.')
    if flow.extension is None:
        raise ValueError('Flow should have model field set with the correct extension.')

    # checks whether the flow exists on the server and flow ids align
    exists = flow_exists(flow.name, flow.external_version)
    if exists != flow.flow_id:
        raise ValueError('This should not happen!')

    openml_param_settings = flow.extension.obtain_parameter_values(flow)
    description = xmltodict.unparse(_to_dict(flow.flow_id,
                                             openml_param_settings),
                                    pretty=True)
    file_elements = {'description': ('description.arff', description)}
    result = openml._api_calls._perform_api_call('/setup/exists/',
                                                 'post',
                                                 file_elements=file_elements)
    result_dict = xmltodict.parse(result)
    setup_id = int(result_dict['oml:setup_exists']['oml:id'])
    if setup_id > 0:
        return setup_id
    else:
        return False


def _get_cached_setup(setup_id):
    """Load a run from the cache."""
    cache_dir = config.get_cache_directory()
    setup_cache_dir = os.path.join(cache_dir, "setups", str(setup_id))
    try:
        setup_file = os.path.join(setup_cache_dir, "description.xml")
        with io.open(setup_file, encoding='utf8') as fh:
            setup_xml = xmltodict.parse(fh.read())
            setup = _create_setup_from_xml(setup_xml, output_format='object')
        return setup

    except (OSError, IOError):
        raise openml.exceptions.OpenMLCacheException(
            "Setup file for setup id %d not cached" % setup_id)


def get_setup(setup_id):
    """
     Downloads the setup (configuration) description from OpenML
     and returns a structured object

    Parameters
    ----------
    setup_id : int
        The Openml setup_id

    Returns
    -------
    dict or OpenMLSetup(an initialized openml setup object)
    """
    setup_dir = os.path.join(config.get_cache_directory(),
                             "setups",
                             str(setup_id))
    setup_file = os.path.join(setup_dir, "description.xml")

    if not os.path.exists(setup_dir):
        os.makedirs(setup_dir)

    try:
        return _get_cached_setup(setup_id)
    except (openml.exceptions.OpenMLCacheException):
        url_suffix = '/setup/%d' % setup_id
        setup_xml = openml._api_calls._perform_api_call(url_suffix, 'get')
        with io.open(setup_file, "w", encoding='utf8') as fh:
            fh.write(setup_xml)

    result_dict = xmltodict.parse(setup_xml)
    return _create_setup_from_xml(result_dict, output_format='object')


def list_setups(
    offset: Optional[int] = None,
    size: Optional[int] = None,
    flow: Optional[int] = None,
    tag: Optional[str] = None,
    setup: Optional[List] = None,
    output_format: str = 'object'
) -> Union[Dict, pd.DataFrame]:
    """
    List all setups matching all of the given filters.

    Parameters
    ----------
    offset : int, optional
    size : int, optional
    flow : int, optional
    tag : str, optional
    setup : list(int), optional
    output_format: str, optional (default='object')
        The parameter decides the format of the output.
        - If 'object' the output is a dict of OpenMLSetup objects
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    Returns
    -------
    dict or dataframe
    """
    if output_format not in ['dataframe', 'dict', 'object']:
        raise ValueError("Invalid output format selected. "
                         "Only 'dict', 'object', or 'dataframe' applicable.")

    batch_size = 1000  # batch size for setups is lower
    return openml.utils._list_all(output_format=output_format,
                                  listing_call=_list_setups,
                                  offset=offset,
                                  size=size,
                                  flow=flow,
                                  tag=tag,
                                  setup=setup,
                                  batch_size=batch_size)


def _list_setups(setup=None, output_format='object', **kwargs):
    """
    Perform API call `/setup/list/{filters}`

    Parameters
    ----------
    The setup argument that is a list is separated from the single value
    filters which are put into the kwargs.

    setup : list(int), optional

    output_format: str, optional (default='dict')
        The parameter decides the format of the output.
        - If 'dict' the output is a dict of dict
        - If 'dataframe' the output is a pandas DataFrame

    kwargs: dict, optional
        Legal filter operators: flow, setup, limit, offset, tag.

    Returns
    -------
    dict or dataframe
        """

    api_call = "setup/list"
    if setup is not None:
        api_call += "/setup/%s" % ','.join([str(int(i)) for i in setup])
    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)

    return __list_setups(api_call=api_call, output_format=output_format)


def __list_setups(api_call, output_format='object'):
    """Helper function to parse API calls which are lists of setups"""
    xml_string = openml._api_calls._perform_api_call(api_call, 'get')
    setups_dict = xmltodict.parse(xml_string, force_list=('oml:setup',))
    openml_uri = 'http://openml.org/openml'
    # Minimalistic check if the XML is useful
    if 'oml:setups' not in setups_dict:
        raise ValueError('Error in return XML, does not contain "oml:setups":'
                         ' %s' % str(setups_dict))
    elif '@xmlns:oml' not in setups_dict['oml:setups']:
        raise ValueError('Error in return XML, does not contain '
                         '"oml:setups"/@xmlns:oml: %s'
                         % str(setups_dict))
    elif setups_dict['oml:setups']['@xmlns:oml'] != openml_uri:
        raise ValueError('Error in return XML, value of  '
                         '"oml:seyups"/@xmlns:oml is not '
                         '"%s": %s'
                         % (openml_uri, str(setups_dict)))

    assert type(setups_dict['oml:setups']['oml:setup']) == list, \
        type(setups_dict['oml:setups'])

    setups = dict()
    for setup_ in setups_dict['oml:setups']['oml:setup']:
        # making it a dict to give it the right format
        current = _create_setup_from_xml({'oml:setup_parameters': setup_},
                                         output_format=output_format)
        if output_format == 'object':
            setups[current.setup_id] = current
        else:
            setups[current['setup_id']] = current

    if output_format == 'dataframe':
        setups = pd.DataFrame.from_dict(setups, orient='index')

    return setups


def initialize_model(setup_id: int) -> Any:
    """
    Initialized a model based on a setup_id (i.e., using the exact
    same parameter settings)

    Parameters
    ----------
    setup_id : int
        The Openml setup_id

    Returns
    -------
    model
    """
    setup = get_setup(setup_id)
    flow = openml.flows.get_flow(setup.flow_id)

    # instead of using scikit-learns or any other library's "set_params" function, we override the
    # OpenMLFlow objects default parameter value so we can utilize the
    # Extension.flow_to_model() function to reinitialize the flow with the set defaults.
    for hyperparameter in setup.parameters.values():
        structure = flow.get_structure('flow_id')
        if len(structure[hyperparameter.flow_id]) > 0:
            subflow = flow.get_subflow(structure[hyperparameter.flow_id])
        else:
            subflow = flow
        subflow.parameters[hyperparameter.parameter_name] = \
            hyperparameter.value

    model = flow.extension.flow_to_model(flow)
    return model


def _to_dict(flow_id, openml_parameter_settings):
    # for convenience, this function (ab)uses the run object.
    xml = OrderedDict()
    xml['oml:run'] = OrderedDict()
    xml['oml:run']['@xmlns:oml'] = 'http://openml.org/openml'
    xml['oml:run']['oml:flow_id'] = flow_id
    xml['oml:run']['oml:parameter_setting'] = openml_parameter_settings

    return xml


def _create_setup_from_xml(result_dict, output_format='object'):
    """
    Turns an API xml result into a OpenMLSetup object (or dict)
    """
    setup_id = int(result_dict['oml:setup_parameters']['oml:setup_id'])
    flow_id = int(result_dict['oml:setup_parameters']['oml:flow_id'])
    parameters = {}
    if 'oml:parameter' not in result_dict['oml:setup_parameters']:
        parameters = None
    else:
        # basically all others
        xml_parameters = result_dict['oml:setup_parameters']['oml:parameter']
        if isinstance(xml_parameters, dict):
            id = int(xml_parameters['oml:id'])
            parameters[id] = _create_setup_parameter_from_xml(result_dict=xml_parameters,
                                                              output_format=output_format)
        elif isinstance(xml_parameters, list):
            for xml_parameter in xml_parameters:
                id = int(xml_parameter['oml:id'])
                parameters[id] = \
                    _create_setup_parameter_from_xml(result_dict=xml_parameter,
                                                     output_format=output_format)
        else:
            raise ValueError('Expected None, list or dict, received '
                             'something else: %s' % str(type(xml_parameters)))

    if output_format in ['dataframe', 'dict']:
        return_dict = {'setup_id': setup_id, 'flow_id': flow_id}
        return_dict['parameters'] = parameters
        return(return_dict)
    return OpenMLSetup(setup_id, flow_id, parameters)


def _create_setup_parameter_from_xml(result_dict, output_format='object'):
    if output_format == 'object':
        return OpenMLParameter(input_id=int(result_dict['oml:id']),
                               flow_id=int(result_dict['oml:flow_id']),
                               flow_name=result_dict['oml:flow_name'],
                               full_name=result_dict['oml:full_name'],
                               parameter_name=result_dict['oml:parameter_name'],
                               data_type=result_dict['oml:data_type'],
                               default_value=result_dict['oml:default_value'],
                               value=result_dict['oml:value'])
    else:
        return({'input_id': int(result_dict['oml:id']),
                'flow_id': int(result_dict['oml:flow_id']),
                'flow_name': result_dict['oml:flow_name'],
                'full_name': result_dict['oml:full_name'],
                'parameter_name': result_dict['oml:parameter_name'],
                'data_type': result_dict['oml:data_type'],
                'default_value': result_dict['oml:default_value'],
                'value': result_dict['oml:value']})
