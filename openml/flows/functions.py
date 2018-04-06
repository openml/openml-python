import dateutil.parser

import xmltodict
import six

import openml._api_calls
from . import OpenMLFlow
import openml.utils


def get_flow(flow_id):
    """Download the OpenML flow for a given flow ID.

    Parameters
    ----------
    flow_id : int
        The OpenML flow id.
    """
    # TODO add caching here!
    try:
        flow_id = int(flow_id)
    except:
        raise ValueError("Flow ID must be an int, got %s." % str(flow_id))

    flow_xml = openml._api_calls._perform_api_call("flow/%d" % flow_id)

    flow_dict = xmltodict.parse(flow_xml)
    flow = OpenMLFlow._from_dict(flow_dict)

    return flow


def list_flows(offset=None, size=None, tag=None, **kwargs):

    """
    Return a list of all flows which are on OpenML.
    (Supports large amount of results)

    Parameters
    ----------
    offset : int, optional
        the number of flows to skip, starting from the first
    size : int, optional
        the maximum number of flows to return
    tag : str, optional
        the tag to include
    kwargs: dict, optional
        Legal filter operators: uploader.

    Returns
    -------
    flows : dict
        A mapping from flow_id to a dict giving a brief overview of the
        respective flow.

        Every flow is represented by a dictionary containing
        the following information:
        - flow id
        - full name
        - name
        - version
        - external version
        - uploader
    """
    return openml.utils.list_all(_list_flows, offset=offset, size=size, tag=tag, **kwargs)


def _list_flows(**kwargs):
    """
    Perform the api call that return a list of all flows.

    Parameters
    ----------
    kwargs: dict, optional
        Legal filter operators: uploader, tag, limit, offset.

    Returns
    -------
    flows : dict
    """
    api_call = "flow/list"

    if kwargs is not None:
        for operator, value in kwargs.items():
            api_call += "/%s/%s" % (operator, value)

    return __list_flows(api_call)


def flow_exists(name, external_version):
    """Retrieves the flow id.

    A flow is uniquely identified by name + external_version.

    Parameters
    ----------
    name : string
        Name of the flow
    external_version : string
        Version information associated with flow.

    Returns
    -------
    flow_exist : int
        flow id iff exists, False otherwise

    Notes
    -----
    see http://www.openml.org/api_docs/#!/flow/get_flow_exists_name_version
    """
    if not (isinstance(name, six.string_types) and len(name) > 0):
        raise ValueError('Argument \'name\' should be a non-empty string')
    if not (isinstance(name, six.string_types) and len(external_version) > 0):
        raise ValueError('Argument \'version\' should be a non-empty string')

    xml_response = openml._api_calls._perform_api_call(
        "flow/exists",
        data={'name': name, 'external_version': external_version},
    )

    result_dict = xmltodict.parse(xml_response)
    flow_id = int(result_dict['oml:flow_exists']['oml:id'])
    if flow_id > 0:
        return flow_id
    else:
        return False


def __list_flows(api_call):

    xml_string = openml._api_calls._perform_api_call(api_call)
    flows_dict = xmltodict.parse(xml_string, force_list=('oml:flow',))

    # Minimalistic check if the XML is useful
    assert type(flows_dict['oml:flows']['oml:flow']) == list, \
        type(flows_dict['oml:flows'])
    assert flows_dict['oml:flows']['@xmlns:oml'] == \
        'http://openml.org/openml', flows_dict['oml:flows']['@xmlns:oml']

    flows = dict()
    for flow_ in flows_dict['oml:flows']['oml:flow']:
        fid = int(flow_['oml:id'])
        flow = {'id': fid,
                'full_name': flow_['oml:full_name'],
                'name': flow_['oml:name'],
                'version': flow_['oml:version'],
                'external_version': flow_['oml:external_version'],
                'uploader': flow_['oml:uploader']}
        flows[fid] = flow

    return flows


def _check_flow_for_server_id(flow):
    """Check if the given flow and it's components have a flow_id."""

    # Depth-first search to check if all components were uploaded to the
    # server before parsing the parameters
    stack = list()
    stack.append(flow)
    while len(stack) > 0:
        current = stack.pop()
        if current.flow_id is None:
            raise ValueError("Flow %s has no flow_id!" % current.name)
        else:
            for component in current.components.values():
                stack.append(component)


def assert_flows_equal(flow1, flow2,
                       ignore_parameter_values_on_older_children=None,
                       ignore_parameter_values=False):
    """Check equality of two flows.

    Two flows are equal if their all keys which are not set by the server
    are equal, as well as all their parameters and components.

    Parameters
    ----------
    flow1 : OpenMLFlow

    flow2 : OpenMLFlow

    ignore_parameter_values_on_older_children : str
        If set to ``OpenMLFlow.upload_date``, ignores parameters in a child
        flow if it's upload date predates the upload date of the parent flow.

    ignore_parameter_values : bool
        Whether to ignore parameter values when comparing flows.
    """
    if not isinstance(flow1, OpenMLFlow):
        raise TypeError('Argument 1 must be of type OpenMLFlow, but is %s' %
                        type(flow1))

    if not isinstance(flow2, OpenMLFlow):
        raise TypeError('Argument 2 must be of type OpenMLFlow, but is %s' %
                        type(flow2))

    # TODO as they are actually now saved during publish, it might be good to
    # check for the equality of these as well.
    generated_by_the_server = ['flow_id', 'uploader', 'version', 'upload_date',
                               # Tags aren't directly created by the server,
                               # but the uploader has no control over them!
                               'tags']
    ignored_by_python_api = ['binary_url', 'binary_format', 'binary_md5',
                             'model']

    for key in set(flow1.__dict__.keys()).union(flow2.__dict__.keys()):
        if key in generated_by_the_server + ignored_by_python_api:
            continue
        attr1 = getattr(flow1, key, None)
        attr2 = getattr(flow2, key, None)
        if key == 'components':
            for name in set(attr1.keys()).union(attr2.keys()):
                if name not in attr1:
                    raise ValueError('Component %s only available in '
                                     'argument2, but not in argument1.' % name)
                if name not in attr2:
                    raise ValueError('Component %s only available in '
                                     'argument2, but not in argument1.' % name)
                assert_flows_equal(attr1[name], attr2[name],
                                   ignore_parameter_values_on_older_children,
                                   ignore_parameter_values)

        else:
            if key == 'parameters':
                if ignore_parameter_values or \
                        ignore_parameter_values_on_older_children:
                    parameters_flow_1 = set(flow1.parameters.keys())
                    parameters_flow_2 = set(flow2.parameters.keys())
                    symmetric_difference = parameters_flow_1 ^ parameters_flow_2
                    if len(symmetric_difference) > 0:
                        raise ValueError('Flow %s: parameter set of flow '
                                         'differs from the parameters stored '
                                         'on the server.' % flow1.name)

                if ignore_parameter_values_on_older_children:
                    upload_date_current_flow = dateutil.parser.parse(
                        flow1.upload_date)
                    upload_date_parent_flow = dateutil.parser.parse(
                        ignore_parameter_values_on_older_children)
                    if upload_date_current_flow < upload_date_parent_flow:
                        continue

                if ignore_parameter_values:
                    # Continue needs to be done here as the first if
                    # statement triggers in both special cases
                    continue

            if attr1 != attr2:
                raise ValueError("Flow %s: values for attribute '%s' differ: "
                                 "'%s'\nvs\n'%s'." %
                                 (str(flow1.name), str(key), str(attr1), str(attr2)))
