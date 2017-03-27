import xmltodict

from openml._api_calls import _perform_api_call
from . import OpenMLFlow, flow_to_sklearn


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

    flow_xml = _perform_api_call("flow/%d" % flow_id)

    flow_dict = xmltodict.parse(flow_xml)
    flow = OpenMLFlow._from_dict(flow_dict)

    if 'sklearn' in flow.external_version:
        flow.model = flow_to_sklearn(flow)

    return flow


def list_flows(offset=None, size=None, tag=None):
    """Return a list of all flows which are on OpenML.

    Parameters
    ----------
    offset : int, optional
        the number of flows to skip, starting from the first
    size : int, optional
        the maximum number of flows to return
    tag : str, optional
        the tag to include

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
    api_call = "flow/list"
    if offset is not None:
        api_call += "/offset/%d" % int(offset)

    if size is not None:
        api_call += "/limit/%d" % int(size)

    if tag is not None:
        api_call += "/tag/%s" % tag

    return _list_flows(api_call)


def flow_exists(name, version):
    """Retrieves the flow id of the flow uniquely identified by name+version.

    Parameter
    ---------
    name : string
        Name of the flow
    version : string
        Version information associated with flow.

    Returns
    -------
    flow_exist : int
        flow id iff exists, False otherwise

    Notes
    -----
    see http://www.openml.org/api_docs/#!/flow/get_flow_exists_name_version
    """
    if not (type(name) is str and len(name) > 0):
        raise ValueError('Argument \'name\' should be a non-empty string')
    if not (type(version) is str and len(version) > 0):
        raise ValueError('Argument \'version\' should be a non-empty string')

    xml_response = _perform_api_call("flow/exists",
                                     data={'name': name, 'external_version': version})

    result_dict = xmltodict.parse(xml_response)
    flow_id = int(result_dict['oml:flow_exists']['oml:id'])
    if flow_id > 0:
        return flow_id
    else:
        return False;


def _list_flows(api_call):
    # TODO add proper error handling here!
    xml_string = _perform_api_call(api_call)
    flows_dict = xmltodict.parse(xml_string)

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