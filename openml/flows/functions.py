import xmltodict

from openml._api_calls import _perform_api_call
# Absolute imports, to avoid circular dependencies
from openml.flows.sklearn_converter import flow_to_sklearn
from . import OpenMLFlow


def get_flow(flow_id):
    """Download the OpenML flow for a given flow ID.

    Parameters
    ----------
    flow_id : int
        The OpenML flow id.
    """
    try:
        flow_id = int(flow_id)
    except:
        raise ValueError("Flow ID must be an int, got %s." % str(flow_id))

    return_code, flow_xml = _perform_api_call("flow/%d" % flow_id)

    flow_dict = xmltodict.parse(flow_xml)
    flow = OpenMLFlow._from_xml(flow_dict)

    if 'sklearn' in flow.external_version:
        flow.model = flow_to_sklearn(flow)

    return flow
