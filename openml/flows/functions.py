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
    try:
        flow_id = int(flow_id)
    except:
        raise ValueError("Flow ID must be an int, got %s." % str(flow_id))

    return_code, flow_xml = _perform_api_call("flow/%d" % flow_id)

    flow_dict = xmltodict.parse(flow_xml)
    flow = OpenMLFlow._from_dict(flow_dict)

    if 'sklearn' in flow.external_version:
        flow.model = flow_to_sklearn(flow)

    return flow
