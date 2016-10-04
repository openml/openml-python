import xmltodict

from openml._api_calls import _perform_api_call
from . import OpenMLFlow
from ..util import URLError


def get_flow(flow_id, converter=None):
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

    if converter is not None:
        model = converter.deserialize(flow)
        flow.model = model

    return flow
