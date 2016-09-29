import xmltodict

from openml._api_calls import _perform_api_call
from openml.exceptions import PyOpenMLError
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
        raise ValueError("Flow ID is neither an Integer nor can be "
                         "cast to an Integer.")

    try:
        return_code, flow_xml = _perform_api_call(
            "flow/%d" % flow_id)
    except (URLError, UnicodeEncodeError) as e:
        print(e)
        raise e

    flow_dict = xmltodict.parse(flow_xml)
    flow = OpenMLFlow._from_xml(flow_dict)

    if converter is not None:
        model = converter.deserialize_object(flow)
        flow.model = model

    return flow


def get_flow_dict(flow):
    """Returns a dictionary with keys flow name and values flow id.

        Parameters
        ----------
        flow : OpenMLFlow
        """
    if flow.flow_id is None:
        raise PyOpenMLError("Can only invoke function 'get_flow_map' on a server downloaded flow. ")
    flow_map = {flow.name : flow.flow_id}
    for subflow in flow.components:
        flow_map.update(get_flow_dict(flow.components[subflow]))
    return flow_map