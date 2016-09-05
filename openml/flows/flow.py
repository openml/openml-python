from collections import OrderedDict
import xmltodict

from .._api_calls import _perform_api_call
from .functions import _check_flow_exists


class OpenMLFlow(object):
    """OpenML Flow. Stores machine learning models.

    Parameters
    ----------
    model : scikit-learn compatible model
        The model the flow consists of. The model needs to have fit and predict methods.
    description : string
        Description of the flow (free text).
    creator : string
        FIXME
    contributor : string
        FIXME
    tag : string
        FIXME
    flow_id : int, optional
        Flow ID. Assigned by the server (fixme shouldn't be here?)
    uploader : string, optional
        User uploading the model (fixme shouldn't be here?). Assigned by the server.


    """
    def __init__(self, name, description=None, model=None, components=None,
                 parameters=None, external_version=None, creator=None,
                 uploader=None, tag=None, flow_id=None):
        self.name = name
        self.description = description
        self.model = model

        if components is None:
            components = OrderedDict()
        elif not isinstance(components, OrderedDict):
            raise TypeError('Components must be of type OrderedDict, but are %s.' %
                            type(components))
        self.components = components
        if parameters is None:
            parameters = OrderedDict()
        elif not isinstance(parameters, OrderedDict):
            raise TypeError('Parameters must be of type OrderedDict, but are %s.' %
                            type(parameters))
        self.parameters = parameters

        self.external_version = external_version
        self.creator = creator
        self.upoader = uploader
        self.tag = tag
        self.flow_id = flow_id

    def _generate_flow_xml(self):
        """Generate xml representation of self for upload to server.

        Returns
        -------
        flow_xml : string
            Flow represented as XML string.
        """
        model = self.model

        flow_dict = OrderedDict()
        flow_dict['oml:flow'] = OrderedDict()
        flow_dict['oml:flow']['@xmlns:oml'] = 'http://openml.org/openml'
        flow_dict['oml:flow']['oml:name'] = self._get_name()
        flow_dict['oml:flow']['oml:external_version'] = self.external_version
        flow_dict['oml:flow']['oml:description'] = self.description

        clf_params = model.get_params()
        flow_parameters = []
        for k, v in clf_params.items():
            # data_type, default_value, description, recommendedRange
            # type = v.__class__.__name__    Not using this because it doesn't conform standards
            # eg. int instead of integer
            param_dict = {'oml:name': k}
            flow_parameters.append(param_dict)

        flow_dict['oml:flow']['oml:parameter'] = flow_parameters

        flow_xml = xmltodict.unparse(flow_dict, pretty=True)

        # A flow may not be uploaded with the encoding specification..
        flow_xml = flow_xml.split('\n', 1)[-1]
        return flow_xml

    def publish(self):
        """Publish flow to OpenML server.

        Returns
        -------
        self : OpenMLFlow

        """
        xml_description = self._generate_flow_xml()

        file_elements = {'description': xml_description}
        return_code, return_value = _perform_api_call(
            "flow/", file_elements=file_elements)
        self.flow_id = int(xmltodict.parse(return_value)['oml:upload_flow']['oml:id'])
        return self

    def _ensure_flow_exists(self):
        """ Checks if a flow exists for the given model and possibly creates it.

        If the given flow exists on the server, the flow-id will simply
        be returned. Otherwise it will be uploaded to the server.

        Returns
        -------
        flow_id : int
            Flow id on the server.
        """
        import sklearn
        flow_version = 'sklearn_' + sklearn.__version__
        _, _, flow_id = _check_flow_exists(self._get_name(), flow_version)
        # TODO add numpy and scipy version!

        if int(flow_id) == -1:
            return_code, response_xml = self.publish()

            response_dict = xmltodict.parse(response_xml)
            flow_id = response_dict['oml:upload_flow']['oml:id']
            return int(flow_id)

        return int(flow_id)

    def _get_name(self):
        """Helper function. Can be mocked for testing."""
        return self.name


def create_flow_from_model(model, converter, description=None):
    flow = converter.serialize_object(model)
    if not isinstance(flow, OpenMLFlow):
        raise ValueError('Converter %s did return %s, not OpenMLFlow!' %
                         (str(converter), type(flow)))
    if description is not None:
        flow.description = description

    return flow
