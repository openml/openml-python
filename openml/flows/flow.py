from collections import OrderedDict
import re

import six
import xmltodict

from .._api_calls import _perform_api_call
from openml.util import oml_cusual_string


class OpenMLFlow(object):
    """OpenML Flow. Stores machine learning models.

    Parameters
    ----------
    model : scikit-learn compatible model
        The model the flow consists of. The model needs to have fit and predict methods.
    description : string
        Description of the flow (free text).
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
                 parameters=None, parameters_meta_info=None,
                 external_version=None, uploader=None, tags=None,
                 binary_url=None, binary_format=None, binary_md5=None,
                 version=None, upload_date=None, language=None,
                 dependencies=None, flow_id=None):
        self.name = name
        self.description = description
        self.model = model

        if components is None:
            components = OrderedDict()
        elif not isinstance(components, OrderedDict):
            raise TypeError('components must be of type OrderedDict, but is %s.' %
                            type(components))
        self.components = components
        if parameters is None:
            parameters = OrderedDict()
        elif not isinstance(parameters, OrderedDict):
            raise TypeError('parameters must be of type OrderedDict, but is %s.' %
                            type(parameters))
        if parameters_meta_info is None:
            parameters_meta_info = OrderedDict()
        elif not isinstance(parameters_meta_info, OrderedDict):
            raise TypeError('parameters_meta_info must be of type OrderedDict, but is %s.' %
                            type(parameters_meta_info))
        keys_parameters = set(parameters.keys())
        keys_parameters_meta_info = set(parameters_meta_info.keys())
        if len(keys_parameters.difference(keys_parameters_meta_info)) > 0:
            raise ValueError('Parameter %s only in parameters, but not in'
                             'parameters_meta_info.' %
                             str(keys_parameters.difference(keys_parameters_meta_info)))
        if len(keys_parameters_meta_info.difference(keys_parameters)) > 0:
            raise ValueError('Parameter %s only in parameters_meta_info, but not in'
                             'parameters.' %
                             str(keys_parameters_meta_info.difference(keys_parameters)))

        self.parameters = parameters
        self.parameters_meta_info = parameters_meta_info

        self.external_version = external_version
        self.uploader = uploader

        if tags is None:
            tags = []
        self.tags = tags
        self.binary_url = binary_url
        self.binary_format = binary_format
        self.binary_md5 = binary_md5
        self.version = version
        self.upload_date = upload_date
        self.language = language
        self.dependencies = dependencies
        self.flow_id = flow_id

    def _to_xml(self):
        """Generate xml representation of self for upload to server.

        Returns
        -------
        flow_xml : string
            Flow represented as XML string.
        """
        flow_dict = self.__to_dict()
        flow_xml = xmltodict.unparse(flow_dict, pretty=True)

        # A flow may not be uploaded with the encoding specification..
        flow_xml = flow_xml.split('\n', 1)[-1]
        return flow_xml

    def __to_dict(self):
        flow_dict = OrderedDict()
        flow_dict['oml:flow'] = OrderedDict()
        flow_dict['oml:flow']['@xmlns:oml'] = 'http://openml.org/openml'
        if self.flow_id is not None:
            flow_dict['oml:flow']['oml:id'] = self.flow_id
        if self.uploader is not None:
            flow_dict['oml:flow']['oml:uploader'] = self.uploader
        flow_dict['oml:flow']['oml:name'] = self._get_name()
        if self.version is not None:
            flow_dict['oml:flow']['oml:version'] = self.version
        flow_dict['oml:flow']['oml:external_version'] = self.external_version
        flow_dict['oml:flow']['oml:description'] = self.description
        if self.upload_date is not None:
            flow_dict['oml:flow']['oml:upload_date'] = self.upload_date
        if self.language is not None:
            flow_dict['oml:flow']['oml:language'] = self.language
        if self.dependencies is not None:
            flow_dict['oml:flow']['oml:dependencies'] = self.dependencies

        flow_parameters = []
        for key in self.parameters:
            param_dict = OrderedDict()
            param_dict['oml:name'] = key
            if self.parameters_meta_info[key]['data_type'] is not None:
                param_dict['oml:data_type'] = self.parameters_meta_info[key].get('data_type')
            param_dict['oml:default_value'] = self.parameters[key]
            if self.parameters_meta_info[key]['description'] is not None:
                param_dict['oml:description'] = self.parameters_meta_info[key].get('description')

            for key, value in param_dict.items():
                if key is not None and not isinstance(key, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key, type(key)))
                if value is not None and not isinstance(value, six.string_types):
                    raise ValueError('Parameter value %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (value, type(value)))

            flow_parameters.append(param_dict)

        flow_dict['oml:flow']['oml:parameter'] = flow_parameters

        components = []
        for key in self.components:
            component_dict = OrderedDict()
            component_dict['oml:identifier'] = key
            component_dict['oml:flow'] = self.components[key].__to_dict()['oml:flow']

            for key in component_dict:
                # We can only check the key here, because the value is a flow.
                # The flow itself has to be valid by recursion
                if key is not None and not isinstance(key, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key, type(key)))

            components.append(component_dict)

        flow_dict['oml:flow']['oml:component'] = components

        flow_dict['oml:flow']['oml:tag'] = self.tags

        if self.binary_url is not None:
            flow_dict['oml:flow']['oml:binary_url'] = self.binary_url
        if self.binary_format is not None:
            flow_dict['oml:flow']['oml:binary_format'] = self.binary_format
        if self.binary_md5 is not None:
            flow_dict['oml:flow']['oml:binary_md5'] = self.binary_md5

        return flow_dict

    @classmethod
    def _from_xml(cls, xml_dict):
        dic = xml_dict["oml:flow"]
        flow_id = int(dic['oml:id']) if 'oml:id' in dic else None
        uploader = dic.get('oml:uploader')
        name = dic['oml:name']
        external_version = dic.get('oml:external_version')
        description = dic.get('oml:description')
        upload_date = dic.get('oml:upload_date')
        language = dic.get('oml:language')
        dependencies = dic.get('oml:dependencies')
        version = dic.get('oml:version')
        binary_url = dic.get('oml:binary_url')
        binary_format = dic.get('oml:binary_format')
        binary_md5 = dic.get('oml:binary_md5')

        parameters = OrderedDict()
        parameters_meta_info = OrderedDict()
        if 'oml:parameter' in dic:
            if isinstance(dic['oml:parameter'], dict):
                oml_parameters = [dic['oml:parameter']]
            else:
                oml_parameters = dic['oml:parameter']

            for oml_parameter in oml_parameters:
                parameter_name = oml_parameter['oml:name']
                default_value = oml_parameter['oml:default_value']
                parameters[parameter_name] = default_value

                meta_info = OrderedDict()
                meta_info['description'] = oml_parameter.get('oml:description')
                meta_info['data_type'] = oml_parameter.get('oml:data_type')
                parameters_meta_info[parameter_name] = meta_info

        components = OrderedDict()
        if 'oml:component' in dic:
            if isinstance(dic['oml:component'], dict):
                oml_components = [dic['oml:component']]
            else:
                oml_components = dic['oml:component']

            for component in oml_components:
                flow = OpenMLFlow._from_xml(component)
                components[component['oml:identifier']] = flow

        tags = []
        if 'oml:tag' in dic and dic['oml:tag'] is not None:
            if isinstance(dic['oml:tag'], dict):
                oml_tags = [dic['oml:tag']]
            else:
                oml_tags = dic['oml:tag']

            for tag in oml_tags:
                tags.append(tag)

        return cls(name=name, description=description, model=None,
                   components=components, parameters=parameters,
                   parameters_meta_info=parameters_meta_info,
                   external_version=external_version,
                   uploader=uploader, tags=tags, version=version,
                   upload_date=upload_date, language=language,
                   dependencies=dependencies, binary_url=binary_url,
                   binary_format=binary_format, binary_md5=binary_md5,
                   flow_id=flow_id)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            this_dict = self.__dict__.copy()
            this_parameters = this_dict['parameters']
            del this_dict['parameters']
            this_components = this_dict['components']
            del this_dict['components']
            del this_dict['model']

            other_dict = other.__dict__.copy()
            other_parameters = other_dict['parameters']
            del other_dict['parameters']
            other_components = other_dict['components']
            del other_dict['components']
            del other_dict['model']

            # Name is actually not generated by the server, but it will be tested further down with a getter (allows mocking)
            generated_by_the_server = ['name', 'flow_id', 'uploader', 'version',
                                       'upload_date', 'source_url',
                                       'binary_url', 'source_format',
                                       'binary_format', 'source_md5',
                                       'binary_md5']
            for field in generated_by_the_server:
                if field in this_dict:
                    del this_dict[field]
                if field in other_dict:
                    del other_dict[field]
            equal = this_dict == other_dict
            equal_name = self._get_name() == other._get_name()

            parameters_equal = this_parameters.keys() == other_parameters.keys() and \
                               all([this_parameter == other_parameter
                                    for this_parameter, other_parameter in
                                    zip(this_parameters.values(), other_parameters.values())])
            components_equal = this_components.keys() == other_components.keys() and \
                               all([this_component == other_component
                                    for this_component, other_component in
                                    zip(this_components.values(), other_components.values())])

            return parameters_equal and components_equal and equal and equal_name
        return NotImplemented

    def publish(self):
        """Publish flow to OpenML server.

        Returns
        -------
        self : OpenMLFlow

        """
        # Checking that the name adheres to oml:casual_string
        match = re.match(oml_cusual_string, self.name)
        if not match or ((match.span()[1] - match.span()[0]) < len(self.name)):
            raise ValueError('Flow name does not adhere to the '
                             'oml:system_string, the name %s must be matched by '
                             'the following regular expression: %s' %
                             (self.name, oml_cusual_string))

        xml_description = self._to_xml()
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


def _check_flow_exists(name, version):
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
        Flow id or -1 if the flow doesn't exist.

    Notes
    -----
    see http://www.openml.org/api_docs/#!/flow/get_flow_exists_name_version
    """
    if not (type(name) is str and len(name) > 0):
        raise ValueError('Argument \'name\' should be a non-empty string')
    if not (type(version) is str and len(version) > 0):
        raise ValueError('Argument \'version\' should be a non-empty string')

    return_code, xml_response = _perform_api_call(
        "flow/exists/%s/%s" % (name, version))
    # TODO check with latest version of code if this raises an exception
    if return_code != 200:
        # fixme raise appropriate error
        raise ValueError("api call failed: %s" % xml_response)
    xml_dict = xmltodict.parse(xml_response)
    flow_id = xml_dict['oml:flow_exists']['oml:id']
    return return_code, xml_response, flow_id
