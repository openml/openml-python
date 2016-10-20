from collections import OrderedDict

import six
import xmltodict

from .._api_calls import _perform_api_call


class OpenMLFlow(object):
    """OpenML Flow. Stores machine learning models.

    Flows should not be generated manually, but by the function
    :meth:`openml.flows.create_flow_from_model`. Using this helper function
    ensures that all relevant fields are filled in.

    Implements https://github.com/openml/website/blob/master/openml_OS/ \
        views/pages/api_new/v1/xsd/openml.implementation.upload.xsd.

    Parameters
    ----------
    name : str
        Name of the flow. Is used together with the attribute
        `external_version` as a unique identifier of the flow.
    description : str
        Human-readable description of the flow (free text).
    model : object
        ML model which is described by this flow.
    components : OrderedDict
        Mapping from component identifier to an OpenMLFlow object. Components
        are usually subfunctions of an algorithm (e.g. kernels), base learners
        in ensemble algorithms (decision tree in adaboost) or building blocks
        of a machine learning pipeline. Components are modeled as independent
        flows and can be shared between flows (different pipelines can use
        the same components).
    parameters : OrderedDict
        Mapping from parameter name to the parameter default value. The
        parameter default value must be of type `str`, so that the respective
        toolbox plugin can take care of casting the parameter default value to
        the correct type.
    parameters_meta_info : OrderedDict
        Mapping from parameter name to `dict`. Stores additional information
        for each parameter. Required keys are `data_type` and `description`.
    external_version : str
        Version number of the software the flow is implemented in. Is used
        together with the attribute `name` as a uniquer identifier of the flow.
    tags : list
        List of tags. Created on the server by other API calls.
    language : str
        Natural language the flow is described in (not the programming
        language).
    dependencies : str
        A list of dependencies necessary to run the flow. This field should
        contain all libraries the flow depends on. To allow reproducibility
        it should also specify the exact version numbers.
    binary_url : str, optional
        Url from which the binary can be downloaded. Added by the server.
        Ignored when uploaded manually. Will not be used by the python API
        because binaries aren't compatible across machines.
    binary_format : str, optional
        Format in which the binary code was uploaded. Will not be used by the
        python API because binaries aren't compatible across machines.
    binary_md5 : str, optional
        MD5 checksum to check if the binary code was correctly downloaded. Will
        not be used by the python API because binaries aren't compatible across
        machines.
    uploader : str, optional
        OpenML user ID of the uploader. Filled in by the server.
    upload_date : str, optional
        Date the flow was uploaded. Filled in by the server.
    flow_id : int, optional
        Flow ID. Assigned by the server.
    version : str, optional
        OpenML version of the flow. Assigned by the server.
    """

    def __init__(self, name, description, model, components, parameters,
                 parameters_meta_info, external_version, tags, language,
                 dependencies, binary_url=None, binary_format=None,
                 binary_md5=None, uploader=None, upload_date=None,
                 flow_id=None, version=None):
        self.name = name
        self.description = description
        self.model = model

        for variable, variable_name in [
                [components, 'components'],
                [parameters, 'parameters'],
                [parameters_meta_info, 'parameters_meta_info']]:
            if not isinstance(variable, OrderedDict):
                raise TypeError('%s must be of type OrderedDict, '
                                'but is %s.' % (variable_name, type(variable)))

        self.components = components
        self.parameters = parameters
        self.parameters_meta_info = parameters_meta_info

        keys_parameters = set(parameters.keys())
        keys_parameters_meta_info = set(parameters_meta_info.keys())
        if len(keys_parameters.difference(keys_parameters_meta_info)) > 0:
            raise ValueError('Parameter %s only in parameters, but not in'
                             'parameters_meta_info.' %
                             str(keys_parameters.difference(
                                 keys_parameters_meta_info)))
        if len(keys_parameters_meta_info.difference(keys_parameters)) > 0:
            raise ValueError('Parameter %s only in parameters_meta_info, '
                             'but not in parameters.' %
                             str(keys_parameters_meta_info.difference(
                                 keys_parameters)))

        self.external_version = external_version
        self.uploader = uploader

        self.tags = tags if tags is not None else []
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
        str
            Flow represented as XML string.
        """
        flow_dict = self._to_dict()
        flow_xml = xmltodict.unparse(flow_dict, pretty=True)

        # A flow may not be uploaded with the xml encoding specification:
        # <?xml version="1.0" encoding="utf-8"?>
        flow_xml = flow_xml.split('\n', 1)[-1]
        return flow_xml

    def _to_dict(self):
        """ Helper function used by _to_xml and _to_dict.

        Creates a dictionary representation of self which can be serialized
        to xml by the function _to_xml. Since a flow can contain subflows
        (components) this helper function calls itself recursively to also
        serialize these flows to dictionaries.

        Uses OrderedDict everywhere to make sure that the order of data stays
        at it is added here. The return value (OrderedDict) will be used to
        create the upload xml file. The xml file must have the tags in exactly
        the order given in the xsd schema of a flow (see class docstring).

        Returns
        -------
        OrderedDict
            Flow represented as OrderedDict.

        """
        flow_container = OrderedDict()
        flow_dict = OrderedDict([('@xmlns:oml', 'http://openml.org/openml')])
        flow_container['oml:flow'] = flow_dict
        _add_if_nonempty(flow_dict, 'oml:id', self.flow_id)

        for attribute in ["uploader", "name", "version", "external_version",
                          "description", "upload_date", "language",
                          "dependencies"]:
            _add_if_nonempty(flow_dict, 'oml:{}'.format(attribute),
                             getattr(self, attribute))

        flow_parameters = []
        for key in self.parameters:
            param_dict = OrderedDict()
            param_dict['oml:name'] = key
            meta_info = self.parameters_meta_info[key]

            _add_if_nonempty(param_dict, 'oml:data_type',
                             meta_info['data_type'])
            param_dict['oml:default_value'] = self.parameters[key]
            _add_if_nonempty(param_dict, 'oml:description',
                             meta_info['description'])

            for key_, value in param_dict.items():
                if key_ is not None and not isinstance(key_, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key_, type(key_)))
                if value is not None and not isinstance(value, six.string_types):
                    raise ValueError('Parameter value %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (value, type(value)))

            flow_parameters.append(param_dict)

        flow_dict['oml:parameter'] = flow_parameters

        components = []
        for key in self.components:
            component_dict = OrderedDict()
            component_dict['oml:identifier'] = key
            component_dict['oml:flow'] = \
                self.components[key]._to_dict()['oml:flow']

            for key_ in component_dict:
                # We only need to check if the key is a string, because the
                # value is a flow. The flow itself is valid by recursion
                if key_ is not None and not isinstance(key_, six.string_types):
                    raise ValueError('Parameter name %s cannot be serialized '
                                     'because it is of type %s. Only strings '
                                     'can be serialized.' % (key_, type(key_)))

            components.append(component_dict)

        flow_dict['oml:component'] = components
        flow_dict['oml:tag'] = self.tags
        for attribute in ["binary_url", "binary_format", "binary_md5"]:
            _add_if_nonempty(flow_dict, 'oml:{}'.format(attribute),
                             getattr(self, attribute))

        return flow_container

    @classmethod
    def _from_xml(cls, xml_dict):
        """Create a flow from an xml description.

        Calls itself recursively to create :class:`OpenMLFlow` objects of
        subflows (components).

        Parameters
        ----------
        xml_dict : dict
            Dictionary representation of the flow as created by _to_dict()

        Returns
        -------
            OpenMLFlow

        """
        arguments = {}
        dic = xml_dict["oml:flow"]

        # Mandatory parts in the xml file
        for key in ['name', 'external_version']:
            arguments[key] = dic["oml:" + key]

        # non-mandatory parts in the xml file
        for key in ['uploader', 'description', 'upload_date', 'language',
                    'dependencies', 'version', 'binary_url', 'binary_format',
                    'binary_md5']:
            arguments[key] = dic.get("oml:" + key)

        # has to be converted to an int if present and cannot parsed in the
        # two loops above
        arguments['flow_id'] = (int(dic['oml:id']) if dic.get("oml:id")
                                is not None else None)

        # Now parse parts of a flow which can occur multiple times like
        # parameters, components (subflows) and tags. These can't be tackled
        # in the loops above because xmltodict returns a dict if such an
        # entity occurs once, and a list if it occurs multiple times.
        # Furthermore, they must be treated differently, for example
        # for components this method is called recursively and
        # for parameters the actual information is split into two dictionaries
        # for easier access in python.

        parameters = OrderedDict()
        parameters_meta_info = OrderedDict()
        if 'oml:parameter' in dic:
            # In case of a single parameter, xmltodict returns a dictionary,
            # otherwise a list.
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
        arguments['parameters'] = parameters
        arguments['parameters_meta_info'] = parameters_meta_info

        components = OrderedDict()
        if 'oml:component' in dic:
            # In case of a single component xmltodict returns a dict,
            # otherwise a list.
            if isinstance(dic['oml:component'], dict):
                oml_components = [dic['oml:component']]
            else:
                oml_components = dic['oml:component']

            for component in oml_components:
                flow = OpenMLFlow._from_xml(component)
                components[component['oml:identifier']] = flow
        arguments['components'] = components

        tags = []
        if 'oml:tag' in dic and dic['oml:tag'] is not None:
            # In case of a single tag xmltodict returns a dict, otherwise a list
            if isinstance(dic['oml:tag'], dict):
                oml_tags = [dic['oml:tag']]
            else:
                oml_tags = dic['oml:tag']

            for tag in oml_tags:
                tags.append(tag)
        arguments['tags'] = tags

        arguments['model'] = None
        return cls(**arguments)

    def __eq__(self, other):
        """Check equality.

        Two flows are equal if their all keys which are not set by the server
        are equal, as well as all their parameters and components.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        # Name is actually not generated by the server, but it will be
        # tested further down with a getter (allows mocking in the tests)
        generated_by_the_server = ['name', 'flow_id', 'uploader', 'version',
                                   'upload_date', 'source_url',
                                   'binary_url', 'source_format',
                                   'binary_format', 'source_md5',
                                   'binary_md5', 'model']

        for key in set(self.__dict__.keys()).union(other.__dict__.keys()):
            if key in generated_by_the_server:
                continue
            if getattr(self, key, None) != getattr(other, key, None):
                return False
        return True

    def publish(self):
        """Publish flow to OpenML server.

        Returns
        -------
        self : OpenMLFlow

        """

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
        _, _, flow_id = _check_flow_exists(self.name, flow_version)
        # TODO add numpy and scipy version!

        if int(flow_id) == -1:
            return_code, response_xml = self.publish()

            response_dict = xmltodict.parse(response_xml)
            flow_id = response_dict['oml:upload_flow']['oml:id']
            return int(flow_id)

        return int(flow_id)


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


def _add_if_nonempty(dic, key, value):
    if value is not None:
        dic[key] = value
