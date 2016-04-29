import re
from collections import OrderedDict, deque

import xmltodict
import sklearn

from ..util import URLError
from .._api_calls import _perform_api_call
from ..util import oml_cusual_string



class OpenMLFlow(object):
    """OpenML Flow. Stores machine learning models.

    Parameters
    ----------
    description : string
        Description of the flow (free text).
    name : str
        Name of the flow. This is usually generated automatically. Do not
        change this unless you know what you're doing!
    model : scikit-learn compatible model
        The model the flow consists of. The model needs to have fit and predict methods.
    id : int
        OpenML ID of the flow. Do not pass this, will be created by the server.
    uploader : int
        OpenML user ID of the uploader
    version : int
        OpenML version of the flow. Do not pass this, will be created by the
        server.
    external_version : str
        Version number of the flow/software the flow is implemented in.
    upload_date : str
        Day this flow was uploaded.
    language : str
        Natural language the flow is described in.
    dependencies : str
        A list of dependencies necessary to run the flow.
    source : str
        Programming code of the flow.
    parameters : list
        List of hyperparameters of the flow. Each entry is a dictionary with
        the following keys:
        * name
        * data_type
        * default_value
        * description
    components : list
        List of components of the flow. Each entry is a dictionary with the
        following keys:
        * identifier
        * flow
    """
    def __init__(self, description, name=None, model=None, id=None,
                 uploader=None, version=None, external_version=None,
                 upload_date=None, language=None, dependencies=None,
                 source=None, parameters=None, components=None):
        self.model = model
        self.id = id
        self.upoader = uploader
        self.name = name
        self.version = version
        self.external_version = external_version
        self.description = description
        self.upoad_date = upload_date
        self.language = language
        self.dependencies = dependencies
        self.parameters = parameters
        self.components = components
        self.source = source
        self.name = name
        self.external_version = external_version

    def _generate_flow_xml(self, model=None, description=None,
                           return_dict=False):
        """Generate xml representation of self for upload to server.

        Returns
        -------
        flow_xml : string
            Flow represented as XML string.
        """
        flow_dict = OrderedDict()
        flow_dict['oml:flow'] = OrderedDict()

        # Necessary when this method is called recursive!
        if model is None:
            model = self.model
        if description is None:
            description = self.description

        flow_dict['oml:flow']['oml:description'] = description
        flow_dict['oml:flow']['@xmlns:oml'] = 'http://openml.org/openml'
        if self.external_version is not None:
            flow_dict['oml:flow']['oml:external_version'] = self.external_version

        clf_params = model.get_params()
        flow_parameters = []
        flow_components = OrderedDict()
        component_sort = []
        expected_flow_components = set()

        def is_estimator(v):
            return (hasattr(v, 'fit') and hasattr(v, 'predict') and
                    hasattr(v, 'get_params') and hasattr(v, 'set_params'))

        def is_transformer(v):
            return (hasattr(v, 'fit') and hasattr(v, 'transform') and
                    hasattr(v, 'get_params') and hasattr(v, 'set_params'))

        for k, v in sorted(clf_params.items(), key=lambda t: t[0]):
            # data_type, default_value, description, recommendedRange
            # type = v.__class__.__name__    Not using this because it doesn't
            # conform standards eg. int instead of integer
            # If the default value is another estimator object, we treat it
            # as a component, not as a hyperparameter. Consequently,
            # all hyperparameters with a double underscore in its name will
            # not be registered as a hyperparameter
            if '__' in k:
                k_ = k.split('__')
                for k__ in k_[:-1]:
                    expected_flow_components.add(k__)

            # Properly handle sub-components
            # pipelines and feature unions are estimators themselves
            # they also return estimators in their in get_params which
            # makes this piece of code work, also, they return the estimator in
            # steps which allows us to set the steps parameter further down
            elif is_estimator(v) or is_transformer(v) or \
                    (isinstance(v, tuple) and is_estimator(v[1])) or \
                    (isinstance(v, tuple) and is_transformer(v[1])):
                # TODO check if component already exists, if yes, reuse it!
                # TODO factor all this out in a scikit-learn specifit
                # sub-package
                component_xml = self._generate_flow_xml(
                    model=v, description='Automatically created '
                                          'sub-component.',
                    return_dict=True)
                flow_component = OrderedDict()
                flow_component['oml:identifier'] = k
                flow_component['oml:flow'] = component_xml['oml:flow']
                flow_components[k] = flow_component

            # Try to handle list or tuples from pipelines/feature unions
            elif isinstance(v, list) or isinstance(v, tuple):
                all_subcomponents = [is_transformer(elem[1]) or
                                     is_estimator(elem[1]) for elem in v]
                if not all(all_subcomponents):
                    raise ValueError('%s contains elements which are neither '
                                     'an estimator nor a transformer.')
                new_value = []
                for elem in v:
                    sub_name = elem[1].__module__ + "." + \
                               elem[1].__class__.__name__
                    new_value.append('(%s, %s)' % (elem[0], sub_name))
                    component_sort.append(elem[0])
                new_value = '(' + ', '.join(new_value) + ')'

                param_dict = OrderedDict()
                param_dict['oml:name'] = k
                param_dict['oml:default_value'] = new_value
                flow_parameters.append(param_dict)

            else:
                param_dict = OrderedDict()
                param_dict['oml:name'] = k
                if v:
                    param_dict['oml:default_value'] = str(v)
                flow_parameters.append(param_dict)

        # Check if all expected flow components which are in the
        # hyperparameters (given by __) are actually present as sub-components
        found_flow_components = set()
        to_visit = deque()
        to_visit.extendleft(flow_components.values())
        while len(to_visit) > 0:
            component = to_visit.pop()
            found_flow_components.add(component['oml:identifier'])
            to_visit.extendleft(component['oml:flow']['oml:component'])
        if len(found_flow_components) != len(expected_flow_components):
            raise ValueError('%s != %s' % (found_flow_components,
                                           expected_flow_components))

        # Sort all components - this makes the naming nice and consistent
        # with the step order in pipelines
        for component in component_sort:
            tmp = flow_components[component]
            del flow_components[component]
            flow_components[component] = tmp

        # Components must be a list:
        flow_components_list = []
        for value in flow_components.values():
            flow_components_list.append(value)

        flow_dict['oml:flow']['oml:parameter'] = flow_parameters
        flow_dict['oml:flow']['oml:component'] = flow_components_list

        sub_components_names = ",".join([sub_component['oml:flow']['oml:name'] for
                                         sub_component in
                                         flow_components_list])
        name = model.__module__ + "." + model.__class__.__name__
        if sub_components_names:
            name = '%s(%s)' % (name, sub_components_names)

        # We have to add the name at the beginning of the xml. Only python
        # 2/3 compatible way is to create a new ordered dict
        new_flow_dict = OrderedDict()
        new_flow_dict['oml:name'] = name
        for k in flow_dict['oml:flow']:
            new_flow_dict[k] = flow_dict['oml:flow'][k]
        flow_dict['oml:flow'] = new_flow_dict
        del new_flow_dict

        if return_dict:
            return flow_dict
        else:
            self.name = name

        flow_xml = xmltodict.unparse(flow_dict, pretty=True)
        # A flow may not be uploaded with the encoding specification..
        flow_xml = flow_xml.split('\n', 1)[-1]
        return flow_xml

    def publish(self):
        """
        The 'description' is binary data of an XML file according to the XSD Schema (OUTDATED!):
        https://github.com/openml/website/blob/master/openml_OS/views/pages/rest_api/xsd/openml.implementation.upload.xsd
        """
        xml_description = self._generate_flow_xml()

        # Checking that the name adheres to oml:system_string
        match = re.match(oml_cusual_string, self.name)
        if not match or ((match.span()[1] - match.span()[0]) < len(self.name)):
            raise ValueError('Flow name does not adhere to the '
                             'oml:system_string, the name must be matched by '
                             'the following regular expression: %s' % oml_cusual_string)

        data = {'description': xml_description, 'source': self.source}
        return_code, return_value = _perform_api_call(
            "/flow/", data=data)
        return return_code, return_value

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
        # MF not sure if this is necessary - what would we get from that?

        if int(flow_id) == -1:
            return_code, response_xml = self.publish()

            response_dict = xmltodict.parse(response_xml)
            flow_id = response_dict['oml:upload_flow']['oml:id']
            return int(flow_id)

        return int(flow_id)


def _check_flow_exists(name, version):
    """Retrieves the flow id of the flow uniquely identified by name+external version.

    Parameter
    ---------
    name : string
        Name of the flow
    version : string
        Version information associated with flow.

    Returns
    -------
    return_code : int
        Return code of API call
    xml_responese : str
        String version of XML response of API call
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
        "/flow/exists/%s/%s" % (name, version))
    # TODO check with latest version of code if this raises an exception
    if return_code != 200:
        # fixme raise appropriate error
        raise ValueError("api call failed: %s" % xml_response)
    xml_dict = xmltodict.parse(xml_response)
    flow_id = xml_dict['oml:flow_exists']['oml:id']
    return return_code, xml_response, flow_id


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
        raise ValueError("Flow ID is neither an Integer nor can be "
                         "cast to an Integer.")

    try:
        return_code, flow_xml = _perform_api_call(
            "flow/%d" % flow_id)
    except (URLError, UnicodeEncodeError) as e:
        print(e)
        raise e

    flow_dict = xmltodict.parse(flow_xml)
    flow = _create_flow_from_dict(flow_dict)
    return flow


def _create_flow_from_dict(xml_dict):
    dic = xml_dict["oml:flow"]

    flow_id = int(dic['oml:id'])
    uploader = dic['oml:uploader']
    name = dic['oml:name']
    version = dic['oml:version']
    external_version = dic.get('oml:external_version', None)
    description = dic['oml:description']
    upload_date = dic['oml:upload_date']
    language = dic.get('oml:language', None)
    dependencies = dic.get('oml:dependencies', None)

    parameters = []
    if 'oml:parameter' in dic:
        if isinstance(dic['oml:parameter'], dict):
            oml_parameters = [dic['oml:parameter']]
        else:
            oml_parameters = dic['oml:parameter']

        for oml_parameter in oml_parameters:
            parameter_name = oml_parameter['oml:name']
            data_type = oml_parameter['oml:data_type']
            default_value = oml_parameter['oml:default_value']
            parameter_description = oml_parameter['oml:description']
            parameters.append({'name': parameter_name,
                               'data_type': data_type,
                               'default_value': default_value,
                               'description': parameter_description})

    components = []
    if 'oml:component' in dic:
        if isinstance(dic['oml:component'], dict):
            oml_components = [dic['oml:component']]
        else:
            oml_components = dic['oml:component']

        for component in oml_components:
            identifier = component['oml:identifier']
            flow = _create_flow_from_dict({'oml:flow': component['oml:flow']})
            components.append({'identifier': identifier, 'flow': flow})

    return OpenMLFlow(id=flow_id, uploader=uploader, name=name,
                      version=version, external_version=external_version,
                      description=description, upload_date=upload_date,
                      language=language, dependencies=dependencies,
                      parameters=parameters, components=components)

