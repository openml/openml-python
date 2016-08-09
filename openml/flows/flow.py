import ast
from collections import OrderedDict, deque
import copy
import distutils.version
import re
import warnings
import importlib

# Necessary to have signature available in python 2.7
from sklearn.utils.fixes import signature

import numpy as np
import six
import xmltodict

from ..exceptions import OpenMLRestrictionViolated
from ..util import URLError
from .._api_calls import _perform_api_call
from ..util import oml_cusual_string
from .. import config
from ..sklearn.stats import Distribution, Unparametrized


MAXIMAL_FLOW_LENGTH = 1024


def _is_estimator(v):
    return (hasattr(v, 'fit') and hasattr(v, 'predict') and
            hasattr(v, 'get_params') and hasattr(v, 'set_params'))


def _is_transformer(v):
    return (hasattr(v, 'fit') and hasattr(v, 'transform') and
            hasattr(v, 'get_params') and hasattr(v, 'set_params'))


def _is_crossvalidator(v):
    return (hasattr(v, '_iter_test_masks') and hasattr(v, 'get_n_splits')) or \
           (hasattr(v, '_iter_test_indices') and hasattr(v, 'get_n_splits'))


def _is_distribution(v):
    return ((hasattr(v, 'rvs') and hasattr(v, 'get_params'))) or \
            isinstance(v, Distribution)


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
        A list of dependencies necessary to run the flow. The default values
        will be numpy, scipy and scikit-learn (sklearn).
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
        if dependencies is None:
            import numpy as np
            numpy_version = np.__version__
            import scipy
            scipy_version = scipy.__version__
            import sklearn
            sklearn_version = sklearn.__version__
            dependencies = ['numpy=%s' % numpy_version,
                            'scipy=%s' % scipy_version,
                            'sklearn=%s' % sklearn_version]
            dependencies = ' '.join(dependencies)

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

    def init_parameters_and_components(self, model=None, description=None):
        if description is None and model is None:
            flow = self
        else:
            flow = OpenMLFlow(description, model=model,
                              external_version=self.external_version)

        if model is None:
            model = flow.model

        clf_params = model.get_params()
        flow_parameters = []
        flow_components = OrderedDict()
        flow_distribution_components = OrderedDict()
        flow_distribution_parameters = []
        component_sort = []
        expected_flow_components = set()

        contains_parameter_distribution = False
        parametrized_parameters = {}

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
            elif _is_estimator(v) or _is_transformer(v) or \
                    _is_crossvalidator(v) or _is_distribution(v):
                # TODO check if component already exists, if yes, reuse it!
                # TODO factor all this out in a scikit-learn specific
                # sub-package?
                subflow = self.init_parameters_and_components(
                    model=v, description='Automatically created '
                                         'sub-component.')
                flow_component = OrderedDict()
                flow_component['oml:identifier'] = k
                flow_component['oml:flow'] = subflow
                flow_components[k] = flow_component

                # Only add components as a parameter which are direct
                # arguments to the constructor like
                # AdaBoostClassifier(base_estimator=DecisionTreeClassifier).
                # Since Pipeline and FeatureUnion also return estimators and
                # transformers in the 'steps' list with get_params(), we must
                # add them as a component, but not as a parameter of the
                # flow. The next if makes sure that we only add parameters
                # for the first case.
                model_parameters = signature(model.__init__)
                if k in model_parameters.parameters:
                    param_dict = OrderedDict()
                    param_dict['oml:name'] = k
                    param_dict['oml:default_value'] = v.__module__ + "." + \
                                                      v.__class__.__name__
                    flow_parameters.append(param_dict)
            elif isinstance(v, dict) and k == 'param_distributions':
                # Treat param_distributions as a special case: add each as a
                # tunable hyperparameter as a parameter to the flow.
                contains_parameter_distribution = True

                for k_ in v:
                    if not _is_distribution(v[k_]):
                        raise ValueError('Can only work with subclasses of '
                                         'openml.sklearn.stats.distribution, '
                                         'got %s!' % str(v[k_]))
                    parametrized_parameters[k_] = v[k_]

            else:
                param_dict = OrderedDict()
                param_dict['oml:name'] = k
                if not (hasattr(v, '__len__') and len(v) == 0):
                    new_value, tmp = _param_value_to_string(v)
                    param_dict['oml:default_value'] = new_value
                    component_sort.extend(tmp)
                flow_parameters.append(param_dict)

        # Add a component for each tunable hyperparameter. The ones specified
        # in the dict param_distributions are serialized to a string,
        # all others unparametrized in the default setting. They can be
        # overridden by uploading a run.
        if contains_parameter_distribution:
            for k, v in sorted(clf_params.items(), key=lambda t: t[0]):

                param_dict = OrderedDict()
                param_dict['oml:name'] = 'parameter_distribution__%s' % k

                if k.startswith('estimator__') and \
                        k.replace('estimator__', '', 1) in parametrized_parameters:
                    distribution = parametrized_parameters[
                        k.replace('estimator__', '', 1)]
                    distribution_name = distribution.__class__.__name__
                    distribution_parameters = distribution.get_params()
                    default_value = '%s(%s)' % (
                        distribution_name,
                        ','.join(['%s=%s' % items for items in sorted(distribution_parameters.items())]))
                    default_value = 'LogUniformInt(base=2,expon_lower=0,' \
                                    'expo_upper=3)'
                else:
                    default_value = 'Unparametrized'

                param_dict['oml:default_value'] = default_value
                flow_distribution_parameters.append(param_dict)

        # Check if all expected flow components which are in the
        # hyperparameters (given by __) are actually present as sub-components
        found_flow_components = set()
        to_visit = deque()
        to_visit.extendleft(flow_components.values())
        while len(to_visit) > 0:
            component = to_visit.pop()
            found_flow_components.add(component['oml:identifier'])
            to_visit.extendleft(component['oml:flow'].components)
        if len(found_flow_components) != len(expected_flow_components):
            raise ValueError('%s != %s' %
                             (found_flow_components,
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

        sub_components_names = "__".join(
            [sub_component['oml:flow'].name
             for sub_component in flow_components_list])
        name = model.__module__ + "." + model.__class__.__name__
        if sub_components_names:
            name = '%s(%s)' % (name, sub_components_names)

        if len(name) > MAXIMAL_FLOW_LENGTH:
            raise OpenMLRestrictionViolated('Flow name must not be longer '
                                            'than %d characters!' % MAXIMAL_FLOW_LENGTH)

        flow.parameters = flow_parameters
        flow.components = flow_components_list
        # Add the distribution components afterwards to not let them make the
        #  name unreadable
        for flow_distribution in flow_distribution_components:
            flow.components.append(flow_distribution_components[flow_distribution])
        for flow_distribution in flow_distribution_parameters:
            flow.parameters.append(flow_distribution)

        flow.name = name

        return flow

    def _generate_flow_xml(self, return_dict=False):
        """Generate xml representation of self for upload to server.

        Returns
        -------
        flow_xml : string
            Flow represented as XML string.
        """

        # Necessary when this method is called recursive!
        self.init_parameters_and_components()

        flow_dict = OrderedDict()
        flow_dict['oml:flow'] = OrderedDict()

        flow_dict['oml:flow']['@xmlns:oml'] = 'http://openml.org/openml'

        if config._testmode:
            flow_name = '%s%s' % (config.testsentinel, self.name)
            if len(flow_name) > MAXIMAL_FLOW_LENGTH:
                raise OpenMLRestrictionViolated('Flow name must not be longer '
                                                'than %d characters!' % MAXIMAL_FLOW_LENGTH)
        else:
            flow_name = self.name

        flow_dict['oml:flow']['oml:name'] = flow_name
        if self.external_version is not None:
            flow_dict['oml:flow']['oml:external_version'] = self.external_version
        flow_dict['oml:flow']['oml:description'] = self.description

        components = []
        for component in self.components:
            # If a flow with the same name and same external version already
            # exists, it will be re-used as a sub_flow!
            sub_flow = component['oml:flow']

            component_dict = OrderedDict()
            component_dict['oml:identifier'] = component['oml:identifier']
            component_dict['oml:flow'] = sub_flow._generate_flow_xml(
                return_dict=True)['oml:flow']
            components.append(component_dict)

        flow_dict['oml:flow']['oml:parameter'] = self.parameters
        flow_dict['oml:flow']['oml:component'] = components

        if return_dict:
            return flow_dict

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

        print(xml_description)

        # Checking that the name adheres to oml:casual_string
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
        """ Checks if a flow and its components exists for the given model and
        possibly creates it.

        If the given flow exists on the server, the flow-id will be set on
        this instance and be returned as well. Otherwise the flow will be
        uploaded to the server. Does this recursively for all components.

        Returns
        -------
        flow_id : int
            Flow id on the server.
        """
        import sklearn
        external_version = 'sklearn_' + sklearn.__version__
        _, _, flow_id = _check_flow_exists(self.name, external_version)
        # TODO add numpy and scipy version!
        # MF not sure if this is necessary - what would we get from that?

        if int(flow_id) == -1:
            return_code, response_xml = self.publish()

            response_dict = xmltodict.parse(response_xml)
            flow_id = response_dict['oml:upload_flow']['oml:id']

        # Go through the flow and correctly set all IDs
        flow = get_flow(flow_id)
        queue_local_flow = deque()
        queue_openml_flow = deque()
        queue_local_flow.extendleft(self.components)
        queue_openml_flow.extendleft(flow.components)
        while len(queue_local_flow) > 0 and len(queue_openml_flow) > 0:
            local_component = queue_local_flow.pop()
            openml_component = queue_openml_flow.pop()
            local_component_identifier = \
                'identifier' if 'identifier' in local_component else 'oml:identifier'
            openml_component_identifier = \
                'identifier' if 'identifier' in openml_component else 'oml:identifier'
            if local_component[local_component_identifier] != openml_component[
                    openml_component_identifier]:
                raise ValueError('%s != %s' % (local_component[
                                                   local_component_identifier],
                                               openml_component[
                                                   openml_component_identifier]))
            local_component_flow = 'flow' if 'flow' in local_component else 'oml:flow'
            openml_component_flow = 'flow' if 'flow' in openml_component else 'oml:flow'
            local_component = local_component[local_component_flow]
            openml_component = openml_component[openml_component_flow]
            local_component_name = local_component.name
            openml_component_name = openml_component.name
            if config._testmode:
                local_component_name = local_component_name.replace(config.testsentinel, '')
                openml_component_name = openml_component_name.replace(config.testsentinel, '')

            if local_component_name != openml_component_name:
                raise ValueError('%s != %s' % (local_component_name,
                                               openml_component_name))

            # Transfer the ID
            local_component.id = openml_component.id
            queue_local_flow.extendleft(local_component.components)
            queue_openml_flow.extendleft(openml_component.components)

        if len(queue_local_flow) != len(queue_openml_flow):
            raise ValueError('%s != %s' % (str(queue_local_flow),
                                           str(queue_openml_flow)))

        self.id = flow_id
        return int(flow_id)


def _param_value_to_string(value):
    component_sort = []
    # Try to handle list or tuples from pipelines/feature unions or
    # 'categorical' list to a OneHotEncoder
    if isinstance(value, list) or isinstance(value, tuple):
        all_subcomponents = [(hasattr(elem, '__len__')
                              and len(elem) == 2)
                             and
                             (_is_transformer(elem[1]) or
                              _is_estimator(elem[1]))
                             for elem in value]
        # XOR
        if (not all(all_subcomponents)) ^ (not any(all_subcomponents)):
            raise ValueError('%s mixes elements that are lists like '
                             '("name", estimator/transformer) and '
                             'other values.' % str(value))
        elif not any(all_subcomponents):
            rval = str(value)
        else:
            new_value = []
            for elem in value:
                sub_name = elem[1].__module__ + "." + \
                           elem[1].__class__.__name__
                new_value.append('("%s", "%s")' % (elem[0], sub_name))
                component_sort.append(elem[0])
            new_value = '(' + ', '.join(new_value) + ')'

            rval = new_value

    elif isinstance(value, dict):
        value = copy.deepcopy(value)
        for k in value:
            if _is_distribution(value[k]):
                value[k] = "%s" % str(value[k])
        rval = str(value)
    else:
        rval = str(value)

    return rval, component_sort


def _check_flow_exists(name, external_version):
    """Retrieves the flow id of the flow uniquely identified by name+external version.

    Parameter
    ---------
    name : string
        Name of the flow
    external_version : string
        External version information associated with flow.

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
    if not (type(external_version) is str and len(external_version) > 0):
        raise ValueError('Argument \'version\' should be a non-empty string')

    if config._testmode:
        # It could already be in the name, for example when checking if a
        # downloaded flow exists on the server
        if config.testsentinel not in name:
            name = '%s%s' % (config.testsentinel, name)

    if len(name) > MAXIMAL_FLOW_LENGTH:
        raise OpenMLRestrictionViolated('Flow name must not be longer '
                                        'than %d characters!' % MAXIMAL_FLOW_LENGTH)

    return_code, xml_response = _perform_api_call(
        "/flow/exists/", data={'name': name,
                               'external_version': external_version})
    # TODO check with latest version of code if this raises an exception
    if return_code != 200:
        # fixme raise appropriate error
        raise ValueError("api call failed: %s" % xml_response)
    xml_dict = xmltodict.parse(xml_response)
    flow_id = int(xml_dict['oml:flow_exists']['oml:id'])
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
    # The description field can be empty which causes the server not return
    # the description tag in the XML response.
    description = dic.get('oml:description', '')
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
            data_type = oml_parameter.get('oml:data_type', None)
            default_value = oml_parameter.get('oml:default_value', None)
            parameter_description = oml_parameter.get('oml:description', None)
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

    flow = OpenMLFlow(id=flow_id, uploader=uploader, name=name,
                      version=version, external_version=external_version,
                      description=description, upload_date=upload_date,
                      language=language, dependencies=dependencies,
                      parameters=parameters, components=components)

    flow.model = _construct_model_for_flow(flow)
    _check_dependencies(flow.dependencies, flow.id)
    return flow


def _check_dependencies(dependencies, flow_id):
    """Check dependencies of a flow and emit warning if these are not satisfied.

    A warning is emitted in the following cases:
    * no dependencies are specified
    * the dependencies cannot be imported
    * the dependencies have a different version number

    This function assumes that a dependency has a field __version__.

    Parameters
    ----------
    dependencies : str
        Dependency string in pip's requirements format

    Returns
    -------
    dict
        {name: bool}

    """

    fulfilled_dependencies = {}
    if dependencies is not None:
        dependencies = dependencies.split()
        for dependency in dependencies:
            if dependency:
                dependency_ = re.split(r'([>=<\!]{1,2})', dependency)
                package_name = dependency_[0]

                if len(dependency_) == 1:
                    depended_version = None
                    version_operator = None
                elif len(dependency_) == 3:
                    version_operator = dependency_[1]
                    depended_version = distutils.version.LooseVersion(
                        dependency_[2])
                else:
                    warnings.warn('Cannot parse dependency %s of flow %d' %
                                  (dependency, flow_id))
                    fulfilled_dependencies[dependency] = False
                    continue

                try:
                    module = importlib.import_module(package_name)
                except ImportError:
                    warnings.warn('Cannot import dependency %s of flow %d' %
                                  (package_name, flow_id))
                    fulfilled_dependencies[dependency] = False
                    continue

                installed_version = distutils.version.LooseVersion(
                    module.__version__)

                if depended_version is not None:
                    fulfillment_lookup_table = {
                        '=': depended_version == installed_version,
                        '==': depended_version == installed_version,
                        '>=': depended_version >= installed_version,
                        '>': depended_version > installed_version,
                        '<=': depended_version <= installed_version,
                        '<': depended_version < installed_version,
                        '!=': depended_version != installed_version
                    }
                    if version_operator in fulfillment_lookup_table:
                        dependency_fulfilled = fulfillment_lookup_table[version_operator]
                    else:
                        fulfilled_dependencies[dependency] = False
                        raise ValueError('Cannot parse dependency operator %s '
                                         'for flow %d' % (version_operator,
                                                          flow_id))
                else:
                    dependency_fulfilled = True

                if not dependency_fulfilled:
                    fulfilled_dependencies[dependency] = False
                    warnings.warn('Dependency %s installed in wrong version:'
                                  '%s %s %s' % (module, depended_version,
                                                version_operator,
                                                installed_version))
                else:
                    fulfilled_dependencies[dependency] = True

    return fulfilled_dependencies


def _construct_model_for_flow(flow):
    model_name = flow.name
    if config._testmode:
        model_name = model_name.replace(config.testsentinel, '')

    # Generate a parameters dict because some sklearn objects have mandatory
    # arguments (for example the pipeline)
    parameters = flow.parameters
    parameter_dict = {}
    # If we reconstruct a SearchCV object, it need a dict called
    # param_distributions
    add_param_distributions = {}

    for parameter in parameters:
        name = parameter['name']
        value = parameter.get('default_value', None)

        if value is None:
            continue

        try:
            value = int(value)
        except:
            pass

        if isinstance(value, str):
            try:
                value = float(value)
            except:
                pass

        # FeatureUnion and Pipeline can have lists or tuples as arguments.
        # This tries to recreate the list of tuples of strings representation
        if isinstance(value, six.string_types):
            try:
                value = ast.literal_eval(value)
                if isinstance(value, tuple):
                    value = list(value)

            except:
                pass

        # Figure out whether a string is actually represents a type object.
        # This can happen for the OneHotEncoder which has a dtype parameter
        if isinstance(value, six.string_types):
            match = re.match(r"^<class '([A-Za-z0-9\.]+)'>$", value)

            dtypes = {'float': float,
                      'numpy.float': np.float,
                      'np.float': np.float,
                      'numpy.float16': np.float16,
                      'np.float16': np.float16,
                      'float16': np.float16,
                      'numpy.float32': np.float32,
                      'np.float32': np.float32,
                      'float32': np.float32,
                      'numpy.float64': np.float64,
                      'np.float64': np.float64,
                      'float64': np.float64,
                      }

            if match and match.group(1) in dtypes:
                value = dtypes[match.group(1)]

        if name.startswith('parameter_distribution__') and value == \
                'Unparametrized':
            add_param_distributions = True
            continue

        parameter_dict[name] = value

    if len(flow.components) > 0:
        added = False
        for component in flow.components:
            # Search for hyperparameters to which we can assign the components
            for name in parameter_dict:
                # List of tuple like for Pipeline or FeatureUnion
                if isinstance(parameter_dict[name], list):
                    steps = parameter_dict[name]
                    for i in range(len(steps)):
                        if steps[i][0] == component['identifier']:
                            steps[i] = (component['identifier'],
                                        component['flow'].model)
                            added = True

            # A regular component
            if added is False:
                parameter_dict[component['identifier']] = component[
                    'flow'].model

    # Remove everything after the first bracket
    pos = model_name.find('(')
    if pos >= 0:
        model_name = model_name[:pos]

    module_name = model_name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow %d.' %
                      (model_name, flow.id))
        return None

    if add_param_distributions:
        parameter_dict['param_distributions'] = {}

    model = model_class(**parameter_dict)

    return model
