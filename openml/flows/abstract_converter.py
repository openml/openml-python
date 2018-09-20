from collections import OrderedDict
from distutils.version import LooseVersion
import importlib
import openml
import re
import copy
import sys
import inspect

from abc import abstractmethod

DEPENDENCIES_PATTERN = re.compile(
    '^(?P<name>[\w\-]+)((?P<operation>==|>=|>)(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$')

class AbstractConverter(object):
    def __init__(self, model):
        self._external_version = None
        self._model = model

        # stores all entities that should become subcomponents
        self._sub_components = OrderedDict()
        # stores the keys of all subcomponents that should become
        self._sub_components_explicit = set()
        self._parameters = OrderedDict()
        self._parameters_meta_info = OrderedDict()

        self.extract_information_from_model()
        self.check_multiple_occurence_of_component_in_flow()

    @staticmethod
    @abstractmethod
    def from_flow(flow, components=None, initialize_with_defaults=False):
        """Initializes a model based on a flow.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialzied
            parameter value that is accepted by)

        components : dict


        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        mixed
        """
        print("asdf")

    @abstractmethod
    def to_flow(self):
        """Creates an OpenML flow of the models.

        Returns
        -------
        OpenMLFlow
        """

    @abstractmethod
    def extract_information_from_model(self):
        """
        """

    @abstractmethod
    def check_multiple_occurence_of_component_in_flow(self):
        """
        """
        to_visit_stack = []
        to_visit_stack.extend(self._sub_components.values())
        known_sub_components = set()
        while len(to_visit_stack) > 0:
            visitee = to_visit_stack.pop()
            if visitee.name in known_sub_components:
                raise ValueError('Found a second occurence of component %s when '
                                 'trying to serialize %s.' % (visitee.name, self._model))
            else:
                known_sub_components.add(visitee.name)
                to_visit_stack.extend(visitee.components.values())


    @property
    def external_version(self):
        if self._external_version:
            return self._external_version
        # Create external version string for a flow, given the model and the
        # already parsed dictionary of sub_components. Retrieves the external
        # version of all subcomponents, which themselves already contain all
        # requirements for their subcomponents. The external version string is a
        # sorted concatenation of all modules which are present in this run.
        model_package_name = self._model.__module__.split('.')[0]
        module = importlib.import_module(model_package_name)
        model_package_version_number = module.__version__
        external_version = self.format_external_version(model_package_name,
                                                    model_package_version_number)
        openml_version = self.format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in self._sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)
        external_versions = list(sorted(external_versions))
        self._external_version = ','.join(external_versions)
        return self._external_version

    @staticmethod
    def format_external_version(model_package_name, model_package_version_number):
        return '%s==%s' % (model_package_name, model_package_version_number)

    @staticmethod
    def _get_fn_arguments_with_defaults(fn_name):
        """
        Returns i) a dict with all parameter names (as key) that have a default value (as value) and ii) a set with all
        parameter names that do not have a default

        Parameters
        ----------
        fn_name : callable
            The function of which we want to obtain the defaults

        Returns
        -------
        params_with_defaults: dict
            a dict mapping parameter name to the default value
        params_without_defaults: dict
            a set with all parameters that do not have a default value
        """
        if sys.version_info[0] >= 3:
            signature = inspect.getfullargspec(fn_name)
        else:
            signature = inspect.getargspec(fn_name)

        # len(signature.defaults) <= len(signature.args). Thus, by definition, the last entrees of signature.args
        # actually have defaults. Iterate backwards over both arrays to keep them in sync
        len_defaults = len(signature.defaults) if signature.defaults is not None else 0
        params_with_defaults = {signature.args[-1*i]: signature.defaults[-1*i] for i in range(1, len_defaults + 1)}
        # retrieve the params without defaults
        params_without_defaults = {signature.args[i] for i in range(len(signature.args) - len_defaults)}
        return params_with_defaults, params_without_defaults

    @classmethod
    def _deserialize_model(cls, flow, keep_defaults):
        model_name = flow.class_name
        cls._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        components = flow.components
        parameter_dict = OrderedDict()

        # Do a shallow copy of the components dictionary so we can remove the
        # components from this copy once we added them into the pipeline. This
        # allows us to not consider them any more when looping over the
        # components, but keeping the dictionary of components untouched in the
        # original components dictionary.
        components_ = copy.copy(components)

        for name in parameters:
            value = parameters.get(name)
            rval = cls.from_flow(value, components=components_, initialize_with_defaults=keep_defaults)
            parameter_dict[name] = rval

        for name in components:
            if name in parameter_dict:
                continue
            if name not in components_:
                continue
            value = components[name]
            rval = cls.from_flow(value, **kwargs)
            parameter_dict[name] = rval

        module_name = model_name.rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])

        if keep_defaults:
            # obtain all params with a default
            param_defaults, _ = cls._get_fn_arguments_with_defaults(model_class.__init__)

            # delete the params that have a default from the dict,
            # so they get initialized with their default value
            # except [...]
            for param in param_defaults:
                # [...] the ones that also have a key in the components dict. As OpenML stores different flows for ensembles
                # with different (base-)components, in OpenML terms, these are not considered hyperparameters but rather
                # constants (i.e., changing them would result in a different flow)
                if param not in components.keys():
                    del parameter_dict[param]
        return model_class(**parameter_dict)

    @classmethod
    def _check_dependencies(cls, dependencies):
        if not dependencies:
            return

        dependencies = dependencies.split('\n')
        for dependency_string in dependencies:
            match = DEPENDENCIES_PATTERN.match(dependency_string)
            dependency_name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')

            module = importlib.import_module(dependency_name)
            required_version = LooseVersion(version)
            installed_version = LooseVersion(module.__version__)

            if operation == '==':
                check = required_version == installed_version
            elif operation == '>':
                check = installed_version > required_version
            elif operation == '>=':
                check = installed_version > required_version or \
                        installed_version == required_version
            else:
                raise NotImplementedError(
                    'operation \'%s\' is not supported' % operation)
            if not check:
                raise ValueError('Trying to deserialize a model with dependency '
                                 '%s not satisfied.' % dependency_string)


