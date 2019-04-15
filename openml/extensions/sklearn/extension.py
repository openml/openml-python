from collections import OrderedDict  # noqa: F401
import copy
from distutils.version import LooseVersion
import importlib
import inspect
import json
import logging
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import scipy.stats
import sklearn.base
import sklearn.model_selection
import sklearn.pipeline

import openml
from openml.exceptions import PyOpenMLError
from openml.extensions import Extension, register_extension
from openml.flows import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration, PREFIX
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLLearningCurveTask,
    OpenMLClusteringTask,
    OpenMLRegressionTask,
)


if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError


DEPENDENCIES_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)((?P<operation>==|>=|>)'
    r'(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$'
)


SIMPLE_NUMPY_TYPES = [nptype for type_cat, nptypes in np.sctypes.items()
                      for nptype in nptypes if type_cat != 'others']
SIMPLE_TYPES = tuple([bool, int, float, str] + SIMPLE_NUMPY_TYPES)


class SklearnExtension(Extension):
    """Connect scikit-learn to OpenML-Python."""

    ################################################################################################
    # General setup

    @classmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        """Check whether a given describes a scikit-learn estimator.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_sklearn_flow(flow)

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``sklearn.base.BaseEstimator``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, sklearn.base.BaseEstimator)

    ################################################################################################
    # Methods for flow serialization and de-serialization

    def flow_to_model(self, flow: 'OpenMLFlow', initialize_with_defaults: bool = False) -> Any:
        """Initializes a sklearn model based on a flow.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        mixed
        """
        return self._deserialize_sklearn(flow, initialize_with_defaults=initialize_with_defaults)

    def _deserialize_sklearn(
        self,
        o: Any,
        components: Optional[Dict] = None,
        initialize_with_defaults: bool = False,
        recursion_depth: int = 0,
    ) -> Any:
        """Recursive function to deserialize a scikit-learn flow.

        This function delegates all work to the respective functions to deserialize special data
        structures etc.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        components : dict


        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        recursion_depth : int
            The depth at which this flow is called, mostly for debugging
            purposes

        Returns
        -------
        mixed
        """

        logging.info('-%s flow_to_sklearn START o=%s, components=%s, '
                     'init_defaults=%s' % ('-' * recursion_depth, o, components,
                                           initialize_with_defaults))
        depth_pp = recursion_depth + 1  # shortcut var, depth plus plus

        # First, we need to check whether the presented object is a json string.
        # JSON strings are used to encoder parameter values. By passing around
        # json strings for parameters, we make sure that we can flow_to_sklearn
        # the parameter values to the correct type.

        if isinstance(o, str):
            try:
                o = json.loads(o)
            except JSONDecodeError:
                pass

        if isinstance(o, dict):
            # Check if the dict encodes a 'special' object, which could not
            # easily converted into a string, but rather the information to
            # re-create the object were stored in a dictionary.
            if 'oml-python:serialized_object' in o:
                serialized_type = o['oml-python:serialized_object']
                value = o['value']
                if serialized_type == 'type':
                    rval = self._deserialize_type(value)
                elif serialized_type == 'rv_frozen':
                    rval = self._deserialize_rv_frozen(value)
                elif serialized_type == 'function':
                    rval = self._deserialize_function(value)
                elif serialized_type == 'component_reference':
                    assert components is not None  # Necessary for mypy
                    value = self._deserialize_sklearn(value, recursion_depth=depth_pp)
                    step_name = value['step_name']
                    key = value['key']
                    component = self._deserialize_sklearn(
                        components[key],
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp
                    )
                    # The component is now added to where it should be used
                    # later. It should not be passed to the constructor of the
                    # main flow object.
                    del components[key]
                    if step_name is None:
                        rval = component
                    elif 'argument_1' not in value:
                        rval = (step_name, component)
                    else:
                        rval = (step_name, component, value['argument_1'])
                elif serialized_type == 'cv_object':
                    rval = self._deserialize_cross_validator(
                        value, recursion_depth=recursion_depth
                    )
                else:
                    raise ValueError('Cannot flow_to_sklearn %s' % serialized_type)

            else:
                rval = OrderedDict(
                    (
                        self._deserialize_sklearn(
                            o=key,
                            components=components,
                            initialize_with_defaults=initialize_with_defaults,
                            recursion_depth=depth_pp,
                        ),
                        self._deserialize_sklearn(
                            o=value,
                            components=components,
                            initialize_with_defaults=initialize_with_defaults,
                            recursion_depth=depth_pp,
                        )
                    )
                    for key, value in sorted(o.items())
                )
        elif isinstance(o, (list, tuple)):
            rval = [
                self._deserialize_sklearn(
                    o=element,
                    components=components,
                    initialize_with_defaults=initialize_with_defaults,
                    recursion_depth=depth_pp,
                )
                for element in o
            ]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, str)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            if not self._is_sklearn_flow(o):
                raise ValueError('Only sklearn flows can be reinstantiated')
            rval = self._deserialize_model(
                flow=o,
                keep_defaults=initialize_with_defaults,
                recursion_depth=recursion_depth,
            )
        else:
            raise TypeError(o)
        logging.info('-%s flow_to_sklearn END   o=%s, rval=%s'
                     % ('-' * recursion_depth, o, rval))
        return rval

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        """Transform a scikit-learn model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types
        return self._serialize_sklearn(model)

    def _serialize_sklearn(self, o: Any, parent_model: Optional[Any] = None) -> Any:
        rval = None  # type: Any

        # TODO: assert that only on first recursion lvl `parent_model` can be None
        if self.is_estimator(o):
            # is the main model or a submodel
            rval = self._serialize_model(o)
        elif isinstance(o, (list, tuple)):
            # TODO: explain what type of parameter is here
            rval = [self._serialize_sklearn(element, parent_model) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, SIMPLE_TYPES) or o is None:
            if isinstance(o, tuple(SIMPLE_NUMPY_TYPES)):
                o = o.item()
            # base parameter values
            rval = o
        elif isinstance(o, dict):
            # TODO: explain what type of parameter is here
            if not isinstance(o, OrderedDict):
                o = OrderedDict([(key, value) for key, value in sorted(o.items())])

            rval = OrderedDict()
            for key, value in o.items():
                if not isinstance(key, str):
                    raise TypeError('Can only use string as keys, you passed '
                                    'type %s for value %s.' %
                                    (type(key), str(key)))
                key = self._serialize_sklearn(key, parent_model)
                value = self._serialize_sklearn(value, parent_model)
                rval[key] = value
            rval = rval
        elif isinstance(o, type):
            # TODO: explain what type of parameter is here
            rval = self._serialize_type(o)
        elif isinstance(o, scipy.stats.distributions.rv_frozen):
            rval = self._serialize_rv_frozen(o)
        # This only works for user-defined functions (and not even partial).
        # I think this is exactly what we want here as there shouldn't be any
        # built-in or functool.partials in a pipeline
        elif inspect.isfunction(o):
            # TODO: explain what type of parameter is here
            rval = self._serialize_function(o)
        elif self._is_cross_validator(o):
            # TODO: explain what type of parameter is here
            rval = self._serialize_cross_validator(o)
        else:
            raise TypeError(o, type(o))

        return rval

    def get_version_information(self) -> List[str]:
        """List versions of libraries required by the flow.

        Libraries listed are ``Python``, ``scikit-learn``, ``numpy`` and ``scipy``.

        Returns
        -------
        List
        """

        # This can possibly be done by a package such as pyxb, but I could not get
        # it to work properly.
        import sklearn
        import scipy
        import numpy

        major, minor, micro, _, _ = sys.version_info
        python_version = 'Python_{}.'.format(
            ".".join([str(major), str(minor), str(micro)]))
        sklearn_version = 'Sklearn_{}.'.format(sklearn.__version__)
        numpy_version = 'NumPy_{}.'.format(numpy.__version__)
        scipy_version = 'SciPy_{}.'.format(scipy.__version__)

        return [python_version, sklearn_version, numpy_version, scipy_version]

    def create_setup_string(self, model: Any) -> str:
        """Create a string which can be used to reinstantiate the given model.

        Parameters
        ----------
        model : Any

        Returns
        -------
        str
        """
        run_environment = " ".join(self.get_version_information())
        # fixme str(model) might contain (...)
        return run_environment + " " + str(model)

    def _is_cross_validator(self, o: Any) -> bool:
        return isinstance(o, sklearn.model_selection.BaseCrossValidator)

    @classmethod
    def _is_sklearn_flow(cls, flow: OpenMLFlow) -> bool:
        return (
            flow.external_version.startswith('sklearn==')
            or ',sklearn==' in flow.external_version
        )

    def _serialize_model(self, model: Any) -> OpenMLFlow:
        """Create an OpenMLFlow.

        Calls `sklearn_to_flow` recursively to properly serialize the
        parameters to strings and the components (other models) to OpenMLFlows.

        Parameters
        ----------
        model : sklearn estimator

        Returns
        -------
        OpenMLFlow

        """

        # Get all necessary information about the model objects itself
        parameters, parameters_meta_info, subcomponents, subcomponents_explicit = \
            self._extract_information_from_model(model)

        # Check that a component does not occur multiple times in a flow as this
        # is not supported by OpenML
        self._check_multiple_occurence_of_component_in_flow(model, subcomponents)

        # Create a flow name, which contains all components in brackets, e.g.:
        # RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),
        # StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
        class_name = model.__module__ + "." + model.__class__.__name__

        # will be part of the name (in brackets)
        sub_components_names = ""
        for key in subcomponents:
            if key in subcomponents_explicit:
                sub_components_names += "," + key + "=" + subcomponents[key].name
            else:
                sub_components_names += "," + subcomponents[key].name

        if sub_components_names:
            # slice operation on string in order to get rid of leading comma
            name = '%s(%s)' % (class_name, sub_components_names[1:])
        else:
            name = class_name

        # Get the external versions of all sub-components
        external_version = self._get_external_version_string(model, subcomponents)

        dependencies = '\n'.join([
            self._format_external_version(
                'sklearn',
                sklearn.__version__,
            ),
            'numpy>=1.6.1',
            'scipy>=0.9',
        ])

        sklearn_version = self._format_external_version('sklearn', sklearn.__version__)
        sklearn_version_formatted = sklearn_version.replace('==', '_')
        flow = OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created scikit-learn flow.',
                          model=model,
                          components=subcomponents,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version,
                          tags=['openml-python', 'sklearn', 'scikit-learn',
                                'python', sklearn_version_formatted,
                                # TODO: add more tags based on the scikit-learn
                                # module a flow is in? For example automatically
                                # annotate a class of sklearn.svm.SVC() with the
                                # tag svm?
                                ],
                          language='English',
                          # TODO fill in dependencies!
                          dependencies=dependencies)

        return flow

    def _get_external_version_string(
        self,
        model: Any,
        sub_components: Dict[str, OpenMLFlow],
    ) -> str:
        # Create external version string for a flow, given the model and the
        # already parsed dictionary of sub_components. Retrieves the external
        # version of all subcomponents, which themselves already contain all
        # requirements for their subcomponents. The external version string is a
        # sorted concatenation of all modules which are present in this run.
        model_package_name = model.__module__.split('.')[0]
        module = importlib.import_module(model_package_name)
        model_package_version_number = module.__version__  # type: ignore
        external_version = self._format_external_version(
            model_package_name, model_package_version_number,
        )
        openml_version = self._format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)
        return ','.join(list(sorted(external_versions)))

    def _check_multiple_occurence_of_component_in_flow(
        self,
        model: Any,
        sub_components: Dict[str, OpenMLFlow],
    ) -> None:
        to_visit_stack = []  # type: List[OpenMLFlow]
        to_visit_stack.extend(sub_components.values())
        known_sub_components = set()  # type: Set[OpenMLFlow]
        while len(to_visit_stack) > 0:
            visitee = to_visit_stack.pop()
            if visitee.name in known_sub_components:
                raise ValueError('Found a second occurence of component %s when '
                                 'trying to serialize %s.' % (visitee.name, model))
            else:
                known_sub_components.add(visitee.name)
                to_visit_stack.extend(visitee.components.values())

    def _extract_information_from_model(
        self,
        model: Any,
    ) -> Tuple[
        'OrderedDict[str, Optional[str]]',
        'OrderedDict[str, Optional[Dict]]',
        'OrderedDict[str, OpenMLFlow]',
        Set,
    ]:
        # This function contains four "global" states and is quite long and
        # complicated. If it gets to complicated to ensure it's correctness,
        # it would be best to make it a class with the four "global" states being
        # the class attributes and the if/elif/else in the for-loop calls to
        # separate class methods

        # stores all entities that should become subcomponents
        sub_components = OrderedDict()  # type: OrderedDict[str, OpenMLFlow]
        # stores the keys of all subcomponents that should become
        sub_components_explicit = set()
        parameters = OrderedDict()  # type: OrderedDict[str, Optional[str]]
        parameters_meta_info = OrderedDict()  # type: OrderedDict[str, Optional[Dict]]

        model_parameters = model.get_params(deep=False)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._serialize_sklearn(v, model)

            def flatten_all(list_):
                """ Flattens arbitrary depth lists of lists (e.g. [[1,2],[3,[1]]] -> [1,2,3,1]). """
                for el in list_:
                    if isinstance(el, (list, tuple)):
                        yield from flatten_all(el)
                    else:
                        yield el

            # In case rval is a list of lists (or tuples), we need to identify two situations:
            # - sklearn pipeline steps, feature union or base classifiers in voting classifier.
            #   They look like e.g. [("imputer", Imputer()), ("classifier", SVC())]
            # - a list of lists with simple types (e.g. int or str), such as for an OrdinalEncoder
            #   where all possible values for each feature are described: [[0,1,2], [1,2,5]]
            is_non_empty_list_of_lists_with_same_type = (
                isinstance(rval, (list, tuple))
                and len(rval) > 0
                and isinstance(rval[0], (list, tuple))
                and all([isinstance(rval_i, type(rval[0])) for rval_i in rval])
            )

            # Check that all list elements are of simple types.
            nested_list_of_simple_types = (
                is_non_empty_list_of_lists_with_same_type
                and all([isinstance(el, SIMPLE_TYPES) for el in flatten_all(rval)])
            )

            if is_non_empty_list_of_lists_with_same_type and not nested_list_of_simple_types:
                # If a list of lists is identified that include 'non-simple' types (e.g. objects),
                # we assume they are steps in a pipeline, feature union, or base classifiers in
                # a voting classifier.
                parameter_value = list()  # type: List
                reserved_keywords = set(model.get_params(deep=False).keys())

                for sub_component_tuple in rval:
                    identifier = sub_component_tuple[0]
                    sub_component = sub_component_tuple[1]
                    sub_component_type = type(sub_component_tuple)
                    if not 2 <= len(sub_component_tuple) <= 3:
                        # length 2 is for {VotingClassifier.estimators,
                        # Pipeline.steps, FeatureUnion.transformer_list}
                        # length 3 is for ColumnTransformer
                        msg = 'Length of tuple does not match assumptions'
                        raise ValueError(msg)
                    if not isinstance(sub_component, (OpenMLFlow, type(None))):
                        msg = 'Second item of tuple does not match assumptions. ' \
                              'Expected OpenMLFlow, got %s' % type(sub_component)
                        raise TypeError(msg)

                    if identifier in reserved_keywords:
                        parent_model = "{}.{}".format(model.__module__,
                                                      model.__class__.__name__)
                        msg = 'Found element shadowing official ' \
                              'parameter for %s: %s' % (parent_model,
                                                        identifier)
                        raise PyOpenMLError(msg)

                    if sub_component is None:
                        # In a FeatureUnion it is legal to have a None step

                        pv = [identifier, None]
                        if sub_component_type is tuple:
                            parameter_value.append(tuple(pv))
                        else:
                            parameter_value.append(pv)

                    else:
                        # Add the component to the list of components, add a
                        # component reference as a placeholder to the list of
                        # parameters, which will be replaced by the real component
                        # when deserializing the parameter
                        sub_components_explicit.add(identifier)
                        sub_components[identifier] = sub_component
                        component_reference = OrderedDict()  # type: Dict[str, Union[str, Dict]]
                        component_reference['oml-python:serialized_object'] = 'component_reference'
                        cr_value = OrderedDict()  # type: Dict[str, Any]
                        cr_value['key'] = identifier
                        cr_value['step_name'] = identifier
                        if len(sub_component_tuple) == 3:
                            cr_value['argument_1'] = sub_component_tuple[2]
                        component_reference['value'] = cr_value
                        parameter_value.append(component_reference)

                # Here (and in the elif and else branch below) are the only
                # places where we encode a value as json to make sure that all
                # parameter values still have the same type after
                # deserialization
                if isinstance(rval, tuple):
                    parameter_json = json.dumps(tuple(parameter_value))
                else:
                    parameter_json = json.dumps(parameter_value)
                parameters[k] = parameter_json

            elif isinstance(rval, OpenMLFlow):

                # A subcomponent, for example the base model in
                # AdaBoostClassifier
                sub_components[k] = rval
                sub_components_explicit.add(k)
                component_reference = OrderedDict()
                component_reference['oml-python:serialized_object'] = 'component_reference'
                cr_value = OrderedDict()
                cr_value['key'] = k
                cr_value['step_name'] = None
                component_reference['value'] = cr_value
                cr = self._serialize_sklearn(component_reference, model)
                parameters[k] = json.dumps(cr)

            else:
                # a regular hyperparameter
                if not (hasattr(rval, '__len__') and len(rval) == 0):
                    rval = json.dumps(rval)
                    parameters[k] = rval
                else:
                    parameters[k] = None

            parameters_meta_info[k] = OrderedDict((('description', None), ('data_type', None)))

        return parameters, parameters_meta_info, sub_components, sub_components_explicit

    def _get_fn_arguments_with_defaults(self, fn_name: Callable) -> Tuple[Dict, Set]:
        """
        Returns:
            i) a dict with all parameter names that have a default value, and
            ii) a set with all parameter names that do not have a default

        Parameters
        ----------
        fn_name : callable
            The function of which we want to obtain the defaults

        Returns
        -------
        params_with_defaults: dict
            a dict mapping parameter name to the default value
        params_without_defaults: set
            a set with all parameters that do not have a default value
        """
        # parameters with defaults are optional, all others are required.
        signature = inspect.getfullargspec(fn_name)
        if signature.defaults:
            optional_params = dict(zip(reversed(signature.args), reversed(signature.defaults)))
        else:
            optional_params = dict()
        required_params = {arg for arg in signature.args if arg not in optional_params}
        return optional_params, required_params

    def _deserialize_model(
        self,
        flow: OpenMLFlow,
        keep_defaults: bool,
        recursion_depth: int,
    ) -> Any:
        logging.info('-%s deserialize %s' % ('-' * recursion_depth, flow.name))
        model_name = flow.class_name
        self._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        components = flow.components
        parameter_dict = OrderedDict()  # type: Dict[str, Any]

        # Do a shallow copy of the components dictionary so we can remove the
        # components from this copy once we added them into the pipeline. This
        # allows us to not consider them any more when looping over the
        # components, but keeping the dictionary of components untouched in the
        # original components dictionary.
        components_ = copy.copy(components)

        for name in parameters:
            value = parameters.get(name)
            logging.info('--%s flow_parameter=%s, value=%s' %
                         ('-' * recursion_depth, name, value))
            rval = self._deserialize_sklearn(
                value,
                components=components_,
                initialize_with_defaults=keep_defaults,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        for name in components:
            if name in parameter_dict:
                continue
            if name not in components_:
                continue
            value = components[name]
            logging.info('--%s flow_component=%s, value=%s'
                         % ('-' * recursion_depth, name, value))
            rval = self._deserialize_sklearn(
                value,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        module_name = model_name.rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])

        if keep_defaults:
            # obtain all params with a default
            param_defaults, _ = \
                self._get_fn_arguments_with_defaults(model_class.__init__)

            # delete the params that have a default from the dict,
            # so they get initialized with their default value
            # except [...]
            for param in param_defaults:
                # [...] the ones that also have a key in the components dict.
                # As OpenML stores different flows for ensembles with different
                # (base-)components, in OpenML terms, these are not considered
                # hyperparameters but rather constants (i.e., changing them would
                # result in a different flow)
                if param not in components.keys():
                    del parameter_dict[param]
        return model_class(**parameter_dict)

    def _check_dependencies(self, dependencies: str) -> None:
        if not dependencies:
            return

        dependencies_list = dependencies.split('\n')
        for dependency_string in dependencies_list:
            match = DEPENDENCIES_PATTERN.match(dependency_string)
            if not match:
                raise ValueError('Cannot parse dependency %s' % dependency_string)

            dependency_name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')

            module = importlib.import_module(dependency_name)
            required_version = LooseVersion(version)
            installed_version = LooseVersion(module.__version__)  # type: ignore

            if operation == '==':
                check = required_version == installed_version
            elif operation == '>':
                check = installed_version > required_version
            elif operation == '>=':
                check = (installed_version > required_version
                         or installed_version == required_version)
            else:
                raise NotImplementedError(
                    'operation \'%s\' is not supported' % operation)
            if not check:
                raise ValueError('Trying to deserialize a model with dependency '
                                 '%s not satisfied.' % dependency_string)

    def _serialize_type(self, o: Any) -> 'OrderedDict[str, str]':
        mapping = {float: 'float',
                   np.float: 'np.float',
                   np.float32: 'np.float32',
                   np.float64: 'np.float64',
                   int: 'int',
                   np.int: 'np.int',
                   np.int32: 'np.int32',
                   np.int64: 'np.int64'}
        ret = OrderedDict()  # type: 'OrderedDict[str, str]'
        ret['oml-python:serialized_object'] = 'type'
        ret['value'] = mapping[o]
        return ret

    def _deserialize_type(self, o: str) -> Any:
        mapping = {'float': float,
                   'np.float': np.float,
                   'np.float32': np.float32,
                   'np.float64': np.float64,
                   'int': int,
                   'np.int': np.int,
                   'np.int32': np.int32,
                   'np.int64': np.int64}
        return mapping[o]

    def _serialize_rv_frozen(self, o: Any) -> 'OrderedDict[str, Union[str, Dict]]':
        args = o.args
        kwds = o.kwds
        a = o.a
        b = o.b
        dist = o.dist.__class__.__module__ + '.' + o.dist.__class__.__name__
        ret = OrderedDict()  # type: 'OrderedDict[str, Union[str, Dict]]'
        ret['oml-python:serialized_object'] = 'rv_frozen'
        ret['value'] = OrderedDict((('dist', dist), ('a', a), ('b', b),
                                    ('args', args), ('kwds', kwds)))
        return ret

    def _deserialize_rv_frozen(self, o: 'OrderedDict[str, str]') -> Any:
        args = o['args']
        kwds = o['kwds']
        a = o['a']
        b = o['b']
        dist_name = o['dist']

        module_name = dist_name.rsplit('.', 1)
        try:
            rv_class = getattr(importlib.import_module(module_name[0]),
                               module_name[1])
        except AttributeError:
            warnings.warn('Cannot create model %s for flow.' % dist_name)
            return None

        dist = scipy.stats.distributions.rv_frozen(rv_class(), *args, **kwds)
        dist.a = a
        dist.b = b

        return dist

    def _serialize_function(self, o: Callable) -> 'OrderedDict[str, str]':
        name = o.__module__ + '.' + o.__name__
        ret = OrderedDict()  # type: 'OrderedDict[str, str]'
        ret['oml-python:serialized_object'] = 'function'
        ret['value'] = name
        return ret

    def _deserialize_function(self, name: str) -> Callable:
        module_name = name.rsplit('.', 1)
        function_handle = getattr(importlib.import_module(module_name[0]), module_name[1])
        return function_handle

    def _serialize_cross_validator(self, o: Any) -> 'OrderedDict[str, Union[str, Dict]]':
        ret = OrderedDict()  # type: 'OrderedDict[str, Union[str, Dict]]'

        parameters = OrderedDict()  # type: 'OrderedDict[str, Any]'

        # XXX this is copied from sklearn.model_selection._split
        cls = o.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        # Ignore varargs, kw and default values and pop self
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        if init is object.__init__:
            args = []  # type: List
        else:
            args = sorted([p.name for p in init_signature.parameters.values()
                           if p.name != 'self' and p.kind != p.VAR_KEYWORD])

        for key in args:
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", DeprecationWarning)
                value = getattr(o, key, None)
                if w is not None and len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue

            if not (hasattr(value, '__len__') and len(value) == 0):
                value = json.dumps(value)
                parameters[key] = value
            else:
                parameters[key] = None

        ret['oml-python:serialized_object'] = 'cv_object'
        name = o.__module__ + "." + o.__class__.__name__
        value = OrderedDict([('name', name), ('parameters', parameters)])
        ret['value'] = value

        return ret

    def _deserialize_cross_validator(
        self,
        value: 'OrderedDict[str, Any]',
        recursion_depth: int,
    ) -> Any:
        model_name = value['name']
        parameters = value['parameters']

        module_name = model_name.rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
        for parameter in parameters:
            parameters[parameter] = self._deserialize_sklearn(
                parameters[parameter],
                recursion_depth=recursion_depth + 1,
            )
        return model_class(**parameters)

    def _format_external_version(
        self,
        model_package_name: str,
        model_package_version_number: str,
    ) -> str:
        return '%s==%s' % (model_package_name, model_package_version_number)

    @staticmethod
    def _check_parameter_value_recursive(param_grid: Union[Dict, List[Dict]], parameter_name: str, legal_values: Optional[List]):
        """
        Checks within a flow (recursively) whether a given hyperparameter complies to one of the values presented in a
        grid. If the hyperparameter does not exist in the grid, True is returned.

        Parameters
        ----------
        param_grid: Union[Dict, List[Dict]]
            Dict mapping from hyperparameter list to value, to a list of such dicts

        parameter_name: str
            The hyperparameter that needs to be inspected

        legal_values: List
            The values that are accepted. None if no values are legal (the presence of the hyperparameter will trigger
            to return False)

        Returns
        -------
        bool
            True if all occurrences of the hyperparameter only have legal values, False otherwise

        """
        if isinstance(param_grid, dict):
            for param, value in param_grid.items():
                # n_jobs is scikitlearn parameter for paralizing jobs
                if param.split('__')[-1] == parameter_name:
                    # 0 = illegal value (?), 1 / None = use one core,
                    # n = use n cores,
                    # -1 = use all available cores -> this makes it hard to
                    # measure runtime in a fair way
                    if legal_values is None or value not in legal_values:
                        return False
            return True
        elif isinstance(param_grid, list):
            return all(
                SklearnExtension._check_parameter_value_recursive(sub_grid, parameter_name, legal_values)
                for sub_grid in param_grid
            )

    def _prevent_optimize_n_jobs(self, model):
        """
        Ensures that HPO classess will not optimize the n_jobs hyperparameter

        Parameters:
        -----------
        model:
            The model that will be fitted
        """
        if self.is_hpo_class(model):
            if isinstance(model, sklearn.model_selection.GridSearchCV):
                param_distributions = model.param_grid
            elif isinstance(model, sklearn.model_selection.RandomizedSearchCV):
                param_distributions = model.param_distributions
            else:
                if hasattr(model, 'param_distributions'):
                    param_distributions = model.param_distributions
                else:
                    raise AttributeError('Using subclass BaseSearchCV other than '
                                         '{GridSearchCV, RandomizedSearchCV}. '
                                         'Could not find attribute '
                                         'param_distributions.')
                print('Warning! Using subclass BaseSearchCV other than '
                      '{GridSearchCV, RandomizedSearchCV}. '
                      'Should implement param check. ')

            if not SklearnExtension._check_parameter_value_recursive(param_distributions, 'n_jobs', None):
                raise PyOpenMLError('openml-python should not be used to '
                                    'optimize the n_jobs parameter.')

    def _can_measure_cputime(self, model: Any) -> bool:
        """
        Returns True if the parameter settings of model are chosen s.t. the model
        will run on a single core (if so, openml-python can measure cpu-times)

        Parameters:
        -----------
        model:
            The model that will be fitted

        Returns:
        --------
        bool:
            True if all n_jobs parameters will be either set to None or 1, False otherwise
        """
        if not (
                isinstance(model, sklearn.base.BaseEstimator) or self.is_hpo_class(model)
        ):
            raise ValueError('model should be BaseEstimator or BaseSearchCV')

        # check the parameters for n_jobs
        return SklearnExtension._check_parameter_value_recursive(model.get_params(), 'n_jobs', [1, None])

    def _can_measure_wallclocktime(self, model: Any) -> bool:
        """
        Returns True if the parameter settings of model are chosen s.t. the model
        will run on a preset number of cores (if so, openml-python can measure wallclock time)

        Parameters:
        -----------
        model:
            The model that will be fitted

        Returns:
        --------
        bool:
            True if none n_jobs parameters is set ot -1, False otherwise
        """
        if not (
                isinstance(model, sklearn.base.BaseEstimator) or self.is_hpo_class(model)
        ):
            raise ValueError('model should be BaseEstimator or BaseSearchCV')

        # check the parameters for n_jobs
        return not SklearnExtension._check_parameter_value_recursive(model.get_params(), 'n_jobs', [-1])

    ################################################################################################
    # Methods for performing runs with extension modules

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a scikit-learn estimator.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        o = model
        return hasattr(o, 'fit') and hasattr(o, 'get_params') and hasattr(o, 'set_params')

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """Set the random state of all the unseeded components of a model and return the seeded
        model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Models that are already seeded will maintain the seed. In this case,
        only integer seeds are allowed (An exception is raised when a RandomState was used as
        seed).

        Parameters
        ----------
        model : sklearn model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """

        def _seed_current_object(current_value):
            if isinstance(current_value, int):  # acceptable behaviour
                return False
            elif isinstance(current_value, np.random.RandomState):
                raise ValueError(
                    'Models initialized with a RandomState object are not '
                    'supported. Please seed with an integer. ')
            elif current_value is not None:
                raise ValueError(
                    'Models should be seeded with int or None (this should never '
                    'happen). ')
            else:
                return True

        rs = np.random.RandomState(seed)
        model_params = model.get_params()
        random_states = {}
        for param_name in sorted(model_params):
            if 'random_state' in param_name:
                current_value = model_params[param_name]
                # important to draw the value at this point (and not in the if
                # statement) this way we guarantee that if a different set of
                # subflows is seeded, the same number of the random generator is
                # used
                new_value = rs.randint(0, 2 ** 16)
                if _seed_current_object(current_value):
                    random_states[param_name] = new_value

            # Also seed CV objects!
            elif isinstance(model_params[param_name], sklearn.model_selection.BaseCrossValidator):
                if not hasattr(model_params[param_name], 'random_state'):
                    continue

                current_value = model_params[param_name].random_state
                new_value = rs.randint(0, 2 ** 16)
                if _seed_current_object(current_value):
                    model_params[param_name].random_state = new_value

        model.set_params(**random_states)
        return model

    def _run_model_on_fold(
        self,
        model: Any,
        task: 'OpenMLTask',
        rep_no: int,
        fold_no: int,
        sample_no: int,
        add_local_measures: bool,
    ) -> Tuple[List[List], List[List], 'OrderedDict[str, float]', Any]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are doing.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
        task : OpenMLTask
            The task to run the model on.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        sample_no : int
            In case of learning curves, the index of the subsample (0-based; in case of no
            learning curve, always 0)
        add_local_measures : bool
            Determines whether to calculate a set of measures (i.e., predictive accuracy)
            locally,
            to later verify server behaviour.

        Returns
        -------
        arff_datacontent : List[List]
            Arff representation (list of lists) of the predictions that were
            generated by this fold (required to populate predictions.arff)
        arff_tracecontent :  List[List]
            Arff representation (list of lists) of the trace data that was generated by this
            fold
            (will be used to populate trace.arff, leave it empty if the model did not perform
            any
            hyperparameter optimization).
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        model : Any
            The model trained on this repeat,fold,subsample triple. Will be used to generate
            trace
            information later on (in ``obtain_arff_trace``).
        """

        def _prediction_to_probabilities(
                y: np.ndarray,
                model_classes: List,
        ) -> np.ndarray:
            """Transforms predicted probabilities to match with OpenML class indices.

            Parameters
            ----------
            y : np.ndarray
                Predicted probabilities (possibly omitting classes if they were not present in the
                training data).
            model_classes : list
                List of classes known_predicted by the model, ordered by their index.

            Returns
            -------
            np.ndarray
            """
            # y: list or numpy array of predictions
            # model_classes: sklearn classifier mapping from original array id to
            # prediction index id
            if not isinstance(model_classes, list):
                raise ValueError('please convert model classes to list prior to '
                                 'calling this fn')
            result = np.zeros((len(y), len(model_classes)), dtype=np.float32)
            for obs, prediction_idx in enumerate(y):
                array_idx = model_classes.index(prediction_idx)
                result[obs][array_idx] = 1.0
            return result

        # TODO: if possible, give a warning if model is already fitted (acceptable
        # in case of custom experimentation,
        # but not desirable if we want to upload to OpenML).

        model_copy = sklearn.base.clone(model, safe=True)
        # security check
        self._prevent_optimize_n_jobs(model_copy)
        # Runtime can be measured if the model is run sequentially
        can_measure_cputime = self._can_measure_cputime(model_copy)
        can_measure_wallclocktime = self._can_measure_wallclocktime(model_copy)

        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=rep_no, fold=fold_no, sample=sample_no)
        if isinstance(task, OpenMLSupervisedTask):
            x, y = task.get_X_and_y()
            train_x = x[train_indices]
            train_y = y[train_indices]
            test_x = x[test_indices]
            test_y = y[test_indices]
        elif isinstance(task, OpenMLClusteringTask):
            train_x = train_indices
            test_x = test_indices
        else:
            raise NotImplementedError(task.task_type)

        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'

        try:
            # for measuring runtime. Only available since Python 3.3
            modelfit_start_cputime = None
            modelfit_duration_cputime = None
            modelpredict_start_cputime = None

            modelfit_start_walltime = None
            modelfit_duration_walltime = None
            modelpredict_start_walltime = None
            if can_measure_cputime:
                modelfit_start_cputime = time.process_time()
            if can_measure_wallclocktime:
                modelfit_start_walltime = time.time()

            if isinstance(task, OpenMLSupervisedTask):
                model_copy.fit(train_x, train_y)
            elif isinstance(task, OpenMLClusteringTask):
                model_copy.fit(train_x)

            if can_measure_cputime:
                modelfit_duration_cputime = (time.process_time() - modelfit_start_cputime) * 1000
                user_defined_measures['usercpu_time_millis_training'] = modelfit_duration_cputime
            if can_measure_wallclocktime:
                modelfit_duration_walltime = (time.time() - modelfit_start_walltime) * 1000
                user_defined_measures['wall_clock_time_millis_training'] = modelfit_duration_walltime

        except AttributeError as e:
            # typically happens when training a regressor on classification task
            raise PyOpenMLError(str(e))

        # extract trace, if applicable
        arff_tracecontent = []  # type: List[List]
        if self.is_hpo_class(model_copy):
            arff_tracecontent.extend(self._extract_trace_data(model_copy, rep_no, fold_no))

        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):
            # search for model classes_ (might differ depending on modeltype)
            # first, pipelines are a special case (these don't have a classes_
            # object, but rather borrows it from the last step. We do this manually,
            # because of the BaseSearch check)
            if isinstance(model_copy, sklearn.pipeline.Pipeline):
                used_estimator = model_copy.steps[-1][-1]
            else:
                used_estimator = model_copy

            if self.is_hpo_class(used_estimator):
                model_classes = used_estimator.best_estimator_.classes_
            else:
                model_classes = used_estimator.classes_

        if can_measure_cputime:
            modelpredict_start_cputime = time.process_time()
        if can_measure_wallclocktime:
            modelpredict_start_walltime = time.time()

        # In supervised learning this returns the predictions for Y, in clustering
        # it returns the clusters
        pred_y = model_copy.predict(test_x)

        if can_measure_cputime:
            modelpredict_duration_cputime = (time.process_time() - modelpredict_start_cputime) * 1000
            user_defined_measures['usercpu_time_millis_testing'] = modelpredict_duration_cputime
            user_defined_measures['usercpu_time_millis'] = modelfit_duration_cputime + modelpredict_duration_cputime
        if can_measure_wallclocktime:
            modelpredict_duration_walltime = (time.time() - modelpredict_start_walltime) * 1000
            user_defined_measures['wall_clock_time_millis_testing'] = modelpredict_duration_walltime
            user_defined_measures['wall_clock_time_millis'] = modelfit_duration_walltime + \
                                                              modelpredict_duration_walltime

        # add client-side calculated metrics. These is used on the server as
        # consistency check, only useful for supervised tasks
        def _calculate_local_measure(sklearn_fn, openml_name):
            user_defined_measures[openml_name] = sklearn_fn(test_y, pred_y)

        # Task type specific outputs
        arff_datacontent = []

        if isinstance(task, (OpenMLClassificationTask, OpenMLLearningCurveTask)):

            try:
                proba_y = model_copy.predict_proba(test_x)
            except AttributeError:
                proba_y = _prediction_to_probabilities(pred_y, list(model_classes))

            if proba_y.shape[1] != len(task.class_labels):
                warnings.warn(
                    "Repeat %d Fold %d: estimator only predicted for %d/%d classes!"
                    % (rep_no, fold_no, proba_y.shape[1], len(task.class_labels))
                )

            if add_local_measures:
                _calculate_local_measure(sklearn.metrics.accuracy_score,
                                         'predictive_accuracy')

            for i in range(0, len(test_indices)):
                arff_line = self._prediction_to_row(
                    rep_no=rep_no,
                    fold_no=fold_no,
                    sample_no=sample_no,
                    row_id=test_indices[i],
                    correct_label=task.class_labels[test_y[i]],
                    predicted_label=pred_y[i],
                    predicted_probabilities=proba_y[i],
                    class_labels=task.class_labels,
                    model_classes_mapping=model_classes,
                )
                arff_datacontent.append(arff_line)

        elif isinstance(task, OpenMLRegressionTask):
            if add_local_measures:
                _calculate_local_measure(
                    sklearn.metrics.mean_absolute_error,
                    'mean_absolute_error',
                )

            for i in range(0, len(test_indices)):
                arff_line = [rep_no, fold_no, test_indices[i], pred_y[i], test_y[i]]
                arff_datacontent.append(arff_line)

        elif isinstance(task, OpenMLClusteringTask):
            for i in range(0, len(test_indices)):
                arff_line = [test_indices[i], pred_y[i]]  # row_id, cluster ID
                arff_datacontent.append(arff_line)

        else:
            raise TypeError(type(task))

        return arff_datacontent, arff_tracecontent, user_defined_measures, model_copy

    def _prediction_to_row(
        self,
        rep_no: int,
        fold_no: int,
        sample_no: int,
        row_id: int,
        correct_label: str,
        predicted_label: int,
        predicted_probabilities: np.ndarray,
        class_labels: List,
        model_classes_mapping: List,
    ) -> List:
        """Util function that turns probability estimates of a classifier for a
        given instance into the right arff format to upload to openml.

        Parameters
        ----------
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV,
            always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout,
            always 0)
        sample_no : int
            In case of learning curves, the index of the subsample (0-based;
            in case of no learning curve, always 0)
        row_id : int
            row id in the initial dataset
        correct_label : str
            original label of the instance
        predicted_label : str
            the label that was predicted
        predicted_probabilities : array (size=num_classes)
            probabilities per class
        class_labels : array (size=num_classes)
        model_classes_mapping : list
            A list of classes the model produced.
            Obtained by BaseEstimator.classes_

        Returns
        -------
        arff_line : list
            representation of the current prediction in OpenML format
        """
        if not isinstance(rep_no, (int, np.integer)):
            raise ValueError('rep_no should be int')
        if not isinstance(fold_no, (int, np.integer)):
            raise ValueError('fold_no should be int')
        if not isinstance(sample_no, (int, np.integer)):
            raise ValueError('sample_no should be int')
        if not isinstance(row_id, (int, np.integer)):
            raise ValueError('row_id should be int')
        if not len(predicted_probabilities) == len(model_classes_mapping):
            raise ValueError('len(predicted_probabilities) != len(class_labels)')

        arff_line = [rep_no, fold_no, sample_no, row_id]  # type: List[Any]
        for class_label_idx in range(len(class_labels)):
            if class_label_idx in model_classes_mapping:
                index = np.where(model_classes_mapping == class_label_idx)[0][0]
                # TODO: WHY IS THIS 2D???
                arff_line.append(predicted_probabilities[index])
            else:
                arff_line.append(0.0)

        arff_line.append(class_labels[predicted_label])
        arff_line.append(correct_label)
        return arff_line

    def _extract_trace_data(self, model, rep_no, fold_no):
        arff_tracecontent = []
        for itt_no in range(0, len(model.cv_results_['mean_test_score'])):
            # we use the string values for True and False, as it is defined in
            # this way by the OpenML server
            selected = 'false'
            if itt_no == model.best_index_:
                selected = 'true'
            test_score = model.cv_results_['mean_test_score'][itt_no]
            arff_line = [rep_no, fold_no, itt_no, test_score, selected]
            for key in model.cv_results_:
                if key.startswith('param_'):
                    value = model.cv_results_[key][itt_no]
                    if value is not np.ma.masked:
                        serialized_value = json.dumps(value)
                    else:
                        serialized_value = np.nan
                    arff_line.append(serialized_value)
            arff_tracecontent.append(arff_line)
        return arff_tracecontent

    def obtain_parameter_values(
        self,
        flow: 'OpenMLFlow',
        model: Any = None,
    ) -> List[Dict[str, Any]]:
        """Extracts all parameter settings required for the flow from the model.

        If no explicit model is provided, the parameters will be extracted from `flow.model`
        instead.

        Parameters
        ----------
        flow : OpenMLFlow
            OpenMLFlow object (containing flow ids, i.e., it has to be downloaded from the server)

        model: Any, optional (default=None)
            The model from which to obtain the parameter values. Must match the flow signature.
            If None, use the model specified in ``OpenMLFlow.model``.

        Returns
        -------
        list
            A list of dicts, where each dict has the following entries:
            - ``oml:name`` : str: The OpenML parameter name
            - ``oml:value`` : mixed: A representation of the parameter value
            - ``oml:component`` : int: flow id to which the parameter belongs
        """
        openml.flows.functions._check_flow_for_server_id(flow)

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(_flow, _flow_dict, component_model,
                               _main_call=False, main_id=None):
            def is_subcomponent_specification(values):
                # checks whether the current value can be a specification of
                # subcomponents, as for example the value for steps parameter
                # (in Pipeline) or transformers parameter (in
                # ColumnTransformer). These are always lists/tuples of lists/
                # tuples, size bigger than 2 and an OpenMLFlow item involved.
                if not isinstance(values, (tuple, list)):
                    return False
                for item in values:
                    if not isinstance(item, (tuple, list)):
                        return False
                    if len(item) < 2:
                        return False
                    if not isinstance(item[1], openml.flows.OpenMLFlow):
                        return False
                return True

            # _flow is openml flow object, _param dict maps from flow name to flow
            # id for the main call, the param dict can be overridden (useful for
            # unit tests / sentinels) this way, for flows without subflows we do
            # not have to rely on _flow_dict
            exp_parameters = set(_flow.parameters)
            exp_components = set(_flow.components)
            model_parameters = set([mp for mp in component_model.get_params()
                                    if '__' not in mp])
            if len((exp_parameters | exp_components) ^ model_parameters) != 0:
                flow_params = sorted(exp_parameters | exp_components)
                model_params = sorted(model_parameters)
                raise ValueError('Parameters of the model do not match the '
                                 'parameters expected by the '
                                 'flow:\nexpected flow parameters: '
                                 '%s\nmodel parameters: %s' % (flow_params,
                                                               model_params))

            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current['oml:name'] = _param_name

                current_param_values = self.model_to_flow(component_model.get_params()[_param_name])

                # Try to filter out components (a.k.a. subflows) which are
                # handled further down in the code (by recursively calling
                # this function)!
                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                if is_subcomponent_specification(current_param_values):
                    # complex parameter value, with subcomponents
                    parsed_values = list()
                    for subcomponent in current_param_values:
                        # scikit-learn stores usually tuples in the form
                        # (name (str), subcomponent (mixed), argument
                        # (mixed)). OpenML replaces the subcomponent by an
                        # OpenMLFlow object.
                        if len(subcomponent) < 2 or len(subcomponent) > 3:
                            raise ValueError('Component reference should be '
                                             'size {2,3}. ')

                        subcomponent_identifier = subcomponent[0]
                        subcomponent_flow = subcomponent[1]
                        if not isinstance(subcomponent_identifier, str):
                            raise TypeError('Subcomponent identifier should be '
                                            'string')
                        if not isinstance(subcomponent_flow,
                                          openml.flows.OpenMLFlow):
                            raise TypeError('Subcomponent flow should be string')

                        current = {
                            "oml-python:serialized_object": "component_reference",
                            "value": {
                                "key": subcomponent_identifier,
                                "step_name": subcomponent_identifier
                            }
                        }
                        if len(subcomponent) == 3:
                            if not isinstance(subcomponent[2], list):
                                raise TypeError('Subcomponent argument should be'
                                                'list')
                            current['value']['argument_1'] = subcomponent[2]
                        parsed_values.append(current)
                    parsed_values = json.dumps(parsed_values)
                else:
                    # vanilla parameter value
                    parsed_values = json.dumps(current_param_values)

                _current['oml:value'] = parsed_values
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = component_model.get_params()[_identifier]
                _params.extend(extract_parameters(_flow.components[_identifier],
                                                  _flow_dict, subcomponent_model))
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        parameters = extract_parameters(flow, flow_dict, model, True, flow.flow_id)

        return parameters

    def _openml_param_name_to_sklearn(
        self,
        openml_parameter: openml.setups.OpenMLParameter,
        flow: OpenMLFlow,
    ) -> str:
        """
        Converts the name of an OpenMLParameter into the sklean name, given a flow.

        Parameters
        ----------
        openml_parameter: OpenMLParameter
            The parameter under consideration

        flow: OpenMLFlow
            The flow that provides context.

        Returns
        -------
        sklearn_parameter_name: str
            The name the parameter will have once used in scikit-learn
        """
        if not isinstance(openml_parameter, openml.setups.OpenMLParameter):
            raise ValueError('openml_parameter should be an instance of OpenMLParameter')
        if not isinstance(flow, OpenMLFlow):
            raise ValueError('flow should be an instance of OpenMLFlow')

        flow_structure = flow.get_structure('name')
        if openml_parameter.flow_name not in flow_structure:
            raise ValueError('Obtained OpenMLParameter and OpenMLFlow do not correspond. ')
        name = openml_parameter.flow_name  # for PEP8
        return '__'.join(flow_structure[name] + [openml_parameter.parameter_name])

    ################################################################################################
    # Methods for hyperparameter optimization

    def is_hpo_class(self, model: Any) -> bool:
        """Check whether the model performs hyperparameter optimization.

        Used to check whether an optimization trace can be extracted from the model after
        running it.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        return isinstance(model, sklearn.model_selection._search.BaseSearchCV)

    def instantiate_model_from_hpo_class(
        self,
        model: Any,
        trace_iteration: OpenMLTraceIteration,
    ) -> Any:
        """Instantiate a ``base_estimator`` which can be searched over by the hyperparameter
        optimization model.

        Parameters
        ----------
        model : Any
            A hyperparameter optimization model which defines the model to be instantiated.
        trace_iteration : OpenMLTraceIteration
            Describing the hyperparameter settings to instantiate.

        Returns
        -------
        Any
        """
        if not self.is_hpo_class(model):
            raise AssertionError(
                'Flow model %s is not an instance of sklearn.model_selection._search.BaseSearchCV'
                % model
            )
        base_estimator = model.estimator
        base_estimator.set_params(**trace_iteration.get_parameters())
        return base_estimator

    def obtain_arff_trace(
        self,
        model: Any,
        trace_content: List,
    ) -> 'OpenMLRunTrace':
        """Create arff trace object from a fitted model and the trace content obtained by
        repeatedly calling ``run_model_on_task``.

        Parameters
        ----------
        model : Any
            A fitted hyperparameter optimization model.

        trace_content : List[List]
            Trace content obtained by ``openml.runs.run_flow_on_task``.

        Returns
        -------
        OpenMLRunTrace
        """
        if not self.is_hpo_class(model):
            raise AssertionError(
                'Flow model %s is not an instance of sklearn.model_selection._search.BaseSearchCV'
                % model
            )
        if not hasattr(model, 'cv_results_'):
            raise ValueError('model should contain `cv_results_`')

        # attributes that will be in trace arff, regardless of the model
        trace_attributes = [('repeat', 'NUMERIC'),
                            ('fold', 'NUMERIC'),
                            ('iteration', 'NUMERIC'),
                            ('evaluation', 'NUMERIC'),
                            ('selected', ['true', 'false'])]

        # model dependent attributes for trace arff
        for key in model.cv_results_:
            if key.startswith('param_'):
                # supported types should include all types, including bool,
                # int float
                supported_basic_types = (bool, int, float, str)
                for param_value in model.cv_results_[key]:
                    if isinstance(param_value, supported_basic_types) or \
                            param_value is None or param_value is np.ma.masked:
                        # basic string values
                        type = 'STRING'
                    elif isinstance(param_value, list) and \
                            all(isinstance(i, int) for i in param_value):
                        # list of integers
                        type = 'STRING'
                    else:
                        raise TypeError('Unsupported param type in param grid: %s' % key)

                # renamed the attribute param to parameter, as this is a required
                # OpenML convention - this also guards against name collisions
                # with the required trace attributes
                attribute = (PREFIX + key[6:], type)
                trace_attributes.append(attribute)

        return OpenMLRunTrace.generate(
            trace_attributes,
            trace_content,
        )


register_extension(SklearnExtension)
