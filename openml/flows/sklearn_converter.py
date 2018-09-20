"""Convert scikit-learn estimators into an OpenMLFlows and vice versa."""

from collections import OrderedDict
import importlib
import json
import json.decoder
import re
import six
import warnings
import sys
import inspect

import numpy as np
import scipy.stats.distributions
import sklearn.base
import sklearn.model_selection
# Necessary to have signature available in python 2.7
from sklearn.utils.fixes import signature

import openml
from openml.flows import OpenMLFlow
from openml.exceptions import PyOpenMLError

from .abstract_converter import AbstractConverter

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError


class SKLearnConverter(AbstractConverter):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def _serialize_function(o):
        name = o.__module__ + '.' + o.__name__
        ret = OrderedDict()
        ret['oml-python:serialized_object'] = 'function'
        ret['value'] = name
        return ret

    @staticmethod
    def _deserialize_function(name):
        module_name = name.rsplit('.', 1)
        try:
            function_handle = getattr(importlib.import_module(module_name[0]),
                                      module_name[1])
        except Exception as e:
            warnings.warn('Cannot load function %s due to %s.' % (name, e))
            return None
        return function_handle

    @staticmethod
    def _flow_to_sklearn(o, components=None, initialize_with_defaults=False):
        """Initializes a sklearn model based on a flow.

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

        # First, we need to check whether the presented object is a json string.
        # JSON strings are used to encoder parameter values. By passing around
        # json strings for parameters, we make sure that we can flow_to_sklearn
        # the parameter values to the correct type.

        if isinstance(o, six.string_types):
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
                    rval = SKLearnConverter._deserialize_type(value)
                elif serialized_type == 'rv_frozen':
                    rval = SKLearnConverter._deserialize_rv_frozen(value)
                elif serialized_type == 'function':
                    rval = SKLearnConverter._deserialize_function(value)
                elif serialized_type == 'component_reference':
                    value = SKLearnConverter._flow_to_sklearn(value)
                    step_name = value['step_name']
                    key = value['key']
                    component =  SKLearnConverter._flow_to_sklearn(components[key], initialize_with_defaults=initialize_with_defaults)
                    # The component is now added to where it should be used
                    # later. It should not be passed to the constructor of the
                    # main flow object.
                    del components[key]
                    if step_name is None:
                        rval = component
                    else:
                        rval = (step_name, component)
                elif serialized_type == 'cv_object':
                    rval =  SKLearnConverter._deserialize_cross_validator(value)
                else:
                    raise ValueError('Cannot flow_to_sklearn %s' % serialized_type)

            else:
                rval = OrderedDict((SKLearnConverter._flow_to_sklearn(key, components, initialize_with_defaults),
                                    SKLearnConverter._flow_to_sklearn(value, components, initialize_with_defaults))
                                   for key, value in sorted(o.items()))
        elif isinstance(o, (list, tuple)):
            rval = [SKLearnConverter._flow_to_sklearn(element, components, initialize_with_defaults) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, six.string_types)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            rval = SKLearnConverter._deserialize_model(o, initialize_with_defaults)
        else:
            raise TypeError(o)

        return rval

    @staticmethod
    def _sklearn_to_flow(o, parent_model=None):
        """
        """
        # TODO: assert that only on first recursion lvl `parent_model` can be None

        if SKLearnConverter._is_estimator(o):
            # is the main model or a submodel
            rval = SKLearnConverter(o).to_flow()
        elif isinstance(o, (list, tuple)):
            # TODO: explain what type of parameter is here
            rval = [SKLearnConverter._sklearn_to_flow(element, parent_model) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, six.string_types)) or o is None:
            # base parameter values
            rval = o
        elif isinstance(o, dict):
            # TODO: explain what type of parameter is here
            if not isinstance(o, OrderedDict):
                o = OrderedDict([(key, value) for key, value in sorted(o.items())])

            rval = OrderedDict()
            for key, value in o.items():
                if not isinstance(key, six.string_types):
                    raise TypeError('Can only use string as keys, you passed '
                                    'type %s for value %s.' %
                                    (type(key), str(key)))
                key = SKLearnConverter._sklearn_to_flow(key, parent_model)
                value = SKLearnConverter._sklearn_to_flow(value, parent_model)
                rval[key] = value
            rval = rval
        elif isinstance(o, type):
            # TODO: explain what type of parameter is here
            rval = SKLearnConverter._serialize_type(o)
        elif isinstance(o, scipy.stats.distributions.rv_frozen):
            rval = SKLearnConverter._serialize_rv_frozen(o)
        # This only works for user-defined functions (and not even partial).
        # I think this is exactly what we want here as there shouldn't be any
        # built-in or functool.partials in a pipeline
        elif inspect.isfunction(o):
            # TODO: explain what type of parameter is here
            rval = SKLearnConverter._serialize_function(o)
        elif SKLearnConverter._is_cross_validator(o):
            # TODO: explain what type of parameter is here
            rval = SKLearnConverter._serialize_cross_validator(o)
        else:
            raise TypeError(o, type(o))
        return rval

    @staticmethod
    def _is_estimator(o):
        return (hasattr(o, 'fit') and hasattr(o, 'get_params') and
                hasattr(o, 'set_params'))

    @staticmethod
    def _is_cross_validator(o):
        return isinstance(o, sklearn.model_selection.BaseCrossValidator)

    @staticmethod
    def from_flow(flow, components=None, initialize_with_defaults=False):
        return SKLearnConverter._flow_to_sklearn(
            flow, components=components, initialize_with_defaults=initialize_with_defaults)

    def to_flow(self):
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

        # Create a flow name, which contains all components in brackets, for
        # example RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
        class_name = self._model.__module__ + "." + self._model.__class__.__name__

        # will be part of the name (in brackets)
        sub_components_names = ""
        for key in self._sub_components:
            if key in self._sub_components_explicit:
                sub_components_names += "," + key + "=" + self._sub_components[key].name
            else:
                sub_components_names += "," + self._sub_components[key].name

        if sub_components_names:
            # slice operation on string in order to get rid of leading comma
            name = '%s(%s)' % (class_name, sub_components_names[1:])
        else:
            name = class_name

        dependencies = [self.format_external_version('sklearn', sklearn.__version__),
                        'numpy>=1.6.1', 'scipy>=0.9']
        dependencies = '\n'.join(dependencies)

        return OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created scikit-learn flow.',
                          model=self._model,
                          components=self._sub_components,
                          parameters=self._parameters,
                          parameters_meta_info=self._parameters_meta_info,
                          external_version=self.external_version,
                          tags=['openml-python', 'sklearn', 'scikit-learn',
                                'python',
                                self.format_external_version(
                                    'sklearn', sklearn.__version__).replace('==', '_'),
                                # TODO: add more tags based on the scikit-learn
                                # module a flow is in? For example automatically
                                # annotate a class of sklearn.svm.SVC() with the
                                # tag svm?
                                ],
                          language='English',
                          # TODO fill in dependencies!
                          dependencies=dependencies)

    def extract_information_from_model(self):
        # This function contains four "global" states and is quite long and
        # complicated. If it gets to complicated to ensure it's correctness,
        # it would be best to make it a class with the four "global" states being
        # the class attributes and the if/elif/else in the for-loop calls to
        # separate class methods
        model_parameters = self._model.get_params(deep=False)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._sklearn_to_flow(v, self._model)

            if (isinstance(rval, (list, tuple)) and len(rval) > 0 and
                    isinstance(rval[0], (list, tuple)) and
                    [type(rval[0]) == type(rval[i]) for i in range(len(rval))]):

                self._extract_sklearn_model_information(rval, k)
            elif isinstance(rval, OpenMLFlow):
                self._extract_openml_flow_information(rval, k)
            else:
                # a regular hyperparameter
                if not (hasattr(rval, '__len__') and len(rval) == 0):
                    rval = json.dumps(rval)
                    self._parameters[k] = rval
                else:
                    self._parameters[k] = None

            self._parameters_meta_info[k] = OrderedDict((('description', None),
                                                   ('data_type', None)))

    def _extract_sklearn_model_information(self, rval, parameter_name):
        # Steps in a pipeline or feature union, or base classifiers in voting classifier
        parameter_value = list()
        reserved_keywords = set(self._model.get_params(deep=False).keys())

        for sub_component_tuple in rval:
            identifier, sub_component = sub_component_tuple
            sub_component_type = type(sub_component_tuple)

            if identifier in reserved_keywords:
                parent_model_name = self._model.__module__ + "." + \
                                    self._model.__class__.__name__
                raise PyOpenMLError('Found element shadowing official ' + \
                                    'parameter for %s: %s' % (parent_model_name, identifier))

            if sub_component is None:
                # In a FeatureUnion it is legal to have a None step

                pv = [identifier, None]
                if sub_component_type is tuple:
                    pv = tuple(pv)
                parameter_value.append(pv)

            else:
                # Add the component to the list of components, add a
                # component reference as a placeholder to the list of
                # parameters, which will be replaced by the real component
                # when deserializing the parameter
                self._sub_components_explicit.add(identifier)
                self._sub_components[identifier] = sub_component
                component_reference = OrderedDict()
                component_reference[
                    'oml-python:serialized_object'] = 'component_reference'
                cr_value = OrderedDict()
                cr_value['key'] = identifier
                cr_value['step_name'] = identifier
                component_reference['value'] = cr_value
                parameter_value.append(component_reference)

        if isinstance(rval, tuple):
            parameter_value = tuple(parameter_value)

        # Here (and in the elif and else branch below) are the only
        # places where we encode a value as json to make sure that all
        # parameter values still have the same type after
        # deserialization
        self._parameters[parameter_name] = json.dumps(parameter_value)


    def _extract_openml_flow_information(self, rval, parameter_name):
        """

        """
        # A subcomponent, for example the base model in
        # AdaBoostClassifier
        self._sub_components[parameter_name] = rval
        self._sub_components_explicit.add(parameter_name)
        component_reference = OrderedDict()
        component_reference[
            'oml-python:serialized_object'] = 'component_reference'
        cr_value = OrderedDict()
        cr_value['key'] = parameter_name
        cr_value['step_name'] = None
        component_reference['value'] = cr_value
        component_reference = self._sklearn_to_flow(component_reference, self._model)
        self._parameters[parameter_name] = json.dumps(component_reference)

    @staticmethod
    def _serialize_cross_validator(o):
        ret = OrderedDict()
        parameters = OrderedDict()

        # XXX this is copied from sklearn.model_selection._split
        cls = o.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        # Ignore varargs, kw and default values and pop self
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        if init is object.__init__:
            args = []
        else:
            args = sorted([p.name for p in init_signature.parameters.values()
                           if p.name != 'self' and p.kind != p.VAR_KEYWORD])

        for key in args:
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(o, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            if not (hasattr(value, '__len__') and len(value) == 0):
                value = json.dumps(value)
                parameters[key] = value
            else:
                parameters[key] = None

        ret['oml-python:serialized_object'] = 'cv_object'
        name = o.__module__ + "." + o.__class__.__name__
        value = OrderedDict([['name', name], ['parameters', parameters]])
        ret['value'] = value

        return ret

    @staticmethod
    def _serialize_type(o):
        mapping = {float: 'float',
                   np.float: 'np.float',
                   np.float32: 'np.float32',
                   np.float64: 'np.float64',
                   int: 'int',
                   np.int: 'np.int',
                   np.int32: 'np.int32',
                   np.int64: 'np.int64'}
        ret = OrderedDict()
        ret['oml-python:serialized_object'] = 'type'
        ret['value'] = mapping[o]
        return ret

    @staticmethod
    def _deserialize_type(o):
        mapping = {'float': float,
                   'np.float': np.float,
                   'np.float32': np.float32,
                   'np.float64': np.float64,
                   'int': int,
                   'np.int': np.int,
                   'np.int32': np.int32,
                   'np.int64': np.int64}
        return mapping[o]

    @staticmethod
    def _serialize_rv_frozen(o):
        args = o.args
        kwds = o.kwds
        a = o.a
        b = o.b
        dist = o.dist.__class__.__module__ + '.' + o.dist.__class__.__name__
        ret = OrderedDict()
        ret['oml-python:serialized_object'] = 'rv_frozen'
        ret['value'] = OrderedDict((('dist', dist), ('a', a), ('b', b),
                                    ('args', args), ('kwds', kwds)))
        return ret

    @staticmethod
    def _deserialize_rv_frozen(o):
        args = o['args']
        kwds = o['kwds']
        a = o['a']
        b = o['b']
        dist_name = o['dist']

        module_name = dist_name.rsplit('.', 1)
        try:
            rv_class = getattr(importlib.import_module(module_name[0]),
                               module_name[1])
        except:
            warnings.warn('Cannot create model %s for flow.' % dist_name)
            return None

        dist = scipy.stats.distributions.rv_frozen(rv_class(), *args, **kwds)
        dist.a = a
        dist.b = b

        return dist

    @staticmethod
    def _deserialize_cross_validator(value):
        model_name = value['name']
        parameters = value['parameters']

        module_name = model_name.rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
        for parameter in parameters:
            parameters[parameter] = SKLearnConverter._flow_to_sklearn(parameters[parameter])
        return model_class(**parameters)


#    def run_on_task(self, task):



def _check_n_jobs(model):
    """
    Returns True if the parameter settings of model are chosen s.t. the model
    will run on a single core (in that case, openml-python can measure runtimes)
    """
    def check(param_grid, restricted_parameter_name, legal_values):
        if isinstance(param_grid, dict):
            for param, value in param_grid.items():
                # n_jobs is scikitlearn parameter for paralizing jobs
                if param.split('__')[-1] == restricted_parameter_name:
                    # 0 = illegal value (?), 1 / None = use one core,
                    # n = use n cores,
                    # -1 = use all available cores -> this makes it hard to
                    # measure runtime in a fair way
                    if legal_values is None or value not in legal_values:
                        return False
            return True
        elif isinstance(param_grid, list):
            for sub_grid in param_grid:
                if not check(sub_grid, restricted_parameter_name, legal_values):
                    return False
            return True

    if not (isinstance(model, sklearn.base.BaseEstimator) or
            isinstance(model, sklearn.model_selection._search.BaseSearchCV)):
        raise ValueError('model should be BaseEstimator or BaseSearchCV')

    # make sure that n_jobs is not in the parameter grid of optimization
    # procedure
    if isinstance(model, sklearn.model_selection._search.BaseSearchCV):
        if isinstance(model, sklearn.model_selection.GridSearchCV):
            param_distributions = model.param_grid
        elif isinstance(model, sklearn.model_selection.RandomizedSearchCV):
            param_distributions = model.param_distributions
        else:
            if hasattr(model, 'param_distributions'):
                param_distributions = model.param_distributions
            else:
                raise AttributeError('Using subclass BaseSearchCV other than {GridSearchCV, RandomizedSearchCV}. Could not find attribute param_distributions. ')
            print('Warning! Using subclass BaseSearchCV other than ' \
                  '{GridSearchCV, RandomizedSearchCV}. Should implement param check. ')

        if not check(param_distributions, 'n_jobs', None):
            raise PyOpenMLError('openml-python should not be used to '
                                'optimize the n_jobs parameter.')

    # check the parameters for n_jobs
    return check(model.get_params(), 'n_jobs', [1, None])
