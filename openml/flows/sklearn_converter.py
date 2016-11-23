from collections import OrderedDict, defaultdict
import importlib
import inspect
import json
import json.decoder
import six
import warnings
import sys

import numpy as np
import scipy.stats.distributions
import sklearn.base
import sklearn.model_selection
# Necessary to have signature available in python 2.7
from sklearn.utils.fixes import signature

from .flow import OpenMLFlow


if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError


"""Convert scikit-learn estimators into an OpenMLFlows and vice versa."""


def sklearn_to_flow(o):

    if _is_estimator(o):
        rval = _serialize_model(o)
    elif isinstance(o, (list, tuple)):
        rval = [sklearn_to_flow(element) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif o is None:
        rval = None
    elif isinstance(o, six.string_types):
        rval = o
    elif isinstance(o, (bool, int, float)):
        rval = o
    elif isinstance(o, dict):
        rval = {}
        for key, value in o.items():
            if not isinstance(key, six.string_types):
                raise TypeError('Can only use string as keys, you passed '
                                'type %s for value %s.' %
                                (type(key), str(key)))
            key = sklearn_to_flow(key)
            value = sklearn_to_flow(value)
            rval[key] = value
        rval = rval
    elif isinstance(o, type):
        rval = serialize_type(o)
    elif isinstance(o, scipy.stats.distributions.rv_frozen):
        rval = serialize_rv_frozen(o)
    # This only works for user-defined functions (and not even partial).
    # I think this is exactly we want here as there shouldn't be any
    # built-in or functool.partials in a pipeline
    elif inspect.isfunction(o):
        rval = serialize_function(o)
    elif _is_cross_validator(o):
        rval = _serialize_cross_validator(o)
    else:
        raise TypeError(o, type(o))

    return rval


def _is_estimator(o):
    return (hasattr(o, 'fit') and hasattr(o, 'get_params') and
            hasattr(o, 'set_params'))


def _is_cross_validator(o):
    return isinstance(o, sklearn.model_selection.BaseCrossValidator)


def flow_to_sklearn(o, **kwargs):
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
        if 'oml:name' in o and 'oml:description' in o:
            # TODO check if this code is actually called
            rval = _deserialize_model(o, **kwargs)

        # Check if the dict encodes a 'special' object, which could not
        # easily converted into a string, but rather the information to
        # re-create the object were stored in a dictionary.
        elif 'oml:serialized_object' in o:
            serialized_type = o['oml:serialized_object']
            value = o['value']
            if serialized_type == 'type':
                rval = deserialize_type(value, **kwargs)
            elif serialized_type == 'rv_frozen':
                rval = deserialize_rv_frozen(value, **kwargs)
            elif serialized_type == 'function':
                rval = deserialize_function(value, **kwargs)
            elif serialized_type == 'component_reference':
                value = flow_to_sklearn(value)
                step_name = value['step_name']
                key = value['key']
                component = flow_to_sklearn(kwargs['components'][key])
                if step_name is None:
                    rval = component
                else:
                    rval = (step_name, component)

            else:
                raise ValueError('Cannot flow_to_sklearn %s' % serialized_type)

        else:
            # Regular dictionary
            rval = {flow_to_sklearn(key, **kwargs): flow_to_sklearn(value, **kwargs)
                    for key, value in o.items()}
    elif isinstance(o, (list, tuple)):
        rval = [flow_to_sklearn(element, **kwargs) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif isinstance(o, (bool, int, float)):
        rval = o
    elif isinstance(o, six.string_types):
        rval = o
    elif o is None:
        rval = None
    elif isinstance(o, OpenMLFlow):
        rval = _deserialize_model(o, **kwargs)
    else:
        raise TypeError(o)
    assert o is None or rval is not None

    return rval


def _serialize_model(model):
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
    sub_components = OrderedDict()
    parameters = OrderedDict()
    parameters_meta_info = OrderedDict()

    model_parameters = model.get_params(deep=False)

    for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
        rval = sklearn_to_flow(v)

        if (isinstance(rval, (list, tuple)) and len(rval) > 0 and
                isinstance(rval[0], (list, tuple)) and
                [type(rval[0]) == type(rval[i]) for i in range(len(rval))]):

            # Steps in a pipeline or feature union
            parameter_value = list()
            for sub_component_tuple in rval:
                identifier, sub_component = sub_component_tuple
                sub_component_type = type(sub_component_tuple)

                # Use only the name of the module (and not all submodules
                # in the brackets) as the identifier
                pos = identifier.find('(')
                if pos >= 0:
                    identifier = identifier[:pos]

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
                    # when deserealizing the parameter
                    sub_component_identifier = k + '__' + identifier
                    sub_components[sub_component_identifier] = sub_component
                    component_reference = OrderedDict()
                    component_reference['oml:serialized_object'] = 'component_reference'
                    component_reference['value'] = OrderedDict(
                        key=sub_component_identifier, step_name=identifier)
                    parameter_value.append(component_reference)

            if isinstance(rval, tuple):
                parameter_value = tuple(parameter_value)

            # Here (and in the elif and else branch below) are the only
            # places where we encode a value as json to make sure that all
            # parameter values still have the same type after
            # deserialization
            parameter_value = json.dumps(parameter_value)
            parameters[k] = parameter_value

        elif isinstance(rval, OpenMLFlow):

            # A subcomponent, for example the base model in
            # AdaBoostClassifier
            sub_components[k] = rval
            component_reference = OrderedDict()
            component_reference['oml:serialized_object'] = 'component_reference'
            component_reference['value'] = OrderedDict(key=k, step_name=None)
            component_reference = sklearn_to_flow(component_reference)
            parameters[k] = json.dumps(component_reference)

        else:

            # a regular hyperparameter
            if not (hasattr(rval, '__len__') and len(rval) == 0):
                rval = json.dumps(rval)
                parameters[k] = rval
            else:
                parameters[k] = None

        parameters_meta_info[k] = OrderedDict((('description', None),
                                               ('data_type', None)))

    # Create a flow name, which contains all components in brackets, for
    # example RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
    # TODO the name above is apparently wrong, I need to test and check this
    name = model.__module__ + "." + model.__class__.__name__
    sub_components_names = ",".join(
        [sub_components[key].name for key in sub_components])
    if sub_components_names:
        name = '%s(%s)' % (name, sub_components_names)

    external_version = _get_external_version_info()
    flow = OpenMLFlow(name=name,
                      description='Automatically created sub-component.',
                      model=model,
                      components=sub_components,
                      parameters=parameters,
                      parameters_meta_info=parameters_meta_info,
                      external_version=external_version,
                      tags=[],
                      language='English',
                      # TODO fill in dependencies!
                      dependencies=None)

    return flow


def _deserialize_model(flow, **kwargs):

    model_name = flow.name
    # Remove everything after the first bracket, it is not necessary for
    # creating the current flow
    pos = model_name.find('(')
    if pos >= 0:
        model_name = model_name[:pos]

    parameters = flow.parameters
    components = flow.components
    component_dict = defaultdict(dict)
    parameter_dict = {}

    for name in components:
        if '__' in name:
            parameter_name, step = name.split('__')
            value = components[name]
            rval = flow_to_sklearn(value)
            component_dict[parameter_name][step] = rval
        else:
            value = components[name]
            rval = flow_to_sklearn(value)
            parameter_dict[name] = rval

    for name in parameters:
        value = parameters.get(name)
        rval = flow_to_sklearn(value, components=components)

        # Replace the component placeholder by the actual flow
        if isinstance(rval, dict) and 'oml:serialized_object' in rval:
            parameter_name, step = rval['value'].split('__')
            rval = component_dict[parameter_name][step]
        parameter_dict[name] = rval

    module_name = model_name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % model_name)
        return None

    return model_class(**parameter_dict)


def serialize_type(o):
    mapping = {float: 'float',
               np.float: 'np.float',
               np.float32: 'np.float32',
               np.float64: 'np.float64',
               int: 'int',
               np.int: 'np.int',
               np.int32: 'np.int32',
               np.int64: 'np.int64'}
    ret = OrderedDict()
    ret['oml:serialized_object'] = 'type'
    ret['value'] = mapping[o]
    return ret


def deserialize_type(o, **kwargs):
    mapping = {'float': float,
               'np.float': np.float,
               'np.float32': np.float32,
               'np.float64': np.float64,
               'int': int,
               'np.int': np.int,
               'np.int32': np.int32,
               'np.int64': np.int64}
    return mapping[o]


def serialize_rv_frozen(o):
    args = o.args
    kwds = o.kwds
    a = o.a
    b = o.b
    dist = o.dist.__class__.__module__ + '.' + o.dist.__class__.__name__
    ret = OrderedDict()
    ret['oml:serialized_object'] = 'rv_frozen'
    ret['value'] = OrderedDict(dist=dist, a=a, b=b, args=args, kwds=kwds)
    return ret

def deserialize_rv_frozen(o, **kwargs):
    args = o['args']
    kwds = o['kwds']
    a = o['a']
    b = o['b']
    dist_name = o['dist']

    module_name = dist_name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % dist_name)
        return None

    dist = scipy.stats.distributions.rv_frozen(model_class(), *args, **kwds)
    dist.a = a
    dist.b = b

    return dist


def serialize_function(o):
    name = o.__module__ + '.' + o.__name__
    ret = OrderedDict()
    ret['oml:serialized_object'] = 'function'
    ret['value'] = name
    return ret


def deserialize_function(name, **kwargs):
    module_name = name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except Exception as e:
        warnings.warn('Cannot load function %s due to %s.' % (name, e))
        return None
    return model_class

# This produces a flow, thus it does not need a deserialize function as
# the function _deserialize_model is used for that. It cannot be fed
# to serialize_model() because cross-validators do not have get_params().
def _serialize_cross_validator( o):
    parameters = OrderedDict()
    parameters_meta_info = OrderedDict()

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
        parameters_meta_info[key] = OrderedDict((('description', None),
                                                 ('data_type', None)))

    # Create a flow
    name = o.__module__ + "." + o.__class__.__name__

    external_version = _get_external_version_info()
    flow = OpenMLFlow(name=name,
                      description='Automatically created sub-component.',
                      model=o,
                      parameters=parameters,
                      parameters_meta_info=parameters_meta_info,
                      external_version=external_version,
                      components=OrderedDict(),
                      tags=[],
                      language='English',
                      # TODO fill in dependencies!
                      dependencies=None)

    return flow


def _get_external_version_info():
    return 'sklearn_' + sklearn.__version__
