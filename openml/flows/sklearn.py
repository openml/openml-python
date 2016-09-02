from collections import OrderedDict
import importlib
import inspect
import six
import warnings

import numpy as np
import scipy.stats.distributions
import sklearn.base
import sklearn.model_selection
# Necessary to have signature available in python 2.7
from sklearn.utils.fixes import signature

from .flow import OpenMLFlow
from ..exceptions import OpenMLRestrictionViolated

MAXIMAL_FLOW_LENGTH = 1024


def serialize_object(o):
    if _is_estimator(o) or _is_transformer(o):
        rval = serialize_model(o)
    elif isinstance(o, (list, tuple)):
        rval = [serialize_object(element) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif o is None:
        rval = None
    elif isinstance(o, six.string_types):
        rval = o
    elif isinstance(o, int):
        rval = o
    elif isinstance(o, float):
        rval = o
    elif isinstance(o, dict):
        rval = {}
        for key, value in o.items():
            if not isinstance(key, six.string_types):
                raise TypeError('Can only use string as keys, you passed '
                                'type %s for value %s.' % (type(key), str(key)))
            key = serialize_object(key)
            value = serialize_object(value)
            rval[key] = value
    elif isinstance(o, type):
        rval = serialize_type(o)
    elif isinstance(o, scipy.stats.distributions.rv_frozen):
        rval = serialize_rv_frozen(o)
    # This only works for user-defined functions (and not even partial).
    # I think this exactly we want here as there shouldn't be any built-in or
    # functool.partials in a pipeline
    elif inspect.isfunction(o):
        rval = serialize_function(o)
    elif _is_cross_validator(o):
        rval = serialize_cross_validator(o)
    else:
        raise TypeError(o)

    assert o is None or rval is not None

    return rval


# TODO maybe remove those functions and put the check to the long
# if-constructs above?
def _is_estimator(o):
    # TODO @amueller should one rather check whether this is a subclass of
    # BaseEstimator?
    #return (hasattr(o, 'fit') and hasattr(o, 'predict') and
    #        hasattr(o, 'get_params') and hasattr(o, 'set_params'))
    return isinstance(o, sklearn.base.BaseEstimator)


def _is_transformer(o):
    # TODO @amueller should one rather check whether this is a subclass of
    # BaseTransformer?
    return (hasattr(o, 'fit') and hasattr(o, 'transform') and
            hasattr(o, 'get_params') and hasattr(o, 'set_params'))


def _is_cross_validator(o):
    return isinstance(o, sklearn.model_selection.BaseCrossValidator)


def deserialize_object(o):
    if isinstance(o, dict):
        if 'oml:name' in o and 'oml:description' in o:
            rval = deserialize_model(o)
        elif 'oml:serialized_object' in o:
            serialized_type = o['oml:serialized_object']
            value = o['value']
            if serialized_type == 'type':
                rval = deserialize_type(value)
            elif serialized_type == 'rv_frozen':
                rval = deserialize_rv_frozen(value)
            elif serialized_type == 'function':
                rval = deserialize_function(value)
            else:
                raise ValueError('Cannot deserialize %s' % serialized_type)
        else:
            rval = {deserialize_object(key): deserialize_object(value)
                    for key, value in o.items()}
    elif isinstance(o, (list, tuple)):
        rval = [deserialize_object(element) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif o is None:
        rval = None
    elif isinstance(o, six.string_types):
        rval = o
    elif isinstance(o, int):
        rval = o
    elif isinstance(o, float):
        rval = o
    elif isinstance(o, OpenMLFlow):
        rval = o.model
    else:
        raise TypeError(o)
    assert o is None or rval is not None

    return rval


def serialize_model(model):
    sub_components = []
    parameters = []

    model_parameters = model.get_params()

    for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
        rval = serialize_object(v)

        if isinstance(rval, (list, tuple)):
            # Steps in a pipeline or feature union
            for identifier, sub_component in rval:
                sub_component = OrderedDict((('oml:identifier', 'step__' + identifier),
                                             ('oml:flow', sub_component)))
                sub_components.append(sub_component)
            param_dict = OrderedDict()
            param_dict['oml:name'] = k
            param_dict['oml:default_value'] = rval
            parameters.append(param_dict)
        elif isinstance(rval, OpenMLFlow):
            # Since serialize_object can return a Flow, we need to check
            # whether that flow represents a hyperparameter value (or is a
            # flow which was created because of a pipeline or e feature union)
            model_parameters = signature(model.__init__)
            if k not in model_parameters.parameters:
                continue

            # A subcomponent, for example the base model in AdaBoostClassifier
            identifier = rval.name
            sub_component = OrderedDict((('oml:identifier', identifier),
                                         ('oml:flow', rval)))
            sub_components.append(sub_component)
            param_dict = OrderedDict()
            param_dict['oml:name'] = k
            param_dict['oml:default_value'] = rval
            parameters.append(param_dict)
        else:
            # Since Pipeline and FeatureUnion also return estimators and
            # transformers in the 'steps' list with get_params(), we must
            # add them as a component, but not as a parameter of the
            # flow. The next if makes sure that we only add parameters
            # for the first case.
            model_parameters = signature(model.__init__)
            if k not in model_parameters.parameters:
                continue

            # a regular hyperparameter
            param_dict = OrderedDict()
            param_dict['oml:name'] = k
            if not (hasattr(rval, '__len__') and len(rval) == 0):
                param_dict['oml:default_value'] = rval
            parameters.append(param_dict)

    name = model.__module__ + "." + model.__class__.__name__
    sub_components_names = ",".join(
        [sub_component['oml:flow'].name
         for sub_component in sub_components])
    if sub_components_names:
        name = '%s(%s)' % (name, sub_components_names)
    if len(name) > MAXIMAL_FLOW_LENGTH:
        raise OpenMLRestrictionViolated('Flow name must not be longer '
                                        + 'than %d characters!' % MAXIMAL_FLOW_LENGTH)

    flow = OpenMLFlow(model=model, description='Automatically created '
                                               'sub-component.',
                      parameters=parameters, components=sub_components)
    # TODO add name to the constructor
    flow.name = name


    return flow


def deserialize_model(flow):
    # TODO remove potential test sentinel during testing!
    model_name = flow.name
    # Remove everything after the first bracket
    pos = model_name.find('(')
    if pos >= 0:
        model_name = model_name[:pos]

    parameters = flow.parameters
    parameter_dict = {}

    for parameter in parameters:
        name = parameter['oml:name']
        value = parameter.get('oml:default_value', None)

        rval = deserialize_object(value)
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
    return {'oml:serialized_object': 'type',
            'value': mapping[o]}


def deserialize_type(o):
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
    return {'oml:serialized_object': 'rv_frozen',
            'value': {'dist': dist, 'a': a, 'b': b, 'args': args, 'kwds': kwds}}


def deserialize_rv_frozen(o):
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
    return {'oml:serialized_object': 'function',
            'value': name}


def deserialize_function(name):
    module_name = name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except Exception as e:
        warnings.warn('Cannot load function %s due to %s.' % (name, e))
        return None
    return model_class


# This produces a flow, thus it does not need a deserialize. It cannot be fed
# to serialize_model() because cross-validators do not have get_params().
def serialize_cross_validator(o):
    parameters = []

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

        param_dict = OrderedDict()
        param_dict['oml:name'] = key
        if not (hasattr(value, '__len__') and len(value) == 0):
            param_dict['oml:default_value'] = value
        parameters.append(param_dict)

    # Create a flow
    name = o.__module__ + "." + o.__class__.__name__

    flow = OpenMLFlow(model=o, description='Automatically created '
                                           'sub-component.',
                      parameters=parameters, components=[])
    # TODO add name to the constructor
    flow.name = name

    return flow
