from collections import OrderedDict, defaultdict
import importlib
import inspect
import json
import json.decoder
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


class SklearnToFlowConverter(object):

    def serialize_object(self, o):

        if self._is_estimator(o) or self._is_transformer(o):
            rval = self.serialize_model(o)
        elif isinstance(o, (list, tuple)):
            rval = [self.serialize_object(element) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif o is None:
            rval = None
        elif isinstance(o, six.string_types):
            rval = o
        elif isinstance(o, bool):
            # The check for bool has to be before the check for int, otherwise,
            # isinstance will think the bool is an int and convert the bool will
            # be converted to a string which can't be parsed by json.loads
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
                key = self.serialize_object(key)
                value = self.serialize_object(value)
                rval[key] = value
            rval = rval
        elif isinstance(o, type):
            rval = self.serialize_type(o)
        elif isinstance(o, scipy.stats.distributions.rv_frozen):
            rval = self.serialize_rv_frozen(o)
        # This only works for user-defined functions (and not even partial).
        # I think this exactly we want here as there shouldn't be any built-in or
        # functool.partials in a pipeline
        elif inspect.isfunction(o):
            rval = self.serialize_function(o)
        elif self._is_cross_validator(o):
            rval = self.serialize_cross_validator(o)
        else:
            raise TypeError(o)

        assert o is None or rval is not None

        return rval

    # TODO maybe remove those functions and put the check to the long
    # if-constructs above?
    def _is_estimator(self, o):
        # TODO @amueller should one rather check whether this is a subclass of
        # BaseEstimator?
        #return (hasattr(o, 'fit') and hasattr(o, 'predict') and
        #        hasattr(o, 'get_params') and hasattr(o, 'set_params'))
        return isinstance(o, sklearn.base.BaseEstimator)

    def _is_transformer(self, o):
        # TODO @amueller should one rather check whether this is a subclass of
        # BaseTransformer?
        return (hasattr(o, 'fit') and hasattr(o, 'transform') and
                hasattr(o, 'get_params') and hasattr(o, 'set_params'))

    def _is_cross_validator(self, o):
        return isinstance(o, sklearn.model_selection.BaseCrossValidator)

    def deserialize_object(self, o, **kwargs):
        if isinstance(o, six.string_types):
            try:
                o = json.loads(o)
            except json.decoder.JSONDecodeError:
                pass

        if isinstance(o, dict):
            if 'oml:name' in o and 'oml:description' in o:
                rval = self.deserialize_model(o, **kwargs)
            elif 'oml:serialized_object' in o:
                serialized_type = o['oml:serialized_object']
                value = o['value']
                if serialized_type == 'type':
                    rval = self.deserialize_type(value, **kwargs)
                elif serialized_type == 'rv_frozen':
                    rval = self.deserialize_rv_frozen(value, **kwargs)
                elif serialized_type == 'function':
                    rval = self.deserialize_function(value, **kwargs)
                elif serialized_type == 'component_reference':
                    value = self.deserialize_object(value)
                    step_name = value['step_name']
                    key = value['key']
                    component = self.deserialize_object(kwargs['components'][key])
                    if step_name is None:
                        rval = component
                    else:
                        rval = (step_name, component)
                else:
                    raise ValueError('Cannot deserialize %s' % serialized_type)
            else:
                rval = {self.deserialize_object(key, **kwargs): self.deserialize_object(value, **kwargs)
                        for key, value in o.items()}
        elif isinstance(o, (list, tuple)):
            rval = [self.deserialize_object(element, **kwargs) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, bool):
            rval = o
        elif isinstance(o, int):
            rval = o
        elif isinstance(o, float):
            rval = o
        elif isinstance(o, six.string_types):
            rval = o
        elif o is None:
            rval = None
        elif isinstance(o, OpenMLFlow):
            rval = self.deserialize_model(o, **kwargs)
        else:
            raise TypeError(o)
        assert o is None or rval is not None

        return rval

    def serialize_model(self, model):
        sub_components = OrderedDict()
        parameters = OrderedDict()
        parameters_meta_info = OrderedDict()

        model_parameters = model.get_params()

        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self.serialize_object(v)

            if isinstance(rval, (list, tuple)):
                # Steps in a pipeline or feature union
                parameter_value = list()
                for identifier, sub_component in rval:
                    pos = identifier.find('(')
                    if pos >= 0:
                        identifier = identifier[:pos]

                    sub_component_identifier = k + '__' + identifier
                    sub_components[sub_component_identifier] = sub_component
                    component_reference = {'oml:serialized_object':'component_reference',
                                           'value': {'key': sub_component_identifier,
                                                     'step_name': identifier}}
                    parameter_value.append(component_reference)
                if isinstance(rval, tuple):
                    parameter_value = tuple(parameter_value)

                parameter_value = json.dumps(parameter_value)
                parameters[k] = parameter_value

            elif isinstance(rval, OpenMLFlow):
                # Since serialize_object can return a Flow, we need to check
                # whether that flow represents a hyperparameter value (or is a
                # flow which was created because of a pipeline or e feature union)
                model_parameters = signature(model.__init__)
                if k not in model_parameters.parameters:
                    continue

                # A subcomponent, for example the base model in AdaBoostClassifier
                sub_components[k] = rval
                component_reference = {'oml:serialized_object':'component_reference',
                                           'value': {'key': k,
                                                     'step_name': None}}
                component_reference = self.serialize_object(component_reference)
                parameters[k] = json.dumps(component_reference)

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
                if not (hasattr(rval, '__len__') and len(rval) == 0):
                    rval = json.dumps(rval)
                    parameters[k] = rval
                else:
                    parameters[k] = None

            parameters_meta_info[k] = OrderedDict((('description', None),
                                                   ('data_type', None)))

        name = model.__module__ + "." + model.__class__.__name__
        sub_components_names = ",".join(
            [sub_components[key].name for key in sub_components])
        if sub_components_names:
            name = '%s(%s)' % (name, sub_components_names)
        if len(name) > MAXIMAL_FLOW_LENGTH:
            raise OpenMLRestrictionViolated('Flow name must not be longer ' +
                                            'than %d characters!' % MAXIMAL_FLOW_LENGTH)

        external_version = self._get_external_version_info()
        flow = OpenMLFlow(name=name,
                          description='Automatically created sub-component.',
                          model=model,
                          components=sub_components,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version)

        return flow

    def deserialize_model(self, flow, **kwargs):
        # TODO remove potential test sentinel during testing!
        model_name = flow.name
        # Remove everything after the first bracket
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
                rval = self.deserialize_object(value)
                component_dict[parameter_name][step] = rval
            else:
                value = components[name]
                rval = self.deserialize_object(value)
                parameter_dict[name] = rval

        for name in parameters:
            value = parameters.get(name)
            rval = self.deserialize_object(value, components=components)
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

    def serialize_type(self, o):
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

    def deserialize_type(self, o, **kwargs):
        mapping = {'float': float,
                   'np.float': np.float,
                   'np.float32': np.float32,
                   'np.float64': np.float64,
                   'int': int,
                   'np.int': np.int,
                   'np.int32': np.int32,
                   'np.int64': np.int64}
        return mapping[o]

    def serialize_rv_frozen(self, o):
        args = o.args
        kwds = o.kwds
        a = o.a
        b = o.b
        dist = o.dist.__class__.__module__ + '.' + o.dist.__class__.__name__
        return {'oml:serialized_object': 'rv_frozen',
                            'value': {'dist': dist, 'a': a, 'b': b, 'args': args, 'kwds': kwds}}

    def deserialize_rv_frozen(self, o, **kwargs):
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

    def serialize_function(self, o):
        name = o.__module__ + '.' + o.__name__
        return {'oml:serialized_object': 'function',
                            'value': name}

    def deserialize_function(self, name, **kwargs):
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
    def serialize_cross_validator(self, o):
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
                parameters[key] = value
            else:
                parameters[key] = None
            parameters_meta_info[key] = OrderedDict((('description', None),
                                                     ('data_type', None)))

        # Create a flow
        name = o.__module__ + "." + o.__class__.__name__

        external_version = self._get_external_version_info()
        flow = OpenMLFlow(name=name,
                          description='Automatically created sub-component.',
                          model=o,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info)

        return flow

    def _get_external_version_info(self):
        return 'sklearn_' + sklearn.__version__
