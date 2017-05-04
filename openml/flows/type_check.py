from collections import Sequence, OrderedDict
from scipy import stats
import numpy as np
import inspect
import sklearn
import scipy
import six


def is_estimator(o):
    return hasattr(o, 'fit') and hasattr(o, 'get_params') and hasattr(o, 'set_params')


def is_cross_validator(o):
    return isinstance(o, sklearn.model_selection.BaseCrossValidator)


def is_primitive_parameter(o):
    return isinstance(o, (bool, int, float, six.string_types, type(None)))


def is_list_like(o):
    return is_generator(o) or isinstance(o, (Sequence, np.ndarray)) and not is_string(o)


def is_generator(o):
    return inspect.isgenerator(o)


def is_string(o):
    return isinstance(o, six.string_types)


def is_dict(o):
    return isinstance(o, (dict, OrderedDict))


def is_random_variable(o):
    return isinstance(o, scipy.stats.distributions.rv_frozen)


def is_function(o):
    return inspect.isfunction(o)


def is_tuple(o):
    return isinstance(o, tuple)


def is_type(o):
    return isinstance(o, type)


def is_homogeneous_list(o, types=None):
    # Check if object is a non-empty list
    if not (is_list_like(o) and len(o) > 0):
        return False

    # Set types to given types or to type of first element
    types = type(o[0]) if types is None else types

    # Check if the list is homogeneous
    return all(isinstance(x, types) for x in o)
