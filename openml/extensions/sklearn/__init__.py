# License: BSD 3-Clause

from .extension import SklearnExtension
from openml.extensions import register_extension


__all__ = ["SklearnExtension"]

register_extension(SklearnExtension)


def cont(X):
    """Returns True for all non-categorical columns, False for the rest.
    """
    if not hasattr(X, "dtypes"):
        raise AttributeError("Not a Pandas DataFrame with 'dtypes' as attribute!")
    return X.dtypes != "category"


def cat(X):
    """Returns True for all categorical columns, False for the rest.
    """
    if not hasattr(X, "dtypes"):
        raise AttributeError("Not a Pandas DataFrame with 'dtypes' as attribute!")
    return X.dtypes == "category"
