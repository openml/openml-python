# License: BSD 3-Clause

from .extension import SklearnExtension
from openml.extensions import register_extension


__all__ = ["SklearnExtension"]

register_extension(SklearnExtension)


def cont(X):
    """Returns True for all non-categorical columns, False for the rest.

    This is a helper function for OpenML datasets encoded as DataFrames simplifying the handling
    of mixed data types. To build sklearn models on mixed data types, a ColumnTransformer is
    required to process each type of columns separately.
    This function allows transformations meant for continuous/numeric columns to access the
    continuous/numeric columns given the dataset as DataFrame.
    """
    if not hasattr(X, "dtypes"):
        raise AttributeError("Not a Pandas DataFrame with 'dtypes' as attribute!")
    return X.dtypes != "category"


def cat(X):
    """Returns True for all categorical columns, False for the rest.

    This is a helper function for OpenML datasets encoded as DataFrames simplifying the handling
    of mixed data types. To build sklearn models on mixed data types, a ColumnTransformer is
    required to process each type of columns separately.
    This function allows transformations meant for categorical columns to access the
    categorical columns given the dataset as DataFrame.
    """
    if not hasattr(X, "dtypes"):
        raise AttributeError("Not a Pandas DataFrame with 'dtypes' as attribute!")
    return X.dtypes == "category"
