"""Tests for sklearn typing utilities in utils.sktime."""

__author__ = ["fkiraly"]


import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

KMeans = _safe_import("sklearn.cluster.KMeans")
KNeighborsClassifier = _safe_import("sklearn.neighbors.KNeighborsClassifier")
KNeighborsRegressor = _safe_import("sklearn.neighbors.KNeighborsRegressor")
StandardScaler = _safe_import("sklearn.preprocessing.StandardScaler")

from openml.utils._sklearn_compat import is_sklearn_estimator, sklearn_scitype

if _check_soft_dependencies("scikit-learn", severity="none"):
    CORRECT_SCITYPES = {
        KMeans: "clusterer",
        KNeighborsClassifier: "classifier",
        KNeighborsRegressor: "regressor",
        StandardScaler: "transformer",
    }

    sklearn_estimators = list(CORRECT_SCITYPES.keys())
else:
    sklearn_estimators = []


@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_is_sklearn_estimator_positive(estimator):
    """Test that is_sklearn_estimator recognizes positive examples correctly."""
    msg = (
        f"is_sklearn_estimator incorrectly considers {estimator.__name__} "
        f"as not an sklearn estimator (output False), but output should be True"
    )
    assert is_sklearn_estimator(estimator), msg


@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_sklearn_scitype(estimator):
    """Test that sklearn_scitype returns the correct scitype string."""
    scitype = sklearn_scitype(estimator)
    expected_scitype = CORRECT_SCITYPES[estimator]
    msg = (
        f"is_sklearn_estimator returns the incorrect scitype string for "
        f'"{estimator.__name__}". Should be {expected_scitype}, but '
        f'{scitype}" was returned.'
    )
    assert scitype == expected_scitype, msg
