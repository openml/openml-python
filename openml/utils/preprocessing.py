
from sklearn.preprocessing.imputation import Imputer, check_array, _get_mask, _most_frequent

import warnings

import numpy as np
import numpy.ma as ma
from scipy import sparse
from sklearn.utils.fixes import astype
from sklearn.utils.sparsefuncs import _get_median
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES


class ConditionalImputer(Imputer):
    """Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <imputation>`.

    Parameters
    ----------
    missing_values : integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".

    strategy : string, optional (default="mean")
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis.

    axis : integer, optional (default=0)
        The axis along which to impute.

        - If `axis=0`, then impute along columns.
        - If `axis=1`, then impute along rows. (Not supported)

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If `axis=0` and X is encoded as a CSR matrix;
        - If `axis=1` and X is encoded as a CSC matrix.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature if axis == 0.

    Notes
    -----
    - When ``axis=0``, columns which only contained missing values at `fit`
      are discarded upon `transform`.
    - When ``axis=1``, an exception is raised if there are rows for which it is
      not possible to fill in the missing values (e.g., because they only
      contain missing values).
    """
    def __init__(self, missing_values="NaN", strategy="mean",
                 strategy_nominal="most_frequent",
                 indexes_nominal=None,
                 axis=0, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.strategy_nominal = strategy_nominal
        self.indexes_nominal = indexes_nominal
        self.axis = axis
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0]:
            raise ValueError("Can only impute missing values on axis 0 (axis 1 not supported), "
                             " got axis={0}".format(self.axis))

        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                        force_all_finite=False)

        if sparse.issparse(X):
            statistics_general = self._sparse_fit(X,
                                                  self.strategy,
                                                  self.missing_values,
                                                  self.axis)
            statistics_nominal = self._sparse_fit(X,
                                                  self.strategy_nominal,
                                                  self.missing_values,
                                                  self.axis)
        else:
            statistics_general = self._dense_fit(X,
                                                 self.strategy,
                                                 self.missing_values,
                                                 self.axis)
            statistics_nominal = self._dense_fit(X,
                                                  self.strategy_nominal,
                                                  self.missing_values,
                                                  self.axis)

        # here the indexes of nominal values get set
        self.statistics_ = statistics_general
        if self.indexes_nominal is not None:
            for i in self.indexes_nominal:
                self.statistics_[i] = statistics_nominal[i]

        return self