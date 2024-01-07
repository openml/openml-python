# License: BSD 3-Clause
from __future__ import annotations


class DummyRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0]

    def get_params(self, deep=False):
        return {}

    def set_params(self, params):
        return self
