# License: BSD 3-Clause
from __future__ import annotations


class DummyRegressor:
    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict(self, X):
        return X[:, 0]

    def get_params(self, deep=False):  # noqa: FBT002, ARG002
        return {}

    def set_params(self, params):  # noqa: ARG002
        return self
