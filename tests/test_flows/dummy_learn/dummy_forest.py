# License: BSD 3-Clause


class DummyRegressor(object):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0]

    def get_params(self, deep=False):
        return {}

    def set_params(self, params):
        return self
