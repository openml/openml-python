"""Xgboost classifier."""


from openml.models.apis import _ModelPkgClassifier


class OpenmlPkg__XGBClassifier(_ModelPkgClassifier):

    _tags = {
        "pkg_id": "XGBClassifier",
        "python_dependencies": "xgboost",
    }

    _obj = "xgboost.XGBClassifier"
