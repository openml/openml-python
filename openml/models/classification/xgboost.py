"""Xgboost classifier."""

from __future__ import annotations

from openml.models.apis import _ModelPkgClassifier


class OpenmlPkg__XGBClassifier(_ModelPkgClassifier):
    _tags = {
        "pkg_id": "XGBClassifier",
        "python_dependencies": "xgboost",
        "pkg_pypi_name": "xgboost",
    }

    _obj = "xgboost.XGBClassifier"
