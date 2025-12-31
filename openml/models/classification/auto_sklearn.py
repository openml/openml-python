"""Auto-sklearn classifier."""


from openml.models.apis import _ModelPkgClassifier


class OpenmlPkg__AutoSklearnClassifier(_ModelPkgClassifier):

    _tags = {
        "pkg_id": "AutoSklearnClassifier",
        "python_dependencies": "auto-sklearn",
    }

    _obj = "autosklearn.classification.AutoSklearnClassifier"
