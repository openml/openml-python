"""Base package for sklearn classifiers."""

from openml.models.base import _OpenmlModelPkg


class _ModelPkgClassifier(_OpenmlModelPkg):

    _tags = {
        # tags specific to API type
        "pkg_obj_type": "classifier",
    }

    def get_obj_tags(self):
        """Return tags of the object as a dictionary."""
        return {}  # this needs to be implemented

    def get_obj_param_names(self):
        """Return parameter names of the object as a list.

        Returns
        -------
        list: names of object parameters
        """
        return list(self.materialize()().get_params().keys())
