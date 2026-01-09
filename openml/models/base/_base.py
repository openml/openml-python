"""Base model package class."""

from __future__ import annotations

from openml.base import _BasePkg


class _OpenmlModelPkg(_BasePkg):
    _obj = None
    _obj_dict = {}

    def __init__(self, id=None):
        super().__init__()

        pkg_id = self.get_tag("pkg_id")
        if pkg_id == "__multiple":
            self._obj = self._obj_dict.get(id, None)

    @classmethod
    def contained_ids(cls):
        """Return list of ids of objects contained in this package.

        Returns
        -------
        ids : list of str
            list of unique identifiers of objects contained in this package
        """
        pkg_id = cls.get_class_tag("pkg_id")
        if pkg_id != "__multiple":
            return [cls.get_class_tag("pkg_id")]
        return list(cls._obj_dict.keys())

    def _materialize(self):
        pkg_obj = self.get_tag("pkg_obj")

        _obj = self._obj

        if _obj is None:
            raise ValueError(
                "Error in materialize."
                "Either _materialize must be implemented, or"
                "the _obj attribute must be not None."
            )

        if pkg_obj == "reference":
            from skbase.utils.dependencies import _safe_import

            obj_loc = self._obj
            pkg_name = self.get_tag("pkg_pypi_name")

            return _safe_import(obj_loc, pkg_name=pkg_name)

        if pkg_obj == "code":
            exec(self._obj)

            return obj

        # elif pkg_obj == "craft":
        #    identify and call appropriate craft method

        raise ValueError(
            'Error in package tag "pkg_obj", '
            'must be one of "reference", "code", "craft", '
            f"but found value {pkg_obj}, of type {type(pkg_obj)}"
        )
