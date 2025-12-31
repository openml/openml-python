"""Base model package class."""

from openml.base import _BasePkg


class _OpenmlModelPkg(_BasePkg):

    _obj = None

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

            obj = _safe_import(self._obj)
            return obj

        elif pkg_obj == "code":
            exec(self._obj)

            return obj

        # elif pkg_obj == "craft":
        #    identify and call appropriate craft method

        else:
            raise ValueError(
                'Error in package tag "pkg_obj", '
                'must be one of "reference", "code", "craft", '
                f'but found value {pkg_obj}, of type {type(pkg_obj)}'
            )
