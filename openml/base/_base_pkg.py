"""Base Packager class."""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

from skbase.base import BaseObject
from skbase.utils.dependencies import _check_estimator_deps


class _BasePkg(BaseObject):
    _tags = {
        "python_dependencies": None,
        "python_version": None,
        # package register and manifest
        "pkg_id": None,  # object id contained, "__multiple" if multiple
        "pkg_obj": "reference",  # or "code"
        "pkg_obj_type": None,  # openml API type
        "pkg_compression": "zlib",  # compression
        "pkg_pypi_name": None,  # PyPI package name of objects
    }

    def __init__(self):
        super().__init__()

    def materialize(self):
        try:
            _check_estimator_deps(obj=self)
        except ModuleNotFoundError as e:
            # prettier message, so the reference is to the pkg_id
            # currently, we cannot simply pass the object name to skbase
            # in the error message, so this is a hack
            # todo: fix this in scikit-base
            msg = str(e)
            if len(msg) > 11:
                msg = msg[11:]
            raise ModuleNotFoundError(msg) from e

        return self._materialize()

    def _materialize(self):
        raise RuntimeError("abstract method")

    def serialize(self):
        cls_str = class_to_source(type(self))
        compress_method = self.get_tag("pkg_compression")
        if compress_method in [None, "None"]:
            return cls_str

        cls_str = cls_str.encode("utf-8")
        exec(f"import {compress_method}")
        return eval(f"{compress_method}.compress(cls_str)")


def _has_source(obj) -> bool:
    """Return True if inspect.getsource(obj) should succeed."""
    module_name = getattr(obj, "__module__", None)
    if not module_name or module_name not in sys.modules:
        return False

    module = sys.modules[module_name]
    file = getattr(module, "__file__", None)
    if not file:
        return False

    return Path(file).suffix == ".py"


def class_to_source(cls) -> str:
    """Return full source definition of python class as string.

    Parameters
    ----------
    cls : class to serialize

    Returns
    -------
    str : complete definition of cls, as str.
        Imports are not contained or serialized.
    """ ""

    # Fast path: class has retrievable source
    if _has_source(cls):
        source = inspect.getsource(cls)
        return textwrap.dedent(source)

    # Fallback for dynamically created classes
    lines = []

    bases = [base.__name__ for base in cls.__bases__ if base is not object]
    base_str = f"({', '.join(bases)})" if bases else ""
    lines.append(f"class {cls.__name__}{base_str}:")

    body_added = False

    for name, value in cls.__dict__.items():
        if name.startswith("__") and name.endswith("__"):
            continue

        if inspect.isfunction(value):
            if _has_source(value):
                method_src = inspect.getsource(value)
                method_src = textwrap.indent(textwrap.dedent(method_src), "    ")
                lines.append(method_src)
            else:
                lines.append(f"    def {name}(self): ...")
            body_added = True
        else:
            lines.append(f"    {name} = {value!r}")
            body_added = True

    if not body_added:
        lines.append("    pass")

    return "\n".join(lines)
