# License: BSD 3-Clause

from .extension import SklearnExtension
from openml.extensions import register_extension


__all__ = ["SklearnExtension"]

register_extension(SklearnExtension)
