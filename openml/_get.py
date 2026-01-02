"""Global get dispatch utility."""

# currently just a forward to models
# to discuss and possibly
# todo: add global get utility here
# in general, e.g., datasets will not have same name as models etc
from __future__ import annotations

from openml.models import get

__all__ = ["get"]
