from __future__ import annotations

from openml._api.runtime.core import APIBackend

_backend: APIBackend = APIBackend.build(version="v1", strict=False)
