from __future__ import annotations

from pathlib import Path

import openml
from openml.__version__ import __version__

_HEADERS: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}


class MinIOClient:
    """
    Lightweight client configuration for interacting with a MinIO-compatible
    object storage service.

    This class stores basic configuration such as a base filesystem path and
    default HTTP headers. It is intended to be extended with actual request
    or storage logic elsewhere.

    Attributes
    ----------
    path : pathlib.Path or None
        Configured base path for storage operations.
    headers : dict of str to str
        Default HTTP headers, including a user-agent identifying the
        OpenML Python client version.
    """

    @property
    def path(self) -> Path:
        return Path(openml.config.get_cache_directory())

    @property
    def headers(self) -> dict[str, str]:
        return _HEADERS
