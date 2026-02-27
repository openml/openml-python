from __future__ import annotations

from pathlib import Path

import openml


class MinIOClient:
    """
    Lightweight client configuration for interacting with a MinIO-compatible
    object storage service.

    This class stores basic configuration such as a base filesystem path and
    default HTTP headers. It is intended to be extended with actual request
    or storage logic elsewhere.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Configured base path for storage operations. If None, uses the default
        cache directory from openml.config.

    Attributes
    ----------
    path : pathlib.Path or None
        Configured base path for storage operations.
    headers : dict of str to str
        Default HTTP headers, including a user-agent identifying the
        OpenML Python client version.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        return Path(openml.config.get_cache_directory())
