from __future__ import annotations

from pathlib import Path

from openml.__version__ import __version__


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
        Base path used for local storage or downloads. If ``None``, no
        default path is configured.

    Attributes
    ----------
    path : pathlib.Path or None
        Configured base path for storage operations.
    headers : dict of str to str
        Default HTTP headers, including a user-agent identifying the
        OpenML Python client version.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}
