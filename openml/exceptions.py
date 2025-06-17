# License: BSD 3-Clause
from __future__ import annotations


class PyOpenMLError(Exception):
    """Base class for all exceptions in OpenML-Python."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class OpenMLServerError(PyOpenMLError):
    """class for when something is really wrong on the server
    (result did not parse to dict), contains unparsed error.
    """


class OpenMLServerException(OpenMLServerError):  # noqa: N818
    """exception for when the result of the server was
    not 200 (e.g., listing call w/o results).
    """

    # Code needs to be optional to allow the exception to be picklable:
    # https://stackoverflow.com/questions/16244923/how-to-make-a-custom-exception-class-with-multiple-init-args-pickleable  # noqa: E501
    def __init__(self, message: str, code: int | None = None, url: str | None = None):
        self.message = message
        self.code = code
        self.url = url
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.url} returned code {self.code}: {self.message}"


class OpenMLServerNoResult(OpenMLServerException):
    """Exception for when the result of the server is empty."""


class OpenMLCacheException(PyOpenMLError):  # noqa: N818
    """Dataset / task etc not found in cache"""


class OpenMLHashException(PyOpenMLError):  # noqa: N818
    """Locally computed hash is different than hash announced by the server."""


class OpenMLPrivateDatasetError(PyOpenMLError):
    """Exception thrown when the user has no rights to access the dataset."""


class OpenMLRunsExistError(PyOpenMLError):
    """Indicates run(s) already exists on the server when they should not be duplicated."""

    def __init__(self, run_ids: set[int], message: str) -> None:
        if len(run_ids) < 1:
            raise ValueError("Set of run ids must be non-empty.")
        self.run_ids = run_ids
        super().__init__(message)


class OpenMLNotAuthorizedError(OpenMLServerError):
    """Indicates an authenticated user is not authorized to execute the requested action."""


class ObjectNotPublishedError(PyOpenMLError):
    """Indicates an object has not been published yet."""
