# License: BSD 3-Clause

from typing import Optional


class PyOpenMLError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class OpenMLServerError(PyOpenMLError):
    """class for when something is really wrong on the server
    (result did not parse to dict), contains unparsed error."""

    pass


class OpenMLServerException(OpenMLServerError):
    """exception for when the result of the server was
    not 200 (e.g., listing call w/o results)."""

    # Code needs to be optional to allow the exception to be picklable:
    # https://stackoverflow.com/questions/16244923/how-to-make-a-custom-exception-class-with-multiple-init-args-pickleable  # noqa: E501
    def __init__(self, message: str, code: Optional[int] = None, url: Optional[str] = None):
        self.message = message
        self.code = code
        self.url = url
        super().__init__(message)

    def __str__(self):
        return f"{self.url} returned code {self.code}: {self.message}"


class OpenMLServerNoResult(OpenMLServerException):
    """Exception for when the result of the server is empty."""

    pass


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""

    pass


class OpenMLHashException(PyOpenMLError):
    """Locally computed hash is different than hash announced by the server."""

    pass


class OpenMLPrivateDatasetError(PyOpenMLError):
    """Exception thrown when the user has no rights to access the dataset."""

    pass


class OpenMLRunsExistError(PyOpenMLError):
    """Indicates run(s) already exists on the server when they should not be duplicated."""

    def __init__(self, run_ids: set, message: str):
        if len(run_ids) < 1:
            raise ValueError("Set of run ids must be non-empty.")
        self.run_ids = run_ids
        super().__init__(message)


class OpenMLNotAuthorizedError(OpenMLServerError):
    """Indicates an authenticated user is not authorized to execute the requested action."""

    pass
