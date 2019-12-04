# License: BSD 3-Clause


class PyOpenMLError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class OpenMLServerError(PyOpenMLError):
    """class for when something is really wrong on the server
       (result did not parse to dict), contains unparsed error."""

    def __init__(self, message: str):
        super().__init__(message)


class OpenMLServerException(OpenMLServerError):
    """exception for when the result of the server was
       not 200 (e.g., listing call w/o results). """

    # Code needs to be optional to allow the exceptino to be picklable:
    # https://stackoverflow.com/questions/16244923/how-to-make-a-custom-exception-class-with-multiple-init-args-pickleable  # noqa: E501
    def __init__(self, message: str, code: int = None, url: str = None):
        self.message = message
        self.code = code
        self.url = url
        super().__init__(message)

    def __repr__(self):
        return '%s returned code %s: %s' % (
            self.url, self.code, self.message,
        )


class OpenMLServerNoResult(OpenMLServerException):
    """exception for when the result of the server is empty. """
    pass


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""
    def __init__(self, message: str):
        super().__init__(message)


class OpenMLHashException(PyOpenMLError):
    """Locally computed hash is different than hash announced by the server."""
    pass


class OpenMLPrivateDatasetError(PyOpenMLError):
    """ Exception thrown when the user has no rights to access the dataset. """
    def __init__(self, message: str):
        super().__init__(message)


class OpenMLRunsExistError(PyOpenMLError):
    """ Indicates run(s) already exists on the server when they should not be duplicated. """
    def __init__(self, run_ids: set, message: str):
        if len(run_ids) < 1:
            raise ValueError("Set of run ids must be non-empty.")
        self.run_ids = run_ids
        super().__init__(message)
