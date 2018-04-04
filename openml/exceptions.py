class PyOpenMLError(Exception):
    def __init__(self, message):
        self.message = message
        super(PyOpenMLError, self).__init__(message)


class OpenMLServerError(PyOpenMLError):
    """class for when something is really wrong on the server
       (result did not parse to dict), contains unparsed error."""

    def __init__(self, message):
        super(OpenMLServerError, self).__init__(message)


class OpenMLServerException(OpenMLServerError):
    """exception for when the result of the server was
       not 200 (e.g., listing call w/o results). """

    # Code needs to be optional to allow the exceptino to be picklable:
    # https://stackoverflow.com/questions/16244923/how-to-make-a-custom-exception-class-with-multiple-init-args-pickleable
    def __init__(self, message, code=None, additional=None, url=None):
        self.message = message
        self.code = code
        self.additional = additional
        self.url = url
        super(OpenMLServerException, self).__init__(message)


class OpenMLServerNoResult(OpenMLServerException):
    """exception for when the result of the server is empty. """
    pass


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)


class OpenMLHashException(PyOpenMLError):
    """Locally computed hash is different than hash announced by the server."""
    pass