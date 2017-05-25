class PyOpenMLError(Exception):
    def __init__(self, message):
        self.message = message
        super(PyOpenMLError, self).__init__(message)


class OpenMLServerError(PyOpenMLError):
    """class for when something is really wrong on the server
       (result did not parse to dict), contains unparsed error."""

    def __init__(self, message):
        super(OpenMLServerError, self).__init__(message)

#
class OpenMLServerException(OpenMLServerError):
    """exception for when the result of the server was
       not 200 (e.g., listing call w/o results). """

    def __init__(self, code, message, additional=None):
        self.code = code
        self.additional = additional
        super(OpenMLServerException, self).__init__(message)


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)
