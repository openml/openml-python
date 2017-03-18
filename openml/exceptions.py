class PyOpenMLError(Exception):
    def __init__(self, message):
        super(PyOpenMLError, self).__init__(message)

# class for when something is really wrong on the server (result did not parse to dict)
class OpenMLServerError(PyOpenMLError):
    """Server didn't respond 200, contains unparsed error."""
    def __init__(self, message):
        message = "OpenML Server error: " + message
        super(OpenMLServerError, self).__init__(message)

# class for when the result of the server was not 200 (e.g., listing call w/o results)
class OpenMLServerException(OpenMLServerError):
    """Server didn't respond 200."""
    def __init__(self, code, message, additional=None):
        self.code = code
        self.additional = additional
        message = "OpenML Server exception: " + message
        super(OpenMLServerException, self).__init__(message)


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)
