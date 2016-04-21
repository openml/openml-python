class PyOpenMLError(Exception):
    def __init__(self, message):
        super(PyOpenMLError, self).__init__(message)


class OpenMLServerError(PyOpenMLError):
    """Server didn't respond 200."""
    def __init__(self, message):
        message = "OpenML Server error: " + message
        super(OpenMLServerError, self).__init__(message)


class OpenMLCacheException(PyOpenMLError):
    """Dataset / task etc not found in cache"""
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)
