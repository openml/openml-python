class OpenMLStatusChange(Warning):
    def __init__(self, message):
        super(OpenMLStatusChange, self).__init__(message)


class OpenMLDatasetStatusChange(OpenMLStatusChange):
    def __init__(self, message):
        super(OpenMLDatasetStatusChange, self).__init__(message)


class PyOpenMLError(Exception):
    def __init__(self, message):
        super(PyOpenMLError, self).__init__(message)


class OpenMLServerError(PyOpenMLError):
    def __init__(self, message):
        super(OpenMLServerError, self).__init__(message)


class OpenMLCacheException(PyOpenMLError):
    def __init__(self, message):
        super(OpenMLCacheException, self).__init__(message)