# License: BSD 3-Clause
from __future__ import annotations


class PyOpenMLError(Exception):
    """Base class for all exceptions in OpenML-Python."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# ============================================================================
# Server Communication Errors
# ============================================================================


class OpenMLServerError(PyOpenMLError):
    """Base class for all server-related errors.
    
    Raised when communication with the OpenML server fails or 
    the server returns an unexpected response.
    """


class OpenMLServerException(OpenMLServerError):
    """Exception raised when the server returns a structured error response.
    
    This is raised when the server returns a non-200 status code along with
    a parseable error message containing an error code and description.
    
    Attributes
    ----------
    code : int | None
        The OpenML-specific error code (not HTTP status code)
    url : str | None
        The URL that was called when the error occurred
    message : str
        The error message from the server
    """

    def __init__(self, message: str, code: int | None = None, url: str | None = None):
        self.message = message
        self.code = code
        self.url = url
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.url} returned code {self.code}: {self.message}"


# ============================================================================
# HTTP-Related Server Errors
# ============================================================================


class OpenMLURITooLongError(OpenMLServerError):
    """Exception raised when the request URI exceeds server limits.
    
    HTTP Status: 414 (URI Too Long)
    
    This typically occurs when trying to pass too many parameters in a GET request.
    Consider using POST or breaking the request into smaller chunks.
    """

    def __init__(self, url: str, message: str | None = None):
        self.url = url
        if message is None:
            message = f"URI too long! ({url}). Consider using POST or reducing parameters."
        super().__init__(message)


class OpenMLRateLimitError(OpenMLServerError):
    """Exception raised when API rate limits are exceeded.
    
    HTTP Status: 429 (Too Many Requests)
    
    The user has sent too many requests in a given amount of time.
    """

    def __init__(self, message: str | None = None, retry_after: int | None = None):
        self.retry_after = retry_after
        if message is None:
            message = "Rate limit exceeded. Please try again later."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        super().__init__(message)


class OpenMLNotFoundError(OpenMLServerError):
    """Exception raised when a requested resource does not exist.
    
    HTTP Status: 404 (Not Found)
    
    This can occur when requesting a dataset, task, flow, or run that doesn't exist.
    """

    def __init__(self, resource_type: str | None = None, resource_id: int | None = None, 
                 message: str | None = None):
        self.resource_type = resource_type
        self.resource_id = resource_id
        
        if message is None:
            if resource_type and resource_id:
                message = f"{resource_type} with ID {resource_id} not found."
            else:
                message = "Requested resource not found."
        super().__init__(message)


class OpenMLServerNoResult(OpenMLServerException):
    """Exception raised when the server returns an empty result set.
    
    OpenML Error Codes: 111, 372, 482, 500, 512, 542, 674
    - 111: Dataset descriptions
    - 372: Datasets
    - 482: Tasks
    - 500: Flows
    - 512: Runs
    - 542: Evaluations
    - 674: Setups
    
    This is different from NotFound - the request was valid but returned no results.
    """


# ============================================================================
# Validation and Input Errors
# ============================================================================


class OpenMLValidationError(OpenMLServerException):
    """Exception raised when submitted data fails server-side validation.
    
    OpenML Error Codes: 163 (flow XML validation failure), and others
    
    This occurs when uploading data that doesn't meet the required format or constraints.
    """

    def __init__(self, message: str, code: int | None = None, url: str | None = None,
                 validation_details: str | None = None):
        self.validation_details = validation_details
        super().__init__(message, code, url)


# ============================================================================
# Authentication and Authorization Errors
# ============================================================================


class OpenMLAuthenticationError(OpenMLServerError):
    """Exception raised when API authentication fails.
    
    This occurs when:
    - No API key is provided when required
    - An invalid API key is provided
    - The API key format is incorrect
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "Authentication required. Please configure your API key.\n"
                "See: https://openml.github.io/openml-python/latest/examples/"
                "Basics/introduction_tutorial/#authentication"
            )
        super().__init__(message)


class OpenMLNotAuthorizedError(OpenMLServerError):
    """Exception raised when an authenticated user lacks permission.
    
    OpenML Error Codes: 102, 137, 310, 320, 350, 400, 460
    - 102: Flow exists (POST)
    - 137: Dataset (POST)
    - 310: Flow/<something> (POST)
    - 320: Flow delete
    - 350: Dataset delete
    - 400: Run delete
    - 460: Task delete
    
    This is different from authentication - the user is authenticated but
    doesn't have permission to perform the requested action.
    """


# ============================================================================
# Database and Connectivity Errors
# ============================================================================


class OpenMLDatabaseConnectionError(OpenMLServerException):
    """Exception raised when the server experiences database connectivity issues.
    
    OpenML Error Code: 107
    
    This is typically a temporary issue on the server side. Retrying may resolve it.
    """

    def __init__(self, message: str, code: int = 107, url: str | None = None):
        super().__init__(message, code, url)


# ============================================================================
# Data Integrity Errors
# ============================================================================


class OpenMLHashException(PyOpenMLError):
    """Exception raised when file hash validation fails.
    
    This occurs when the locally computed hash of a downloaded file doesn't match
    the hash provided by the server, indicating potential data corruption or
    transmission errors.
    """


# ============================================================================
# Cache-Related Errors
# ============================================================================


class OpenMLCacheException(PyOpenMLError):
    """Exception raised when requested data is not found in local cache.
    
    This is typically used internally when attempting to load datasets, tasks,
    or other resources from the cache before falling back to downloading from
    the server.
    """


# ============================================================================
# Privacy and Access Errors
# ============================================================================


class OpenMLPrivateDatasetError(PyOpenMLError):
    """Exception raised when attempting to access a private dataset.
    
    This occurs when a user without proper permissions tries to access a
    dataset that is marked as private or restricted.
    """


# ============================================================================
# Upload and Publishing Errors
# ============================================================================


class OpenMLRunsExistError(PyOpenMLError):
    """Exception raised when attempting to upload runs that already exist.
    
    This prevents duplicate run submissions to the server.
    
    Attributes
    ----------
    run_ids : set[int]
        The IDs of the runs that already exist on the server
    """

    def __init__(self, run_ids: set[int], message: str) -> None:
        if len(run_ids) < 1:
            raise ValueError("Set of run ids must be non-empty.")
        self.run_ids = run_ids
        super().__init__(message)


class ObjectNotPublishedError(PyOpenMLError):
    """Exception raised when attempting to access an unpublished object.
    
    Some objects may be uploaded but not yet published/activated on the server.
    """


# ============================================================================
# Timeout Errors
# ============================================================================


class OpenMLTimeoutError(OpenMLServerError):
    """Exception raised when a request to the server times out.
    
    HTTP Status: 408 (Request Timeout) or 504 (Gateway Timeout)
    
    This can occur for long-running operations or poor network connectivity.
    """

    def __init__(self, message: str | None = None, timeout_seconds: float | None = None):
        self.timeout_seconds = timeout_seconds
        if message is None:
            message = "Request to OpenML server timed out."
        if timeout_seconds:
            message += f" (timeout: {timeout_seconds}s)"
        super().__init__(message)


# ============================================================================
# Service Unavailable Errors
# ============================================================================


class OpenMLServiceUnavailableError(OpenMLServerError):
    """Exception raised when the OpenML service is temporarily unavailable.
    
    HTTP Status: 503 (Service Unavailable)
    
    This typically indicates server maintenance or temporary overload.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = "OpenML service is temporarily unavailable. Please try again later."
        super().__init__(message)