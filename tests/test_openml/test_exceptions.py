# License: BSD 3-Clause
"""Comprehensive pytest tests for openml.exceptions module."""

from __future__ import annotations

import pickle

import pytest

from openml.exceptions import (
    ObjectNotPublishedError,
    OpenMLCacheException,
    OpenMLHashException,
    OpenMLNotAuthorizedError,
    OpenMLPrivateDatasetError,
    OpenMLRunsExistError,
    OpenMLServerError,
    OpenMLServerException,
    OpenMLServerNoResult,
    PyOpenMLError,
)


class TestPyOpenMLError:
    """Test PyOpenMLError base exception class."""

    def test_init_with_message(self):
        """Test initialization with message."""
        error = PyOpenMLError("Test error message")
        assert error.message == "Test error message"
        assert str(error) == "Test error message"

    def test_inheritance(self):
        """Test that PyOpenMLError inherits from Exception."""
        error = PyOpenMLError("Test")
        assert isinstance(error, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching PyOpenMLError."""
        with pytest.raises(PyOpenMLError) as exc_info:
            raise PyOpenMLError("Custom error")
        
        assert exc_info.value.message == "Custom error"

    def test_message_attribute(self):
        """Test that message attribute is accessible."""
        error = PyOpenMLError("Error message")
        assert hasattr(error, "message")
        assert error.message == "Error message"

    def test_empty_message(self):
        """Test with empty message."""
        error = PyOpenMLError("")
        assert error.message == ""
        assert str(error) == ""

    def test_special_characters_in_message(self):
        """Test message with special characters."""
        message = "Error: <tag> & 'quote' \"double\""
        error = PyOpenMLError(message)
        assert error.message == message


class TestOpenMLServerError:
    """Test OpenMLServerError exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLServerError("Server error")
        assert error.message == "Server error"
        assert isinstance(error, PyOpenMLError)

    def test_inheritance_chain(self):
        """Test inheritance from PyOpenMLError."""
        error = OpenMLServerError("Test")
        assert isinstance(error, PyOpenMLError)
        assert isinstance(error, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching OpenMLServerError."""
        with pytest.raises(OpenMLServerError) as exc_info:
            raise OpenMLServerError("Server down")
        
        assert exc_info.value.message == "Server down"


class TestOpenMLServerException:
    """Test OpenMLServerException exception class."""

    def test_init_with_all_parameters(self):
        """Test initialization with message, code, and URL."""
        error = OpenMLServerException(
            message="Not found",
            code=404,
            url="https://openml.org/api/test"
        )
        
        assert error.message == "Not found"
        assert error.code == 404
        assert error.url == "https://openml.org/api/test"

    def test_init_with_message_only(self):
        """Test initialization with message only."""
        error = OpenMLServerException("Error message")
        assert error.message == "Error message"
        assert error.code is None
        assert error.url is None

    def test_str_representation(self):
        """Test string representation includes code and URL."""
        error = OpenMLServerException(
            message="Server error",
            code=500,
            url="https://test.openml.org/api"
        )
        
        error_str = str(error)
        assert "500" in error_str
        assert "https://test.openml.org/api" in error_str
        assert "Server error" in error_str

    def test_inheritance(self):
        """Test inheritance from OpenMLServerError."""
        error = OpenMLServerException("Test", 400)
        assert isinstance(error, OpenMLServerError)
        assert isinstance(error, PyOpenMLError)

    def test_picklable(self):
        """Test that exception is picklable."""
        error = OpenMLServerException("Picklable error", code=500, url="https://openml.org")
        
        # Pickle and unpickle
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.message == error.message
        assert unpickled.code == error.code
        assert unpickled.url == error.url

    def test_picklable_without_optional_args(self):
        """Test pickling when code and URL are None."""
        error = OpenMLServerException("Error without code")
        
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.message == "Error without code"
        assert unpickled.code is None
        assert unpickled.url is None

    def test_raise_with_details(self):
        """Test raising exception with full details."""
        with pytest.raises(OpenMLServerException) as exc_info:
            raise OpenMLServerException(
                "Unauthorized access",
                code=401,
                url="https://openml.org/api/data/1"
            )
        
        exc = exc_info.value
        assert exc.code == 401
        assert "401" in str(exc)


class TestOpenMLServerNoResult:
    """Test OpenMLServerNoResult exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLServerNoResult("No results found", code=204)
        assert error.message == "No results found"
        assert error.code == 204

    def test_inheritance(self):
        """Test inheritance from OpenMLServerException."""
        error = OpenMLServerNoResult("Empty result")
        assert isinstance(error, OpenMLServerException)
        assert isinstance(error, OpenMLServerError)
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(OpenMLServerNoResult) as exc_info:
            raise OpenMLServerNoResult("Query returned no results", code=204, url="test.url")
        
        assert exc_info.value.code == 204


class TestOpenMLCacheException:
    """Test OpenMLCacheException exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLCacheException("Cache miss")
        assert error.message == "Cache miss"

    def test_inheritance(self):
        """Test inheritance from PyOpenMLError."""
        error = OpenMLCacheException("Test")
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(OpenMLCacheException) as exc_info:
            raise OpenMLCacheException("Dataset 123 not found in cache")
        
        assert "Dataset 123" in exc_info.value.message

    def test_typical_use_case(self):
        """Test typical use case with dataset ID."""
        dataset_id = 42
        error = OpenMLCacheException(f"Dataset {dataset_id} not found in cache")
        
        assert "Dataset 42" in error.message


class TestOpenMLHashException:
    """Test OpenMLHashException exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLHashException("Hash mismatch")
        assert error.message == "Hash mismatch"

    def test_inheritance(self):
        """Test inheritance from PyOpenMLError."""
        error = OpenMLHashException("Test")
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(OpenMLHashException) as exc_info:
            raise OpenMLHashException("Computed hash differs from server hash")
        
        assert "hash" in exc_info.value.message.lower()

    def test_detailed_message(self):
        """Test with detailed hash information."""
        expected = "abc123"
        actual = "def456"
        error = OpenMLHashException(
            f"Hash mismatch: expected {expected}, got {actual}"
        )
        
        assert expected in error.message
        assert actual in error.message


class TestOpenMLPrivateDatasetError:
    """Test OpenMLPrivateDatasetError exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLPrivateDatasetError("Access denied")
        assert error.message == "Access denied"

    def test_inheritance(self):
        """Test inheritance from PyOpenMLError."""
        error = OpenMLPrivateDatasetError("Test")
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(OpenMLPrivateDatasetError) as exc_info:
            raise OpenMLPrivateDatasetError("You do not have access to dataset 999")
        
        assert "dataset 999" in exc_info.value.message

    def test_typical_use_case(self):
        """Test typical use case."""
        dataset_id = 123
        error = OpenMLPrivateDatasetError(
            f"User has no rights to access dataset {dataset_id}"
        )
        
        assert str(dataset_id) in error.message


class TestOpenMLRunsExistError:
    """Test OpenMLRunsExistError exception class."""

    def test_init_with_single_run_id(self):
        """Test initialization with single run ID."""
        run_ids = {123}
        error = OpenMLRunsExistError(run_ids, "Run already exists")
        
        assert error.run_ids == {123}
        assert error.message == "Run already exists"

    def test_init_with_multiple_run_ids(self):
        """Test initialization with multiple run IDs."""
        run_ids = {1, 2, 3, 4, 5}
        error = OpenMLRunsExistError(run_ids, "Multiple runs exist")
        
        assert error.run_ids == {1, 2, 3, 4, 5}
        assert len(error.run_ids) == 5

    def test_empty_run_ids_raises_error(self):
        """Test that empty run_ids set raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            OpenMLRunsExistError(set(), "Empty set")

    def test_inheritance(self):
        """Test inheritance from PyOpenMLError."""
        error = OpenMLRunsExistError({1}, "Test")
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        run_ids = {100, 200, 300}
        with pytest.raises(OpenMLRunsExistError) as exc_info:
            raise OpenMLRunsExistError(run_ids, "Duplicate runs detected")
        
        assert exc_info.value.run_ids == run_ids
        assert "Duplicate runs" in exc_info.value.message

    def test_run_ids_attribute_accessible(self):
        """Test that run_ids attribute is accessible."""
        run_ids = {42, 43}
        error = OpenMLRunsExistError(run_ids, "Runs exist")
        
        assert hasattr(error, "run_ids")
        assert 42 in error.run_ids
        assert 43 in error.run_ids

    def test_run_ids_immutable_set(self):
        """Test that run_ids is a set."""
        run_ids = {10, 20, 30}
        error = OpenMLRunsExistError(run_ids, "Test")
        
        assert isinstance(error.run_ids, set)


class TestOpenMLNotAuthorizedError:
    """Test OpenMLNotAuthorizedError exception class."""

    def test_init(self):
        """Test initialization."""
        error = OpenMLNotAuthorizedError("Not authorized")
        assert error.message == "Not authorized"

    def test_inheritance(self):
        """Test inheritance from OpenMLServerError."""
        error = OpenMLNotAuthorizedError("Test")
        assert isinstance(error, OpenMLServerError)
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(OpenMLNotAuthorizedError) as exc_info:
            raise OpenMLNotAuthorizedError("User not authorized to delete flow")
        
        assert "not authorized" in exc_info.value.message.lower()

    def test_typical_use_case(self):
        """Test typical authorization error."""
        action = "delete dataset"
        error = OpenMLNotAuthorizedError(f"User is not authorized to {action}")
        
        assert action in error.message


class TestObjectNotPublishedError:
    """Test ObjectNotPublishedError exception class."""

    def test_init(self):
        """Test initialization."""
        error = ObjectNotPublishedError("Object not published")
        assert error.message == "Object not published"

    def test_inheritance(self):
        """Test inheritance from PyOpenMLError."""
        error = ObjectNotPublishedError("Test")
        assert isinstance(error, PyOpenMLError)

    def test_raise_and_catch(self):
        """Test raising and catching."""
        with pytest.raises(ObjectNotPublishedError) as exc_info:
            raise ObjectNotPublishedError("Flow must be published before use")
        
        assert "published" in exc_info.value.message.lower()

    def test_typical_use_case(self):
        """Test typical use case."""
        entity_type = "Flow"
        error = ObjectNotPublishedError(
            f"{entity_type} has not been published to the server yet"
        )
        
        assert "Flow" in error.message


class TestExceptionHierarchy:
    """Test exception hierarchy and relationships."""

    def test_all_inherit_from_pyopenmlerror(self):
        """Test that all custom exceptions inherit from PyOpenMLError."""
        exceptions = [
            OpenMLServerError("test"),
            OpenMLServerException("test"),
            OpenMLServerNoResult("test"),
            OpenMLCacheException("test"),
            OpenMLHashException("test"),
            OpenMLPrivateDatasetError("test"),
            OpenMLRunsExistError({1}, "test"),
            OpenMLNotAuthorizedError("test"),
            ObjectNotPublishedError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, PyOpenMLError)
            assert isinstance(exc, Exception)

    def test_server_exception_hierarchy(self):
        """Test server exception hierarchy."""
        exc = OpenMLServerNoResult("test")
        
        assert isinstance(exc, OpenMLServerNoResult)
        assert isinstance(exc, OpenMLServerException)
        assert isinstance(exc, OpenMLServerError)
        assert isinstance(exc, PyOpenMLError)

    def test_catch_base_exception(self):
        """Test catching specific exception with base class."""
        with pytest.raises(PyOpenMLError):
            raise OpenMLHashException("Hash error")
        
        with pytest.raises(OpenMLServerError):
            raise OpenMLNotAuthorizedError("Not authorized")


class TestExceptionUsagePatterns:
    """Test common usage patterns for exceptions."""

    def test_exception_in_try_except(self):
        """Test exception handling in try-except block."""
        try:
            raise OpenMLCacheException("Cache miss")
        except OpenMLCacheException as e:
            assert "Cache miss" in e.message
        else:
            pytest.fail("Exception should have been raised")

    def test_multiple_exception_types(self):
        """Test catching multiple exception types."""
        def raise_random_exception(exc_type):
            if exc_type == "cache":
                raise OpenMLCacheException("Cache error")
            elif exc_type == "hash":
                raise OpenMLHashException("Hash error")
            else:
                raise PyOpenMLError("Generic error")
        
        # Test each exception type
        with pytest.raises(OpenMLCacheException):
            raise_random_exception("cache")
        
        with pytest.raises(OpenMLHashException):
            raise_random_exception("hash")
        
        with pytest.raises(PyOpenMLError):
            raise_random_exception("other")

    def test_exception_re_raising(self):
        """Test re-raising exceptions."""
        with pytest.raises(OpenMLServerException) as exc_info:
            try:
                raise OpenMLServerException("Original", code=500)
            except OpenMLServerException:
                raise
        
        assert exc_info.value.code == 500

    def test_exception_chaining(self):
        """Test exception chaining with 'from'."""
        original = ValueError("Original error")
        
        with pytest.raises(PyOpenMLError) as exc_info:
            try:
                raise original
            except ValueError as e:
                raise PyOpenMLError("Wrapped error") from e
        
        assert exc_info.value.__cause__ == original


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_message(self):
        """Test exception with very long message."""
        long_message = "Error: " + "x" * 10000
        error = PyOpenMLError(long_message)
        
        assert len(error.message) > 10000
        assert error.message.startswith("Error:")

    def test_unicode_in_message(self):
        """Test exception with unicode characters."""
        message = "Error: データセット not found 🔍"
        error = PyOpenMLError(message)
        
        assert error.message == message

    def test_newlines_in_message(self):
        """Test exception with newlines in message."""
        message = "Error occurred:\nLine 1\nLine 2\nLine 3"
        error = PyOpenMLError(message)
        
        assert "\n" in error.message
        assert error.message.count("\n") == 3

    def test_server_exception_with_zero_code(self):
        """Test ServerException with code 0."""
        error = OpenMLServerException("Error", code=0)
        assert error.code == 0

    def test_server_exception_with_negative_code(self):
        """Test ServerException with negative code."""
        error = OpenMLServerException("Error", code=-1)
        assert error.code == -1

    def test_runs_exist_error_with_large_set(self):
        """Test RunsExistError with large set of IDs."""
        large_set = set(range(1, 10001))  # 10000 run IDs
        error = OpenMLRunsExistError(large_set, "Many runs exist")
        
        assert len(error.run_ids) == 10000

    def test_exception_equality(self):
        """Test exception equality comparison."""
        error1 = PyOpenMLError("Same message")
        error2 = PyOpenMLError("Same message")
        
        # Exceptions are not equal even with same message
        assert error1 is not error2
        assert error1.message == error2.message
