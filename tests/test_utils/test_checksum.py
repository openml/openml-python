import pytest
import openml
from unittest.mock import patch, Mock
from openml.exceptions import OpenMLHashException


def _mock_response():
    mock_response = Mock()
    mock_response.text = "hello"
    mock_response.content = b"hello"
    mock_response.status_code = 200
    mock_response.headers = {"Content-Encoding": "gzip"}  # Required by headers check
    return mock_response


@patch("requests.Session")
def test_checksum_match(Session_class_mock):
    Session_class_mock.return_value.__enter__.return_value.get.return_value = _mock_response()

    with openml.config.overwrite_config_context({"connection_n_retries": 1}): # to avoid retry delays
        openml._api_calls._send_request(
            request_method="get",
            url="/dummy",
            data={},
            md5_checksum="5d41402abc4b2a76b9719d911017c592",
        )


@patch("requests.Session")
def test_checksum_mismatch(Session_class_mock):
    Session_class_mock.return_value.__enter__.return_value.get.return_value = _mock_response()

    with openml.config.overwrite_config_context({"connection_n_retries": 1}): # to avoid retry delays
        with pytest.raises(OpenMLHashException):
            openml._api_calls._send_request(
                request_method="get",
                url="/dummy",
                data={},
                md5_checksum="00000000000000000000000000000000",
            )


@patch("requests.Session")
def test_checksum_skipped_when_flag_off(Session_class_mock):
    Session_class_mock.return_value.__enter__.return_value.get.return_value = _mock_response()

    with openml.config.overwrite_config_context({
        "check_digest": False,
        "connection_n_retries": 1,  # to avoid retry delays
    }):
        # should NOT raise even though checksum mismatches
        openml._api_calls._send_request(
            request_method="get",
            url="/dummy",
            data={},
            md5_checksum="not-a-real-sum",
        )
