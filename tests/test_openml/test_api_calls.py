from __future__ import annotations

import unittest.mock

import pytest

import openml
import openml.testing


class TestConfig(openml.testing.TestBase):
    def test_too_long_uri(self):
        with pytest.raises(openml.exceptions.OpenMLServerError, match="URI too long!"):
            openml.datasets.list_datasets(data_id=list(range(10000)), output_format="dataframe")

    @unittest.mock.patch("time.sleep")
    @unittest.mock.patch("requests.Session")
    def test_retry_on_database_error(self, Session_class_mock, _):
        response_mock = unittest.mock.Mock()
        response_mock.text = (
            "<oml:error>\n"
            "<oml:code>107</oml:code>"
            "<oml:message>Database connection error. "
            "Usually due to high server load. "
            "Please wait for N seconds and try again.</oml:message>\n"
            "</oml:error>"
        )
        Session_class_mock.return_value.__enter__.return_value.get.return_value = response_mock
        with pytest.raises(openml.exceptions.OpenMLServerException, match="/abc returned code 107"):
            openml._api_calls._send_request("get", "/abc", {})

        assert Session_class_mock.return_value.__enter__.return_value.get.call_count == 20
