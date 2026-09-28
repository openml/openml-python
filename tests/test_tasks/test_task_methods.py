# License: BSD 3-Clause
from __future__ import annotations

from unittest import mock

import pytest
import requests

import openml
import openml._api_calls
from openml.testing import TestBase


# Common methods between tasks
class OpenMLTaskMethodsTest(TestBase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_tagging(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882, download_data=False)

        tag = "test_tag_OpenMLTaskMethodsTest"
        tag_url = openml._api_calls._create_url_from_endpoint("task/tag")
        untag_url = openml._api_calls._create_url_from_endpoint("task/untag")
        expected_data = {"task_id": 1882, "tag": tag, "api_key": openml.config.apikey}

        def _make_response(content: str) -> requests.Response:
            response = requests.Response()
            response.status_code = 200
            response._content = content.encode()
            return response

        with mock.patch.object(
            requests.Session,
            "post",
            return_value=_make_response(
                f'<oml:task_tag xmlns:oml="http://openml.org/openml">'
                f"<oml:id>1882</oml:id><oml:tag>{tag}</oml:tag>"
                f"</oml:task_tag>"
            ),
        ) as mock_post:
            task.push_tag(tag)
            mock_post.assert_called_once_with(
                tag_url, data=expected_data, files=None, headers=openml._api_calls._HEADERS
            )

        with mock.patch.object(
            requests.Session,
            "post",
            return_value=_make_response(
                '<oml:task_untag xmlns:oml="http://openml.org/openml">'
                "<oml:id>1882</oml:id>"
                "</oml:task_untag>"
            ),
        ) as mock_post:
            task.remove_tag(tag)
            mock_post.assert_called_once_with(
                untag_url, data=expected_data, files=None, headers=openml._api_calls._HEADERS
            )

    def test_get_train_and_test_split_indices(self):
        openml.config.set_root_cache_directory(self.static_cache_dir)
        task = openml.tasks.get_task(1882)
        train_indices, test_indices = task.get_train_test_split_indices(0, 0)
        assert train_indices[0] == 16
        assert train_indices[-1] == 395
        assert test_indices[0] == 412
        assert test_indices[-1] == 364
        train_indices, test_indices = task.get_train_test_split_indices(2, 2)
        assert train_indices[0] == 237
        assert train_indices[-1] == 681
        assert test_indices[0] == 583
        assert test_indices[-1] == 24
        with pytest.raises(ValueError, match="Fold 10 not known"):
            task.get_train_test_split_indices(10, 0)
        with pytest.raises(ValueError, match="Repeat 10 not known"):
            task.get_train_test_split_indices(0, 10)
