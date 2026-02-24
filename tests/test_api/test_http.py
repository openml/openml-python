from requests import Response, Request, Session
from unittest.mock import patch
import pytest
from openml.testing import TestBase
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
from openml.enums import APIVersion
from openml.exceptions import OpenMLAuthenticationError
from openml._api import HTTPClient
import openml


class TestHTTPClient(TestBase):
    http_client: HTTPClient

    def setUp(self):
        super().setUp()
        self.http_client = self.http_clients[APIVersion.V1]

    def _prepare_url(self, path: str | None = None) -> str:
        server = self.http_client.server
        return urljoin(server, path)

    def test_cache(self):
        path = "task/31"
        params = {"param1": "value1", "param2": "value2"}

        url = self._prepare_url(path=path)

        parsed_url = urlparse(url)
        netloc_parts = parsed_url.netloc.split(".")[::-1]
        path_parts = parsed_url.path.strip("/").split("/")
        params_key = "&".join([f"{k}={v}" for k, v in params.items()])

        key = self.cache.get_key(url, params)
        expected_key = os.path.join(
            *netloc_parts,
            *path_parts,
            params_key,
        )

        # validate key
        self.assertEqual(key, expected_key)

        # create mock response
        req = Request("GET", url).prepare()
        response = Response()
        response.status_code = 200
        response.url = url
        response.reason = "OK"
        response._content = b"<xml>test</xml>"
        response.headers = {"Content-Type": "text/xml"}
        response.encoding = "utf-8"
        response.request = req
        response.elapsed = type("Elapsed", (), {"total_seconds": lambda x: 0.1})()

        # save to cache
        self.cache.save(key, response)

        # load from cache
        cached_response = self.cache.load(key)

        # validate loaded response
        self.assertEqual(cached_response.status_code, 200)
        self.assertEqual(cached_response.url, url)
        self.assertEqual(cached_response.content, b"<xml>test</xml>")
        self.assertEqual(
            cached_response.headers["Content-Type"], "text/xml"
        )

    @pytest.mark.uses_test_server()
    def test_get(self):
        response = self.http_client.get("task/1")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<oml:task", response.content)

    @pytest.mark.uses_test_server()
    def test_get_with_cache_creates_cache(self):
        response = self.http_client.get("task/1", enable_cache=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.cache.path.exists())

        # verify cache directory structure exists
        cache_key = self.cache.get_key(
            self._prepare_url(path="task/1"),
            {},
        )
        cache_path = self.cache._key_to_path(cache_key)

        self.assertTrue((cache_path / "meta.json").exists())
        self.assertTrue((cache_path / "headers.json").exists())
        self.assertTrue((cache_path / "body.bin").exists())

    @pytest.mark.uses_test_server()
    def test_get_uses_cached_response(self):
        # first request populates cache
        response1 = self.http_client.get("task/1", enable_cache=True)

        # second request should load from cache
        response2 = self.http_client.get("task/1", enable_cache=True)

        self.assertEqual(response1.content, response2.content)
        self.assertEqual(response1.status_code, response2.status_code)

    @pytest.mark.uses_test_server()
    def test_get_refresh_cache(self):
        path = "task/1"

        url = self._prepare_url(path=path)
        key = self.cache.get_key(url, {})
        cache_path = self.cache._key_to_path(key) / "meta.json"

        response1 = self.http_client.get(path, enable_cache=True)
        response1_cache_time_stamp = cache_path.stat().st_mtime

        response2 = self.http_client.get(path, enable_cache=True, refresh_cache=True)
        response2_cache_time_stamp = cache_path.stat().st_mtime

        self.assertNotEqual(response1_cache_time_stamp, response2_cache_time_stamp)
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.content, response2.content)

    @pytest.mark.uses_test_server()
    def test_get_with_api_key(self):
        response = self.http_client.get("task/1", use_api_key=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<oml:task", response.content)

    @pytest.mark.uses_test_server()
    def test_get_without_api_key_raises(self):
        api_key = openml.config.SERVERS[APIVersion.V1]["api_key"]
        openml.config.SERVERS[APIVersion.V1]["api_key"] = None

        with pytest.raises(OpenMLAuthenticationError):
            self.http_client.get("task/1", use_api_key=True)

        openml.config.SERVERS[APIVersion.V1]["api_key"] = api_key

    @pytest.mark.uses_test_server()
    def test_download_creates_file(self):
        # small stable resource
        url = self.http_client.server

        path = self.http_client.download(
            url,
            file_name="index.html",
        )

        assert path.exists()
        assert path.is_file()
        assert path.read_text(encoding="utf-8")

    @pytest.mark.uses_test_server()
    def test_download_is_cached_on_disk(self):
        url = self.http_client.server

        path1 = self.http_client.download(
            url,
            file_name="cached.html",
        )
        mtime1 = path1.stat().st_mtime

        # second call should NOT re-download
        path2 = self.http_client.download(
            url,
            file_name="cached.html",
        )
        mtime2 = path2.stat().st_mtime

        assert path1 == path2
        assert mtime1 == mtime2

    @pytest.mark.uses_test_server()
    def test_download_respects_custom_handler(self):
        url = self.http_client.server

        def handler(response, path: Path, encoding: str):
            path.write_text("HANDLED", encoding=encoding)
            return path

        path = self.http_client.download(
            url,
            handler=handler,
            file_name="handled.txt",
        )

        assert path.exists()
        assert path.read_text() == "HANDLED"

    def test_post(self):
        resource_name = "resource"
        resource_files = {"description": """Resource Description File"""}

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200

            self.http_client.post(
                resource_name,
                files=resource_files,
            )

            mock_request.assert_called_once_with(
                method="POST",
                url=urljoin(self.http_client.server, resource_name),
                params={},
                data={'api_key': self.http_client.api_key},
                headers=self.http_client.headers,
                files=resource_files,
            )

    def test_delete(self):
        resource_name = "resource"
        resource_id = 123

        with patch.object(Session, "request") as mock_request:
            mock_request.return_value = Response()
            mock_request.return_value.status_code = 200

            self.http_client.delete(f"{resource_name}/{resource_id}")

            mock_request.assert_called_once_with(
                method="DELETE",
                url=self.http_client.server + self.http_client.base_url + resource_name + "/" + str(resource_id),
                params={'api_key': self.http_client.api_key},
                data={},
                headers=self.http_client.headers,
                files=None,
            )
