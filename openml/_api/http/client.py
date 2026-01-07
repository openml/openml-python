from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urljoin, urlparse

import requests
from requests import Response

from openml.__version__ import __version__
from openml._api.config import settings

if TYPE_CHECKING:
    from openml._api.config import APIConfig


class CacheMixin:
    @property
    def dir(self) -> str:
        return settings.cache.dir

    @property
    def ttl(self) -> int:
        return settings.cache.ttl

    def _get_cache_directory(self, url: str, params: dict[str, Any]) -> Path:
        parsed_url = urlparse(url)
        netloc_parts = parsed_url.netloc.split(".")[::-1]  # reverse domain
        path_parts = parsed_url.path.strip("/").split("/")

        # remove api_key and serialize params if any
        filtered_params = {k: v for k, v in params.items() if k != "api_key"}
        params_part = [urlencode(filtered_params)] if filtered_params else []

        return Path(self.dir).joinpath(*netloc_parts, *path_parts, *params_part)

    def _get_cache_response(self, url: str, params: dict[str, Any]) -> Response | None:  # noqa: ARG002
        return None

    def _set_cache_response(self, url: str, params: dict[str, Any], response: Response) -> None:  # noqa: ARG002
        return None


class HTTPClient(CacheMixin):
    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    @property
    def server(self) -> str:
        return self.config.server

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def _create_url(self, path: str) -> Any:
        return urljoin(self.server, urljoin(self.base_url, path))

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        use_api_key: bool = False,
    ) -> Response:
        url = self._create_url(path)
        params = dict(params) if params is not None else {}

        if use_api_key:
            params["api_key"] = self.config.key

        if use_cache:
            response = self._get_cache_response(url, params)
            if response:
                return response

        response = requests.get(url, params=params, headers=self.headers, timeout=10)

        if use_cache:
            self._set_cache_response(url, params, response)

        return response

    def post(
        self,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        files: Any = None,
    ) -> Response:
        url = self._create_url(path)
        return requests.post(url, data=data, files=files, headers=self.headers, timeout=10)

    def delete(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Response:
        url = self._create_url(path)
        return requests.delete(url, params=params, headers=self.headers, timeout=10)
