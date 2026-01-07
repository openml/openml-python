from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import requests
from requests import Response

from openml.__version__ import __version__

if TYPE_CHECKING:
    from openml._api.config import APIConfig


class HTTPClient:
    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    def _create_url(self, path: str) -> str:
        return self.config.server + self.config.base_url + path

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        url = self._create_url(path)
        return requests.get(url, params=params, headers=self.headers, timeout=10)

    def post(
        self,
        path: str,
        data: Mapping[str, Any] | None = None,
        files: Any = None,
    ) -> Response:
        url = self._create_url(path)
        return requests.post(url, data=data, files=files, headers=self.headers, timeout=10)

    def delete(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        url = self._create_url(path)
        return requests.delete(url, params=params, headers=self.headers, timeout=10)
