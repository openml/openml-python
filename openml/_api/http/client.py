from __future__ import annotations

from typing import Any, Mapping

import requests
from requests import Response

from openml.__version__ import __version__


class HTTPClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        url = f"{self.base_url}/{path}"
        return requests.get(url, params=params, headers=self.headers, timeout=10)

    def post(
        self,
        path: str,
        data: Mapping[str, Any] | None = None,
        files: Any = None,
    ) -> Response:
        url = f"{self.base_url}/{path}"
        return requests.post(url, data=data, files=files, headers=self.headers, timeout=10)

    def delete(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Response:
        url = f"{self.base_url}/{path}"
        return requests.delete(url, params=params, headers=self.headers, timeout=10)
