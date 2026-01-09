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

    def _get_cache_dir(self, url: str, params: dict[str, Any]) -> Path:
        parsed_url = urlparse(url)
        netloc_parts = parsed_url.netloc.split(".")[::-1]  # reverse domain
        path_parts = parsed_url.path.strip("/").split("/")

        # remove api_key and serialize params if any
        filtered_params = {k: v for k, v in params.items() if k != "api_key"}
        params_part = [urlencode(filtered_params)] if filtered_params else []

        return Path(self.dir).joinpath(*netloc_parts, *path_parts, *params_part)

    def _get_cache_response(self, cache_dir: Path) -> Response:  # noqa: ARG002
        return Response()

    def _set_cache_response(self, cache_dir: Path, response: Response) -> None:  # noqa: ARG002
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

    @property
    def key(self) -> str:
        return self.config.key

    @property
    def timeout(self) -> int:
        return self.config.timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        use_cache: bool = False,
        use_api_key: bool = False,
        **request_kwargs: Any,
    ) -> Response:
        url = urljoin(self.server, urljoin(self.base_url, path))

        params = request_kwargs.pop("params", {})
        params = params.copy()
        if use_api_key:
            params["api_key"] = self.key

        headers = request_kwargs.pop("headers", {})
        headers = headers.copy()
        headers.update(self.headers)

        timeout = request_kwargs.pop("timeout", self.timeout)
        cache_dir = self._get_cache_dir(url, params)

        if use_cache:
            try:
                return self._get_cache_response(cache_dir)
            # TODO: handle ttl expired error
            except Exception:
                raise

        response = requests.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            **request_kwargs,
        )

        if use_cache:
            self._set_cache_response(cache_dir, response)

        return response

    def get(
        self,
        path: str,
        *,
        use_cache: bool = False,
        use_api_key: bool = False,
        **request_kwargs: Any,
    ) -> Response:
        # TODO: remove override when cache is implemented
        use_cache = False
        return self.request(
            method="GET",
            path=path,
            use_cache=use_cache,
            use_api_key=use_api_key,
            **request_kwargs,
        )

    def post(
        self,
        path: str,
        **request_kwargs: Any,
    ) -> Response:
        return self.request(
            method="POST",
            path=path,
            use_cache=False,
            use_api_key=True,
            **request_kwargs,
        )

    def delete(
        self,
        path: str,
        **request_kwargs: Any,
    ) -> Response:
        return self.request(
            method="DELETE",
            path=path,
            use_cache=False,
            use_api_key=True,
            **request_kwargs,
        )
