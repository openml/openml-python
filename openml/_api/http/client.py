from __future__ import annotations

import json
import time
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

    def _get_cache_response(self, cache_dir: Path) -> Response:
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        meta_path = cache_dir / "meta.json"
        headers_path = cache_dir / "headers.json"
        body_path = cache_dir / "body.bin"

        if not (meta_path.exists() and headers_path.exists() and body_path.exists()):
            raise FileNotFoundError(f"Incomplete cache at {cache_dir}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        created_at = meta.get("created_at")
        if created_at is None:
            raise ValueError("Cache metadata missing 'created_at'")

        if time.time() - created_at > self.ttl:
            raise TimeoutError(f"Cache expired for {cache_dir}")

        with headers_path.open("r", encoding="utf-8") as f:
            headers = json.load(f)

        body = body_path.read_bytes()

        response = Response()
        response.status_code = meta["status_code"]
        response.url = meta["url"]
        response.reason = meta["reason"]
        response.headers = headers
        response._content = body
        response.encoding = meta["encoding"]

        return response

    def _set_cache_response(self, cache_dir: Path, response: Response) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # body
        (cache_dir / "body.bin").write_bytes(response.content)

        # headers
        with (cache_dir / "headers.json").open("w", encoding="utf-8") as f:
            json.dump(dict(response.headers), f)

        # meta
        meta = {
            "status_code": response.status_code,
            "url": response.url,
            "reason": response.reason,
            "encoding": response.encoding,
            "elapsed": response.elapsed.total_seconds(),
            "created_at": time.time(),
            "request": {
                "method": response.request.method if response.request else None,
                "url": response.request.url if response.request else None,
                "headers": dict(response.request.headers) if response.request else None,
                "body": response.request.body if response.request else None,
            },
        }

        with (cache_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f)


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
            except FileNotFoundError:
                pass
            except TimeoutError:
                pass
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
