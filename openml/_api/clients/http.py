from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urljoin, urlparse

import requests
from requests import Response

from openml.__version__ import __version__

if TYPE_CHECKING:
    from openml._api.config import DelayMethod


class HTTPCache:
    def __init__(self, *, path: Path, ttl: int) -> None:
        self.path = path
        self.ttl = ttl

    def get_key(self, url: str, params: dict[str, Any]) -> str:
        parsed_url = urlparse(url)
        netloc_parts = parsed_url.netloc.split(".")[::-1]
        path_parts = parsed_url.path.strip("/").split("/")

        filtered_params = {k: v for k, v in params.items() if k != "api_key"}
        params_part = [urlencode(filtered_params)] if filtered_params else []

        return str(Path(*netloc_parts, *path_parts, *params_part))

    def _key_to_path(self, key: str) -> Path:
        return self.path.joinpath(key)

    def load(self, key: str) -> Response:
        path = self._key_to_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Cache directory not found: {path}")

        meta_path = path / "meta.json"
        headers_path = path / "headers.json"
        body_path = path / "body.bin"

        if not (meta_path.exists() and headers_path.exists() and body_path.exists()):
            raise FileNotFoundError(f"Incomplete cache at {path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        created_at = meta.get("created_at")
        if created_at is None:
            raise ValueError("Cache metadata missing 'created_at'")

        if time.time() - created_at > self.ttl:
            raise TimeoutError(f"Cache expired for {path}")

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

    def save(self, key: str, response: Response) -> None:
        path = self._key_to_path(key)
        path.mkdir(parents=True, exist_ok=True)

        (path / "body.bin").write_bytes(response.content)

        with (path / "headers.json").open("w", encoding="utf-8") as f:
            json.dump(dict(response.headers), f)

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

        with (path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f)


class HTTPClient:
    def __init__(  # noqa: PLR0913
        self,
        *,
        server: str,
        base_url: str,
        api_key: str,
        timeout: int,
        retries: int,
        delay_method: DelayMethod,
        delay_time: int,
        cache: HTTPCache | None = None,
    ) -> None:
        self.server = server
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.delay_method = delay_method
        self.delay_time = delay_time
        self.cache = cache

        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    def request(
        self,
        method: str,
        path: str,
        *,
        use_cache: bool = False,
        use_api_key: bool = False,
        md5_checksum: str | None,
        **request_kwargs: Any,
    ) -> Response:
        url = urljoin(self.server, urljoin(self.base_url, path))

        # prepare params
        params = request_kwargs.pop("params", {}).copy()
        if use_api_key:
            params["api_key"] = self.api_key

        # prepare headers
        headers = request_kwargs.pop("headers", {}).copy()
        headers.update(self.headers)

        timeout = request_kwargs.pop("timeout", self.timeout)

        if use_cache and self.cache is not None:
            cache_key = self.cache.get_key(url, params)
            try:
                return self.cache.load(cache_key)
            except (FileNotFoundError, TimeoutError):
                pass  # cache miss or expired, continue
            except Exception:
                raise  # propagate unexpected cache errors

        response = requests.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            **request_kwargs,
        )

        if md5_checksum is not None:
            self._verify_checksum(response, md5_checksum)

        if use_cache and self.cache is not None:
            self.cache.save(cache_key, response)

        return response

    def _verify_checksum(self, response: Response, md5_checksum: str) -> None:
        # ruff sees hashlib.md5 as insecure
        actual = hashlib.md5(response.content).hexdigest()  # noqa: S324
        if actual != md5_checksum:
            raise ValueError(f"MD5 checksum mismatch: expected {md5_checksum}, got {actual}")

    def get(
        self,
        path: str,
        *,
        use_cache: bool = False,
        use_api_key: bool = False,
        md5_checksum: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        return self.request(
            method="GET",
            path=path,
            use_cache=use_cache,
            use_api_key=use_api_key,
            md5_checksum=md5_checksum,
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

    def download(
        self,
        url: str,
        handler: Callable[[Response, Path, str], Path] | None = None,
        encoding: str = "utf-8",
        file_name: str = "response.txt",
        md5_checksum: str | None = None,
    ) -> Path:
        # TODO(Shrivaths) find better way to get base path
        base = self.cache.path if self.cache is not None else Path("~/.openml/cache")
        file_path = base / "downloads" / urlparse(url).path.lstrip("/") / file_name
        file_path = file_path.expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            return file_path

        response = self.get(url, md5_checksum=md5_checksum)
        if handler is not None:
            return handler(response, file_path, encoding)

        return self._text_handler(response, file_path, encoding)

    def _text_handler(self, response: Response, path: Path, encoding: str) -> Path:
        with path.open("w", encoding=encoding) as f:
            f.write(response.text)
        return path
