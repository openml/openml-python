from __future__ import annotations

import contextlib
import shutil
import urllib
import urllib.parse
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urljoin, urlparse

import minio
import requests
from requests import Response
from urllib3 import ProxyManager

from openml.__version__ import __version__
from openml._api.config import settings

if TYPE_CHECKING:
    from openml._api.config import APIConfig

import openml.config
from openml.utils import ProgressBar


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

    def download(
        self,
        url: str,
        handler: Callable[[Response, Path, str], Path] | None = None,
        encoding: str = "utf-8",
    ) -> Path:
        response = self.get(url)
        dir_path = self._get_cache_dir(url, {})
        dir_path = dir_path.expanduser()
        if handler is not None:
            return handler(response, dir_path, encoding)

        return self._text_handler(response, dir_path, encoding)

    def _text_handler(self, response: Response, path: Path, encoding: str) -> Path:
        if path.is_dir():
            path = path / "response.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=encoding) as f:
            f.write(response.text)
        return path


class MinIOClient(CacheMixin):
    def __init__(self) -> None:
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    def download_minio_file(
        self,
        source: str,
        destination: str | Path | None = None,
        exists_ok: bool = True,  # noqa: FBT002
        proxy: str | None = "auto",
    ) -> Path:
        """Download file ``source`` from a MinIO Bucket and store it at ``destination``.

        Parameters
        ----------
        source : str
            URL to a file in a MinIO bucket.
        destination : str | Path
            Path to store the file to, if a directory is provided the original filename is used.
        exists_ok : bool, optional (default=True)
            If False, raise FileExists if a file already exists in ``destination``.
        proxy: str, optional (default = "auto")
            The proxy server to use. By default it's "auto" which uses ``requests`` to
            automatically find the proxy to use. Pass None or the environment variable
            ``no_proxy="*"`` to disable proxies.
        """
        destination = self._get_cache_dir(source, {}) if destination is None else Path(destination)
        parsed_url = urllib.parse.urlparse(source)

        # expect path format: /BUCKET/path/to/file.ext
        bucket, object_name = parsed_url.path[1:].split("/", maxsplit=1)
        if destination.is_dir():
            destination = Path(destination, object_name)
        if destination.is_file() and not exists_ok:
            raise FileExistsError(f"File already exists in {destination}.")

        destination = destination.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        if proxy == "auto":
            resolved_proxies = requests.utils.get_environ_proxies(parsed_url.geturl())
            proxy = requests.utils.select_proxy(parsed_url.geturl(), resolved_proxies)  # type: ignore

        proxy_client = ProxyManager(proxy) if proxy else None

        client = minio.Minio(endpoint=parsed_url.netloc, secure=False, http_client=proxy_client)
        try:
            client.fget_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=str(destination),
                progress=ProgressBar() if openml.config.show_progress else None,
                request_headers=self.headers,
            )
            if destination.is_file() and destination.suffix == ".zip":
                with zipfile.ZipFile(destination, "r") as zip_ref:
                    zip_ref.extractall(destination.parent)

        except minio.error.S3Error as e:
            if e.message is not None and e.message.startswith("Object does not exist"):
                raise FileNotFoundError(f"Object at '{source}' does not exist.") from e
            # e.g. permission error, or a bucket does not exist (which is also interpreted as a
            # permission error on minio level).
            raise FileNotFoundError("Bucket does not exist or is private.") from e

        return destination

    def download_minio_bucket(self, source: str, destination: str | Path | None = None) -> None:
        """Download file ``source`` from a MinIO Bucket and store it at ``destination``.

        Does not redownload files which already exist.

        Parameters
        ----------
        source : str
            URL to a MinIO bucket.
        destination : str | Path
            Path to a directory to store the bucket content in.
        """
        destination = self._get_cache_dir(source, {}) if destination is None else Path(destination)
        parsed_url = urllib.parse.urlparse(source)
        if destination.suffix:
            destination = destination.parent
        # expect path format: /BUCKET/path/to/file.ext
        _, bucket, *prefixes, _ = parsed_url.path.split("/")
        prefix = "/".join(prefixes)

        client = minio.Minio(endpoint=parsed_url.netloc, secure=False)

        for file_object in client.list_objects(bucket, prefix=prefix, recursive=True):
            if file_object.object_name is None:
                raise ValueError(f"Object name is None for object {file_object!r}")
            if file_object.etag is None:
                raise ValueError(f"Object etag is None for object {file_object!r}")

            marker = destination / file_object.etag
            if marker.exists():
                continue

            file_destination = destination / file_object.object_name.rsplit("/", 1)[1]
            if (file_destination.parent / file_destination.stem).exists():
                # Marker is missing but archive exists means the server archive changed
                # force a refresh
                shutil.rmtree(file_destination.parent / file_destination.stem)

            with contextlib.suppress(FileExistsError):
                self.download_minio_file(
                    source=source.rsplit("/", 1)[0]
                    + "/"
                    + file_object.object_name.rsplit("/", 1)[1],
                    destination=file_destination,
                    exists_ok=False,
                )

            if file_destination.is_file() and file_destination.suffix == ".zip":
                file_destination.unlink()
                marker.touch()
