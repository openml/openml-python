from __future__ import annotations

import contextlib
import shutil
import urllib
import zipfile
from pathlib import Path

import minio
import requests
from urllib3 import ProxyManager

import openml
from openml.__version__ import __version__
from openml.utils import ProgressBar


class MinIOClient:
    def __init__(self, path: Path) -> None:
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}
        self.path = path

    def _get_path(self, url: str) -> Path:
        parsed_url = urllib.parse.urlparse(url)
        return self.path / "minio" / parsed_url.path.lstrip("/")

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
        destination = self._get_path(source) if destination is None else Path(destination)
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
        destination = self._get_path(source) if destination is None else Path(destination)
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
