from __future__ import annotations

import contextlib
import shutil
import urllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client as BotoS3Client

import openml
from openml.utils import ProgressBar


class ParsedS3URL(NamedTuple):
    endpoint: str
    bucket: str
    object_key: str


class S3Client:
    """
    Lightweight client configuration for interacting with an S3-compatible
    object storage service.

    This class stores basic configuration such as a base filesystem path and
    default HTTP headers. It is intended to be extended with actual request
    or storage logic elsewhere.
    """

    @staticmethod
    def _parse_s3_url(
        source: str,
        path_prefix: str = "s3",
    ) -> ParsedS3URL:
        """Split an object storage URL into endpoint, bucket, and object key.

        The optional path prefix is kept on the
        endpoint URL so path-style S3 requests still go through the reverse
        proxy. Production URLs have no path prefix.

        Parameters
        ----------
        source : str
            URL to a file in an S3 bucket, for example:
            ``http://localhost:8000/s3/datasets/0000/0061/dataset_61.pq``
        path_prefix : str, optional (default = "s3")
            Optional storage path mount.

        Returns
        -------
        ParsedS3URL
            Object containing attributes endpoint URL, bucket name, object key.

        Raises
        ------
        ValueError
            If the URL does not contain both a bucket and an object key.
        """
        parsed_url = urllib.parse.urlparse(source)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid object storage URL: {source!r}.")

        endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path = parsed_url.path.strip("/")
        if path.startswith(f"{path_prefix}/"):
            endpoint += f"/{path_prefix}"
            path = path.removeprefix(f"{path_prefix}/")

        bucket, _, object_key = path.partition("/")
        if not bucket or not object_key:
            raise ValueError(
                f"Object storage URL must contain a bucket and object key: {source!r}."
            )
        return ParsedS3URL(endpoint, bucket, object_key)

    @staticmethod
    def _make_s3_client(
        endpoint_url: str,
        proxy: str | None = None,
    ) -> BotoS3Client:
        """Create an anonymous S3 client for a public object-storage endpoint.

        Parameters
        ----------
        endpoint_url : str
            Endpoint URL, including any reverse-proxy mount prefix.
        proxy : str or None, optional (default=None)
            Proxy URL used for both HTTP and HTTPS requests. ``None`` means
            a direct connection.

        Returns
        -------
        botocore.client.BotoS3Client
            Configured S3 client.
        """
        proxies = {"http": proxy, "https": proxy} if proxy else None
        config = Config(
            signature_version=UNSIGNED,  # Public bucket
            s3={"addressing_style": "path"},  # URL = endpoint/bucket/key for example
            proxies=proxies,  # Optional HTTP proxy
            user_agent_extra=openml.config._HEADERS["user-agent"],  # Keep openml-python/<version>
        )
        # Client requies a region_name but it has no meaning for OpenML,
        # eu-central-1 was chosen for no particular reason.
        return boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url,
            region_name="eu-central-1",
            config=config,
        )

    def download_file(
        self,
        source: str,
        exists_ok: bool = True,  # noqa: FBT002
        proxy: str | None = "auto",
    ) -> Path:
        """Download file ``source`` from an S3 Bucket and store it at ``destination``.

        Parameters
        ----------
        source : str
            URL to a file in an S3 bucket.
        exists_ok : bool, optional (default=True)
            If False, raise FileExists if a file already exists in ``destination``.
        proxy: str, optional (default = "auto")
            Proxy server to use. ``"auto"`` resolves it from the environment.
        """
        parsed_url = self._parse_s3_url(source)
        destination = Path(openml.config.get_download_path(source))

        if destination.is_dir():
            destination = Path(destination, parsed_url.object_key.rsplit("/", 1)[-1])
        if destination.is_file() and not exists_ok:
            raise FileExistsError(f"File already exists in {destination}.")

        destination = destination.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        if proxy == "auto":
            resolved_proxies = requests.utils.get_environ_proxies(source)
            proxy = requests.utils.select_proxy(source, resolved_proxies)
        client = self._make_s3_client(endpoint_url=parsed_url.endpoint, proxy=proxy)

        try:
            callback = None
            if openml.config.show_progress:
                total_length = client.head_object(
                    Bucket=parsed_url.bucket, Key=parsed_url.object_key
                )["ContentLength"]
                progress = ProgressBar()
                progress.set_meta(parsed_url.object_key, total_length)
                callback = progress.update

            client.download_file(
                Bucket=parsed_url.bucket,
                Key=parsed_url.object_key,
                Filename=str(destination),
                Callback=callback,
            )

            if destination.is_file() and destination.suffix == ".zip":
                with zipfile.ZipFile(destination, "r") as zip_ref:
                    zip_ref.extractall(destination.parent)

        except ClientError as err:
            error = err.response.get("Error", {})
            status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404 or error.get("Code") in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(f"Object at '{source}' does not exist.") from err
            raise FileNotFoundError("Bucket is private or does not exist.") from err

        except RecursionError as err:
            raise RuntimeError(
                f"Redirect loop while fetching {source!r}. "
                "The endpoint likely redirects HTTP to HTTPS; use an https:// URL."
            ) from err

        return destination

    def download_bucket(self, source: str) -> None:
        """Download files from S3 Bucket ``source`` and store it at ``destination``.

        Does not redownload files which already exist.

        Parameters
        ----------
        source : str
            URL to a S3 bucket.
        """
        # Parse the source information
        parsed_url = self._parse_s3_url(source)
        destination = Path(openml.config.get_download_path(source))
        if destination.suffix:
            destination = destination.parent
        prefix = parsed_url.object_key.rsplit("/", 1)[0] if "/" in parsed_url.object_key else ""

        client = self._make_s3_client(endpoint_url=parsed_url.endpoint)
        # Calls `list_objects_v2`, a method that is defined on the `boto3.client`.
        # It returns some or all (up to 1,000) of the objects in a bucket with each request.
        paginator = client.get_paginator("list_objects_v2")

        # Iterate over the response pages
        for page in paginator.paginate(Bucket=parsed_url.bucket, Prefix=prefix):
            for file_object in page.get("Contents", []):
                obj_key = file_object["Key"]
                etag = file_object.get("ETag", "").strip('"')

                # We avoid redownloading files already in our cache by disallowing
                # the client's download_file to write to a file that already exists.
                # However, downloaded archives are automatically removed from disk
                # after downloading to reduce disk space usage. This means that we can
                # not rely on the presence of an archive file to avoid redownloading it.
                # Instead, for archives only, we put a marker in the cache directory
                # indicating the successful download and extraction of the archive.
                # We create the marker using the `etag` which is a hash of the file,
                # this means that if the archive itself changes (but not the filename)
                # we can still detect an outdated download. So we use the etag-based
                # marker to determine whether or not to download archives.
                marker = destination / etag if etag else None
                if marker is not None and marker.exists():
                    continue

                # Server archive changed?
                object_source = f"{parsed_url.endpoint}/{parsed_url.bucket}/{obj_key}"
                file_destination = Path(openml.config.get_download_path(object_source))
                if (file_destination.parent / file_destination.stem).exists():
                    shutil.rmtree(file_destination.parent / file_destination.stem)

                # No need to redownload file
                with contextlib.suppress(FileExistsError):
                    self.download_file(source=object_source, exists_ok=False)

                # Clean up unzipped file then leave marker
                if file_destination.is_file() and file_destination.suffix == ".zip":
                    file_destination.unlink()
                    if marker is not None:
                        marker.touch()
