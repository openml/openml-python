from __future__ import annotations

import contextlib
import shutil
import urllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client as BotoS3Client

import openml
from openml.utils import ProgressBar


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
    ) -> tuple[str, str, str]:
        """Split an object storage URL into endpoint, bucket, and object key.

        The optional mount prefix is kept on the
        endpoint URL so path-style S3 requests still go through the reverse
        proxy. Production URLs have no mount prefix.

        Parameters
        ----------
        source : str
            URL to a file in an S3 bucket, for example:
            ``http://localhost:8000/s3/datasets/0000/0061/dataset_61.pq``

        Returns
        -------
        tuple[str, str, str]
            Endpoint URL, bucket name, object key.

        Raises
        ------
        ValueError
            If the URL does not contain both a bucket and an object key.
        """
        # Parse the URL into sections
        parsed_url = urllib.parse.urlparse(source)
        parts = [part for part in parsed_url.path.split("/") if part]
        if len(parts) < 2:  # Bucket and key minimum
            raise ValueError(f"Object storage URL must contain a bucket and object key: {source!r}")

        # Handle the endpoint
        endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parts[0] == "s3":  # URL has mount in first index
            endpoint += f"/{parts.pop(0)}"

        # Handle the bucket and object key
        bucket = parts[0]
        object_key = "/".join(parts[1:])
        if not object_key:
            raise ValueError(f"Object storage URL must contain an object key: {source!r}.")
        return endpoint, bucket, object_key

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
        return boto3.client(
            "s3",
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
        # Parse the source url and get cache location
        endpoint, bucket, object_key = self._parse_s3_url(source)
        destination = Path(openml.config.get_download_path(source))

        # Ensure the cache does not already exist
        if destination.is_dir():
            destination = Path(destination, object_key.rsplit("/", 1)[-1])
        if destination.is_file() and not exists_ok:
            raise FileExistsError(f"File already exists in {destination}.")

        # Ensure download location exists
        destination = destination.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Set up the S3 client with the proxy
        if proxy == "auto":
            resolved_proxies = requests.utils.get_environ_proxies(source)
            proxy = requests.utils.select_proxy(source, resolved_proxies)
        client = self._make_s3_client(endpoint_url=endpoint, proxy=proxy)

        # Get the download object
        try:
            # Create a progress bar if required
            callback = None
            if openml.config.show_progress:
                total_length = client.head_object(Bucket=bucket, Key=object_key)["ContentLength"]
                progress = ProgressBar()
                progress.set_meta(object_key, total_length)
                callback = progress.update
            # Download object
            client.download_file(
                Bucket=bucket,
                Key=object_key,
                Filename=str(destination),
                Callback=callback,
            )
            # Unzip if compressed
            if destination.is_file() and destination.suffix == ".zip":
                with zipfile.ZipFile(destination, "r") as zip_ref:
                    zip_ref.extractall(destination.parent)

        # Error handling and return
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
        endpoint, bucket, object_key = self._parse_s3_url(source)
        destination = Path(openml.config.get_download_path(source))
        if destination.suffix:
            destination = destination.parent
        prefix = object_key.rsplit("/", 1)[0]

        # Create S3 client and paginator
        client = self._make_s3_client(endpoint_url=endpoint)
        paginator = client.get_paginator("list_objects_v2")  # PageSize caps at 1_000

        # Iterate over the response pages
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for file_object in page.get("Contents", []):
                obj_key = file_object["Key"]
                etag = (file_object.get("ETag") or "").strip('"')

                # Check if the object was already downloaded
                marker = destination / etag if etag else None
                if marker is not None and marker.exists():
                    continue

                # Server archive changed?
                object_source = f"{endpoint}/{bucket}/{obj_key}"
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
