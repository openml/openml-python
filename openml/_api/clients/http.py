from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import xml
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode, urljoin, urlparse

import requests
import xmltodict
from requests import Response

import openml
from openml.enums import APIVersion, RetryPolicy
from openml.exceptions import (
    OpenMLAuthenticationError,
    OpenMLHashException,
    OpenMLServerError,
    OpenMLServerException,
    OpenMLServerNoResult,
)


class HTTPCache:
    """
    Filesystem-based cache for HTTP responses.

    This class stores HTTP responses on disk using a structured directory layout
    derived from the request URL and parameters. Each cached response consists of
    three files: metadata (``meta.json``), headers (``headers.json``), and the raw
    body (``body.bin``).

    Notes
    -----
    The cache key is derived from the URL (domain and path components) and query
    parameters, excluding the ``api_key`` parameter.
    """

    @property
    def path(self) -> Path:
        return Path(openml.config.get_cache_directory())

    def get_key(self, url: str, params: dict[str, Any]) -> str:
        """
        Generate a filesystem-safe cache key for a request.

        The key is constructed from the reversed domain components, URL path
        segments, and URL-encoded query parameters (excluding ``api_key``).

        Parameters
        ----------
        url : str
            The full request URL.
        params : dict of str to Any
            Query parameters associated with the request.

        Returns
        -------
        str
            A relative path string representing the cache key.
        """
        parsed_url = urlparse(url)
        netloc_parts = parsed_url.netloc.split(".")[::-1]
        path_parts = parsed_url.path.strip("/").split("/")

        filtered_params = {k: v for k, v in params.items() if k != "api_key"}
        params_part = [urlencode(filtered_params)] if filtered_params else []

        return str(Path(*netloc_parts, *path_parts, *params_part))

    def _key_to_path(self, key: str) -> Path:
        """
        Convert a cache key into an absolute filesystem path.

        Parameters
        ----------
        key : str
            Cache key as returned by :meth:`get_key`.

        Returns
        -------
        pathlib.Path
            Absolute path corresponding to the cache entry.
        """
        return self.path.joinpath(key)

    def load(self, key: str) -> Response:
        """
        Load a cached HTTP response from disk.

        Parameters
        ----------
        key : str
            Cache key identifying the stored response.

        Returns
        -------
        requests.Response
            Reconstructed response object with status code, headers, body, and metadata.

        Raises
        ------
        FileNotFoundError
            If the cache entry or required files are missing.
        ValueError
            If required metadata is missing or malformed.
        """
        path = self._key_to_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Cache entry not found: {path}")

        meta_path = path / "meta.json"
        headers_path = path / "headers.json"
        body_path = path / "body.bin"

        if not (meta_path.exists() and headers_path.exists() and body_path.exists()):
            raise FileNotFoundError(f"Incomplete cache at {path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

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
        """
        Persist an HTTP response to disk.

        Parameters
        ----------
        key : str
            Cache key identifying where to store the response.
        response : requests.Response
            Response object to cache.

        Notes
        -----
        The response body is stored as binary data. Headers and metadata
        (status code, URL, reason, encoding, elapsed time, request info, and
        creation timestamp) are stored as JSON.
        """
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
    """
    HTTP client for interacting with the OpenML API.

    This client supports configurable retry policies, optional filesystem
    caching, API key authentication, and response validation including
    checksum verification.

    Parameters
    ----------
    api_version : APIVersion
        Backend API Version.
    """

    def __init__(
        self,
        *,
        api_version: APIVersion,
    ) -> None:
        self.api_version = api_version

        self.cache = HTTPCache()

    @property
    def server(self) -> str:
        server = openml.config.servers[self.api_version]["server"]
        if server is None:
            servers_repr = {k.value: v for k, v in openml.config.servers.items()}
            raise ValueError(
                f'server found to be None for api_version="{self.api_version}" in {servers_repr}'
            )
        return cast("str", server)

    @property
    def api_key(self) -> str | None:
        return cast("str | None", openml.config.servers[self.api_version]["apikey"])

    @property
    def retries(self) -> int:
        return cast("int", openml.config.connection_n_retries)

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy.HUMAN if openml.config.retry_policy == "human" else RetryPolicy.ROBOT

    @property
    def retry_func(self) -> Callable:
        return self._human_delay if self.retry_policy == RetryPolicy.HUMAN else self._robot_delay

    def _robot_delay(self, n: int) -> float:
        """
        Compute delay for automated retry policy.

        Parameters
        ----------
        n : int
            Current retry attempt number (1-based).

        Returns
        -------
        float
            Number of seconds to wait before the next retry.

        Notes
        -----
        Uses a sigmoid-based growth curve with Gaussian noise to gradually
        increase waiting time.
        """
        wait = (1 / (1 + math.exp(-(n * 0.5 - 4)))) * 60
        variation = random.gauss(0, wait / 10)
        return max(1.0, wait + variation)

    def _human_delay(self, n: int) -> float:
        """
        Compute delay for human-like retry policy.

        Parameters
        ----------
        n : int
            Current retry attempt number (1-based).

        Returns
        -------
        float
            Number of seconds to wait before the next retry.
        """
        return max(1.0, n)

    def _parse_exception_response(
        self,
        response: Response,
    ) -> tuple[int | None, str]:
        """
        Parse an error response returned by the server.

        Parameters
        ----------
        response : requests.Response
            HTTP response containing error details in JSON or XML format.

        Returns
        -------
        tuple of (int or None, str)
            Parsed error code and combined error message. The code may be
            ``None`` if unavailable.
        """
        content_type = response.headers.get("Content-Type", "").lower()

        if "application/json" in content_type:
            server_exception = response.json()
            server_error = server_exception["detail"]
            code = server_error.get("code")
            message = server_error.get("message")
            additional_information = server_error.get("additional_information")
        else:
            server_exception = xmltodict.parse(response.text)
            server_error = server_exception["oml:error"]
            code = server_error.get("oml:code")
            message = server_error.get("oml:message")
            additional_information = server_error.get("oml:additional_information")

        if code is not None:
            code = int(code)

        if message and additional_information:
            full_message = f"{message} - {additional_information}"
        else:
            full_message = message or additional_information or ""

        return code, full_message

    def _raise_code_specific_error(
        self,
        code: int,
        message: str,
        url: str,
        files: Mapping[str, Any] | None,
    ) -> None:
        """
        Raise specialized exceptions based on OpenML error codes.

        Parameters
        ----------
        code : int
            Server-provided error code.
        message : str
            Parsed error message.
        url : str
            Request URL associated with the error.
        files : Mapping of str to Any or None
            Files sent with the request, if any.

        Raises
        ------
        OpenMLServerNoResult
            If the error indicates a missing resource.
        OpenMLNotAuthorizedError
            If authentication is required or invalid.
        OpenMLServerException
            For other server-side errors (except retryable database errors).
        """
        if code in [111, 372, 512, 500, 482, 542, 674]:
            # 512 for runs, 372 for datasets, 500 for flows
            # 482 for tasks, 542 for evaluations, 674 for setups
            # 111 for dataset descriptions
            raise OpenMLServerNoResult(code=code, message=message, url=url)

        # 163: failure to validate flow XML (https://www.openml.org/api_docs#!/flow/post_flow)
        if code == 163 and files is not None and "description" in files:
            # file_elements['description'] is the XML file description of the flow
            message = f"\n{files['description']}\n{message}"

        # Propagate all server errors to the calling functions, except
        # for 107 which represents a database connection error.
        # These are typically caused by high server load,
        # which means trying again might resolve the issue.
        # DATABASE_CONNECTION_ERRCODE
        if code != 107:
            raise OpenMLServerException(code=code, message=message, url=url)

    def _validate_response(
        self,
        method: str,
        url: str,
        files: Mapping[str, Any] | None,
        response: Response,
    ) -> Exception | None:
        """
        Validate an HTTP response and determine whether to retry.

        Parameters
        ----------
        method : str
            HTTP method used for the request.
        url : str
            Full request URL.
        files : Mapping of str to Any or None
            Files sent with the request, if any.
        response : requests.Response
            Received HTTP response.

        Returns
        -------
        Exception or None
            ``None`` if the response is valid. Otherwise, an exception
            indicating the error to raise or retry.

        Raises
        ------
        OpenMLServerError
            For unexpected server errors or malformed responses.
        """
        if (
            "Content-Encoding" not in response.headers
            or response.headers["Content-Encoding"] != "gzip"
        ):
            logging.warning(f"Received uncompressed content from OpenML for {url}.")

        if response.status_code == 200:
            return None

        if response.status_code == requests.codes.URI_TOO_LONG:
            raise OpenMLServerError(f"URI too long! ({url})")

        exception: Exception | None = None
        code: int | None = None
        message: str = ""

        try:
            code, message = self._parse_exception_response(response)

        except (requests.exceptions.JSONDecodeError, xml.parsers.expat.ExpatError) as e:
            if method != "GET":
                extra = f"Status code: {response.status_code}\n{response.text}"
                raise OpenMLServerError(
                    f"Unexpected server error when calling {url}. Please contact the "
                    f"developers!\n{extra}"
                ) from e

            exception = e

        except Exception as e:
            # If we failed to parse it out,
            # then something has gone wrong in the body we have sent back
            # from the server and there is little extra information we can capture.
            raise OpenMLServerError(
                f"Unexpected server error when calling {url}. Please contact the developers!\n"
                f"Status code: {response.status_code}\n{response.text}",
            ) from e

        if code is not None:
            self._raise_code_specific_error(
                code=code,
                message=message,
                url=url,
                files=files,
            )

        if exception is None:
            exception = OpenMLServerException(code=code, message=message, url=url)

        return exception

    def __request(  # noqa: PLR0913
        self,
        session: requests.Session,
        method: str,
        url: str,
        params: Mapping[str, Any],
        data: Mapping[str, Any],
        headers: Mapping[str, str],
        files: Mapping[str, Any] | None,
        **request_kwargs: Any,
    ) -> tuple[Response | None, Exception | None]:
        """
        Execute a single HTTP request attempt.

        Parameters
        ----------
        session : requests.Session
            Active session used to send the request.
        method : str
            HTTP method (e.g., ``GET``, ``POST``).
        url : str
            Full request URL.
        params : Mapping of str to Any
            Query parameters.
        data : Mapping of str to Any
            Request body data.
        headers : Mapping of str to str
            HTTP headers.
        files : Mapping of str to Any or None
            Files to upload.
        **request_kwargs : Any
            Additional arguments forwarded to ``requests.Session.request``.

        Returns
        -------
        tuple of (requests.Response or None, Exception or None)
            Response and potential retry exception.
        """
        exception: Exception | None = None
        response: Response | None = None

        try:
            response = session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=headers,
                files=files,
                **request_kwargs,
            )
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
        ) as e:
            exception = e

        if response is not None:
            exception = self._validate_response(
                method=method,
                url=url,
                files=files,
                response=response,
            )

        return response, exception

    def _request(  # noqa: PLR0913, C901
        self,
        method: str,
        path: str,
        *,
        enable_cache: bool = False,
        refresh_cache: bool = False,
        use_api_key: bool = False,
        md5_checksum: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """
        Send an HTTP request with retry, caching, and validation support.

        Parameters
        ----------
        method : str
            HTTP method to use.
        path : str
            API path relative to the base URL.
        enable_cache : bool, optional
            Whether to load/store response from cache.
        refresh_cache : bool, optional
            Only used when `enable_cache=True`. If True, ignore any existing
            cached response and overwrite it with a fresh one.
        use_api_key : bool, optional
            Whether to include the API key in query parameters.
        md5_checksum : str or None, optional
            Expected MD5 checksum of the response body.
        **request_kwargs : Any
            Additional arguments passed to the underlying request.

        Returns
        -------
        requests.Response
            Final validated response.

        Raises
        ------
        Exception
            Propagates network, validation, or server exceptions after retries.
        OpenMLHashException
            If checksum verification fails.
        """
        url = urljoin(self.server, path)
        retries = max(1, self.retries)

        params = request_kwargs.pop("params", {}).copy()
        data = request_kwargs.pop("data", {}).copy()

        if use_api_key:
            if self.api_key is None:
                raise OpenMLAuthenticationError(
                    message=(
                        f"The API call {url} requires authentication via an API key. "
                        "Please configure OpenML-Python to use your API "
                        "as described in this example: "
                        "https://openml.github.io/openml-python/latest/examples/Basics/introduction_tutorial/#authentication"
                    )
                )
            params["api_key"] = self.api_key

        if method.upper() in {"POST", "PUT", "PATCH"}:
            data = {**params, **data}
            params = {}

        # prepare headers
        headers = request_kwargs.pop("headers", {}).copy()
        headers.update(openml.config._HEADERS)

        files = request_kwargs.pop("files", None)

        if enable_cache and not refresh_cache:
            cache_key = self.cache.get_key(url, params)
            try:
                return self.cache.load(cache_key)
            except FileNotFoundError:
                pass  # cache miss, continue
            except Exception:
                raise  # propagate unexpected cache errors

        with requests.Session() as session:
            for retry_counter in range(1, retries + 1):
                response, exception = self.__request(
                    session=session,
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    files=files,
                    **request_kwargs,
                )

                # executed successfully
                if exception is None:
                    break
                # tries completed
                if retry_counter >= retries:
                    raise exception

                delay = self.retry_func(retry_counter)
                time.sleep(delay)

        # response is guaranteed to be not `None`
        # otherwise an exception would have been raised before
        response = cast("Response", response)

        if md5_checksum is not None:
            self._verify_checksum(response, md5_checksum)

        if enable_cache:
            cache_key = self.cache.get_key(url, params)
            self.cache.save(cache_key, response)

        return response

    def _verify_checksum(self, response: Response, md5_checksum: str) -> None:
        """
        Verify MD5 checksum of a response body.

        Parameters
        ----------
        response : requests.Response
            HTTP response whose content should be verified.
        md5_checksum : str
            Expected hexadecimal MD5 checksum.

        Raises
        ------
        OpenMLHashException
            If the computed checksum does not match the expected value.
        """
        # ruff sees hashlib.md5 as insecure
        actual = hashlib.md5(response.content).hexdigest()  # noqa: S324
        if actual != md5_checksum:
            raise OpenMLHashException(
                f"Checksum of downloaded file is unequal to the expected checksum {md5_checksum} "
                f"when downloading {response.url}.",
            )

    def get(
        self,
        path: str,
        *,
        enable_cache: bool = False,
        refresh_cache: bool = False,
        use_api_key: bool = False,
        md5_checksum: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """
        Send a GET request.

        Parameters
        ----------
        path : str
            API path relative to the base URL.
        enable_cache : bool, optional
            Whether to use the response cache.
        refresh_cache : bool, optional
            Whether to ignore existing cached entries.
        use_api_key : bool, optional
            Whether to include the API key.
        md5_checksum : str or None, optional
            Expected MD5 checksum for response validation.
        **request_kwargs : Any
            Additional request arguments.

        Returns
        -------
        requests.Response
            HTTP response.
        """
        return self._request(
            method="GET",
            path=path,
            enable_cache=enable_cache,
            refresh_cache=refresh_cache,
            use_api_key=use_api_key,
            md5_checksum=md5_checksum,
            **request_kwargs,
        )

    def post(
        self,
        path: str,
        *,
        use_api_key: bool = True,
        **request_kwargs: Any,
    ) -> Response:
        """
        Send a POST request.

        Parameters
        ----------
        path : str
            API path relative to the base URL.
        use_api_key : bool, optional
            Whether to include the API key.
        **request_kwargs : Any
            Additional request arguments.

        Returns
        -------
        requests.Response
            HTTP response.
        """
        return self._request(
            method="POST",
            path=path,
            enable_cache=False,
            use_api_key=use_api_key,
            **request_kwargs,
        )

    def delete(
        self,
        path: str,
        **request_kwargs: Any,
    ) -> Response:
        """
        Send a DELETE request.

        Parameters
        ----------
        path : str
            API path relative to the base URL.
        **request_kwargs : Any
            Additional request arguments.

        Returns
        -------
        requests.Response
            HTTP response.
        """
        return self._request(
            method="DELETE",
            path=path,
            enable_cache=False,
            use_api_key=True,
            **request_kwargs,
        )

    def download(
        self,
        url: str,
        handler: Callable[[Response, Path, str], None] | None = None,
        encoding: str = "utf-8",
        file_name: str = "response.txt",
        md5_checksum: str | None = None,
    ) -> Path:
        """
        Download a resource and store it in the cache directory.

        Parameters
        ----------
        url : str
            Absolute URL of the resource to download.
        handler : callable or None, optional
            Custom handler function accepting ``(response, path, encoding)``
            and returning a ``pathlib.Path``.
        encoding : str, optional
            Text encoding used when writing the response body.
        file_name : str, optional
            Name of the saved file.
        md5_checksum : str or None, optional
            Expected MD5 checksum for integrity verification.

        Returns
        -------
        pathlib.Path
            Path to the downloaded file.

        Raises
        ------
        OpenMLHashException
            If checksum verification fails.
        """
        base = self.cache.path
        file_path = base / "downloads" / urlparse(url).path.lstrip("/") / file_name
        file_path = file_path.expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            return file_path

        response = self.get(url, md5_checksum=md5_checksum)

        def write_to_file(response: Response, path: Path, encoding: str) -> None:
            path.write_text(response.text, encoding)

        handler = handler or write_to_file
        handler(response, file_path, encoding)
        return file_path
