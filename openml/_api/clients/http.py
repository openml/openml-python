from __future__ import annotations

import json
import logging
import math
import random
import time
import xml
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import requests
import xmltodict
from requests import Response

from openml.__version__ import __version__
from openml._api.config import RetryPolicy
from openml.exceptions import (
    OpenMLNotAuthorizedError,
    OpenMLServerError,
    OpenMLServerException,
    OpenMLServerNoResult,
)


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
        retry_policy: RetryPolicy,
        cache: HTTPCache | None = None,
    ) -> None:
        self.server = server
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_policy = retry_policy
        self.cache = cache

        self.retry_func = (
            self._human_delay if retry_policy == RetryPolicy.HUMAN else self._robot_delay
        )
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}

    def _robot_delay(self, n: int) -> float:
        wait = (1 / (1 + math.exp(-(n * 0.5 - 4)))) * 60
        variation = random.gauss(0, wait / 10)
        return max(1.0, wait + variation)

    def _human_delay(self, n: int) -> float:
        return max(1.0, n)

    def _parse_exception_response(
        self,
        response: Response,
    ) -> tuple[int | None, str]:
        content_type = response.headers.get("Content-Type", "").lower()

        if "json" in content_type:
            server_exception = response.json()
            server_error = server_exception["detail"]
            if isinstance(server_error, dict):
                code = server_error.get("code")
                message = server_error.get("message")
                additional_information = server_error.get("additional_information")
            else:
                code = None
                message = str(server_error)
                additional_information = None
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
        elif message:
            full_message = message
        elif additional_information:
            full_message = additional_information
        else:
            full_message = ""

        return code, full_message

    def _raise_code_specific_error(
        self,
        code: int,
        message: str,
        url: str,
        files: Mapping[str, Any] | None,
    ) -> None:
        if code in [111, 372, 512, 500, 482, 542, 674]:
            # 512 for runs, 372 for datasets, 500 for flows
            # 482 for tasks, 542 for evaluations, 674 for setups
            # 111 for dataset descriptions
            raise OpenMLServerNoResult(code=code, message=message, url=url)

        # 163: failure to validate flow XML (https://www.openml.org/api_docs#!/flow/post_flow)
        if code in [163] and files is not None and "description" in files:
            # file_elements['description'] is the XML file description of the flow
            message = f"\n{files['description']}\n{message}"

        if code in [
            102,  # flow/exists post
            137,  # dataset post
            350,  # dataset/42 delete
            310,  # flow/<something> post
            320,  # flow/42 delete
            400,  # run/42 delete
            460,  # task/42 delete
        ]:
            raise OpenMLNotAuthorizedError(
                message=(
                    f"The API call {url} requires authentication via an API key.\nPlease configure "
                    "OpenML-Python to use your API as described in this example:"
                    "\nhttps://openml.github.io/openml-python/latest/examples/Basics/introduction_tutorial/#authentication"
                )
            )

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
        if (
            "Content-Encoding" not in response.headers
            or response.headers["Content-Encoding"] != "gzip"
        ):
            logging.warning(f"Received uncompressed content from OpenML for {url}.")

        if response.status_code == 200:
            return None

        if response.status_code == requests.codes.URI_TOO_LONG:
            raise OpenMLServerError(f"URI too long! ({url})")

        retry_raise_e: Exception | None = None

        try:
            code, message = self._parse_exception_response(response)

        except (requests.exceptions.JSONDecodeError, xml.parsers.expat.ExpatError) as e:
            if method != "GET":
                extra = f"Status code: {response.status_code}\n{response.text}"
                raise OpenMLServerError(
                    f"Unexpected server error when calling {url}. Please contact the "
                    f"developers!\n{extra}"
                ) from e

            retry_raise_e = e

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

        if retry_raise_e is None:
            retry_raise_e = OpenMLServerException(code=code, message=message, url=url)

        return retry_raise_e

    def _request(  # noqa: PLR0913
        self,
        method: str,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float | int,
        files: Mapping[str, Any] | None,
        **request_kwargs: Any,
    ) -> tuple[Response | None, Exception | None]:
        retry_raise_e: Exception | None = None
        response: Response | None = None

        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
                files=files,
                **request_kwargs,
            )
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
        ) as e:
            retry_raise_e = e

        if response is not None:
            retry_raise_e = self._validate_response(
                method=method,
                url=url,
                files=files,
                response=response,
            )

        return response, retry_raise_e

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
        retries = max(1, self.retries)

        # prepare params
        params = request_kwargs.pop("params", {}).copy()
        if use_api_key:
            params["api_key"] = self.api_key

        # prepare headers
        headers = request_kwargs.pop("headers", {}).copy()
        headers.update(self.headers)

        timeout = request_kwargs.pop("timeout", self.timeout)
        files = request_kwargs.pop("files", None)

        use_cache = False

        if use_cache and self.cache is not None:
            cache_key = self.cache.get_key(url, params)
            try:
                return self.cache.load(cache_key)
            except (FileNotFoundError, TimeoutError):
                pass  # cache miss or expired, continue
            except Exception:
                raise  # propagate unexpected cache errors

        for retry_counter in range(1, retries + 1):
            response, retry_raise_e = self._request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
                files=files,
                **request_kwargs,
            )

            # executed successfully
            if retry_raise_e is None:
                break
            # tries completed
            if retry_counter >= retries:
                raise retry_raise_e

            delay = self.retry_func(retry_counter)
            time.sleep(delay)

        assert response is not None

        if use_cache and self.cache is not None:
            self.cache.save(cache_key, response)

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
