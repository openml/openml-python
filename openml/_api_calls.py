# License: BSD 3-Clause
from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import random
import shutil
import time
import urllib.parse
import xml
import zipfile
from pathlib import Path

import minio
import requests
import requests.utils
import xmltodict
from urllib3 import ProxyManager

from . import config
from .__version__ import __version__
from .exceptions import (
    OpenMLAuthenticationError,
    OpenMLHashException,
    OpenMLServerError,
    OpenMLServerException,
    OpenMLServerNoResult,
)
from .utils import ProgressBar

_HEADERS = {"user-agent": f"openml-python/{__version__}"}

DATA_TYPE = dict[str, str | int]
FILE_ELEMENTS_TYPE = dict[str, str | tuple[str, str]]
DATABASE_CONNECTION_ERRCODE = 107

API_TOKEN_HELP_LINK = "https://openml.github.io/openml-python/latest/examples/Basics/introduction_tutorial/#authentication"  # noqa: S105


def _robot_delay(n: int) -> float:
    wait = (1 / (1 + math.exp(-(n * 0.5 - 4)))) * 60
    variation = random.gauss(0, wait / 10)
    return max(1.0, wait + variation)


def _human_delay(n: int) -> float:
    return max(1.0, n)


def resolve_env_proxies(url: str) -> str | None:
    """Attempt to find a suitable proxy for this url.

    Relies on ``requests`` internals to remain consistent. To disable this from the
    environment, please set the enviornment varialbe ``no_proxy="*"``.

    Parameters
    ----------
    url : str
        The url endpoint

    Returns
    -------
    Optional[str]
        The proxy url if found, else None
    """
    resolved_proxies = requests.utils.get_environ_proxies(url)
    return requests.utils.select_proxy(url, resolved_proxies)  # type: ignore


def _create_url_from_endpoint(endpoint: str) -> str:
    url = config.server
    if not url.endswith("/"):
        url += "/"
    url += endpoint
    return url.replace("=", "%3d")


def _perform_api_call(
    call: str,
    request_method: str,
    data: DATA_TYPE | None = None,
    file_elements: FILE_ELEMENTS_TYPE | None = None,
) -> str:
    """
    Perform an API call at the OpenML server.

    Parameters
    ----------
    call : str
        The API call. For example data/list
    request_method : str
        The HTTP request method to perform the API call with. Legal values:
            - get (reading functions, api key optional)
            - post (writing functions, generaly require api key)
            - delete (deleting functions, require api key)
        See REST api documentation which request method is applicable.
    data : dict
        Dictionary with post-request payload.
    file_elements : dict
        Mapping of {filename: str} of strings which should be uploaded as
        files to the server.

    Returns
    -------
    return_value : str
        Return value of the OpenML server
    """
    url = _create_url_from_endpoint(call)
    logging.info("Starting [%s] request for the URL %s", request_method, url)
    start = time.time()

    if file_elements is not None:
        if request_method != "post":
            raise ValueError("request method must be post when file elements are present")
        response = _read_url_files(url, data=data, file_elements=file_elements)
    else:
        response = __read_url(url, request_method, data)

    __check_response(response, url, file_elements)

    logging.info(
        "%.7fs taken for [%s] request for the URL %s",
        time.time() - start,
        request_method,
        url,
    )
    return response.text


def _download_minio_file(
    source: str,
    destination: str | Path,
    exists_ok: bool = True,  # noqa: FBT002
    proxy: str | None = "auto",
) -> None:
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
    destination = Path(destination)
    parsed_url = urllib.parse.urlparse(source)

    # expect path format: /BUCKET/path/to/file.ext
    bucket, object_name = parsed_url.path[1:].split("/", maxsplit=1)
    if destination.is_dir():
        destination = Path(destination, object_name)
    if destination.is_file() and not exists_ok:
        raise FileExistsError(f"File already exists in {destination}.")

    if proxy == "auto":
        proxy = resolve_env_proxies(parsed_url.geturl())

    proxy_client = ProxyManager(proxy) if proxy else None

    client = minio.Minio(endpoint=parsed_url.netloc, secure=False, http_client=proxy_client)
    try:
        client.fget_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=str(destination),
            progress=ProgressBar() if config.show_progress else None,
            request_headers=_HEADERS,
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


def _download_minio_bucket(source: str, destination: str | Path) -> None:
    """Download file ``source`` from a MinIO Bucket and store it at ``destination``.

    Does not redownload files which already exist.

    Parameters
    ----------
    source : str
        URL to a MinIO bucket.
    destination : str | Path
        Path to a directory to store the bucket content in.
    """
    destination = Path(destination)
    parsed_url = urllib.parse.urlparse(source)

    # expect path format: /BUCKET/path/to/file.ext
    _, bucket, *prefixes, _file = parsed_url.path.split("/")
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
            # Marker is missing but archive exists means the server archive changed, force a refresh
            shutil.rmtree(file_destination.parent / file_destination.stem)

        with contextlib.suppress(FileExistsError):
            _download_minio_file(
                source=source.rsplit("/", 1)[0] + "/" + file_object.object_name.rsplit("/", 1)[1],
                destination=file_destination,
                exists_ok=False,
            )

        if file_destination.is_file() and file_destination.suffix == ".zip":
            file_destination.unlink()
            marker.touch()


def _download_text_file(
    source: str,
    output_path: str | Path | None = None,
    md5_checksum: str | None = None,
    exists_ok: bool = True,  # noqa: FBT002
    encoding: str = "utf8",
) -> str | None:
    """Download the text file at `source` and store it in `output_path`.

    By default, do nothing if a file already exists in `output_path`.
    The downloaded file can be checked against an expected md5 checksum.

    Parameters
    ----------
    source : str
        url of the file to be downloaded
    output_path : str | Path | None (default=None)
        full path, including filename, of where the file should be stored. If ``None``,
        this function returns the downloaded file as string.
    md5_checksum : str, optional (default=None)
        If not None, should be a string of hexidecimal digits of the expected digest value.
    exists_ok : bool, optional (default=True)
        If False, raise an FileExistsError if there already exists a file at `output_path`.
    encoding : str, optional (default='utf8')
        The encoding with which the file should be stored.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    if output_path is not None and output_path.exists():
        if not exists_ok:
            raise FileExistsError

        return None

    logging.info("Starting [%s] request for the URL %s", "get", source)
    start = time.time()
    response = __read_url(source, request_method="get", md5_checksum=md5_checksum)
    downloaded_file = response.text

    if output_path is None:
        logging.info(
            "%.7fs taken for [%s] request for the URL %s",
            time.time() - start,
            "get",
            source,
        )
        return downloaded_file

    with output_path.open("w", encoding=encoding) as fh:
        fh.write(downloaded_file)

    logging.info(
        "%.7fs taken for [%s] request for the URL %s",
        time.time() - start,
        "get",
        source,
    )
    return None


def _file_id_to_url(file_id: int, filename: str | None = None) -> str:
    """
    Presents the URL how to download a given file id
    filename is optional
    """
    openml_url = config.server.split("/api/")
    url = openml_url[0] + f"/data/download/{file_id!s}"
    if filename is not None:
        url += "/" + filename
    return url


def _read_url_files(
    url: str,
    data: DATA_TYPE | None = None,
    file_elements: FILE_ELEMENTS_TYPE | None = None,
) -> requests.Response:
    """Do a post request to url with data
    and sending file_elements as files
    """
    data = {} if data is None else data
    data["api_key"] = config.apikey
    if file_elements is None:
        file_elements = {}
    # Using requests.post sets header 'Accept-encoding' automatically to
    # 'gzip,deflate'
    return _send_request(
        request_method="post",
        url=url,
        data=data,
        files=file_elements,
    )


def __read_url(
    url: str,
    request_method: str,
    data: DATA_TYPE | None = None,
    md5_checksum: str | None = None,
) -> requests.Response:
    data = {} if data is None else data
    if config.apikey:
        data["api_key"] = config.apikey
    return _send_request(
        request_method=request_method,
        url=url,
        data=data,
        md5_checksum=md5_checksum,
    )


def __is_checksum_equal(downloaded_file_binary: bytes, md5_checksum: str | None = None) -> bool:
    if md5_checksum is None:
        return True
    md5 = hashlib.md5()  # noqa: S324
    md5.update(downloaded_file_binary)
    md5_checksum_download = md5.hexdigest()
    return md5_checksum == md5_checksum_download


def _send_request(  # noqa: C901, PLR0912
    request_method: str,
    url: str,
    data: DATA_TYPE,
    files: FILE_ELEMENTS_TYPE | None = None,
    md5_checksum: str | None = None,
) -> requests.Response:
    n_retries = max(1, config.connection_n_retries)

    response: requests.Response | None = None
    delay_method = _human_delay if config.retry_policy == "human" else _robot_delay

    # Error to raise in case of retrying too often. Will be set to the last observed exception.
    retry_raise_e: Exception | None = None

    with requests.Session() as session:
        # Start at one to have a non-zero multiplier for the sleep
        for retry_counter in range(1, n_retries + 1):
            try:
                if request_method == "get":
                    response = session.get(url, params=data, headers=_HEADERS)
                elif request_method == "delete":
                    response = session.delete(url, params=data, headers=_HEADERS)
                elif request_method == "post":
                    response = session.post(url, data=data, files=files, headers=_HEADERS)
                else:
                    raise NotImplementedError()

                __check_response(response=response, url=url, file_elements=files)

                if request_method == "get" and not __is_checksum_equal(
                    response.text.encode("utf-8"), md5_checksum
                ):
                    # -- Check if encoding is not UTF-8 perhaps
                    if __is_checksum_equal(response.content, md5_checksum):
                        raise OpenMLHashException(
                            f"Checksum of downloaded file is unequal to the expected checksum"
                            f"{md5_checksum} because the text encoding is not UTF-8 when "
                            f"downloading {url}. There might be a sever-sided issue with the file, "
                            "see: https://github.com/openml/openml-python/issues/1180.",
                        )

                    raise OpenMLHashException(
                        f"Checksum of downloaded file is unequal to the expected checksum "
                        f"{md5_checksum} when downloading {url}.",
                    )

                return response
            except OpenMLServerException as e:
                # Propagate all server errors to the calling functions, except
                # for 107 which represents a database connection error.
                # These are typically caused by high server load,
                # which means trying again might resolve the issue.
                if e.code != DATABASE_CONNECTION_ERRCODE:
                    raise e
                retry_raise_e = e
            except xml.parsers.expat.ExpatError as e:
                if request_method != "get" or retry_counter >= n_retries:
                    if response is not None:
                        extra = f"Status code: {response.status_code}\n{response.text}"
                    else:
                        extra = "No response retrieved."

                    raise OpenMLServerError(
                        f"Unexpected server error when calling {url}. Please contact the "
                        f"developers!\n{extra}"
                    ) from e
                retry_raise_e = e
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
                OpenMLHashException,
            ) as e:
                retry_raise_e = e

            # We can only be here if there was an exception
            assert retry_raise_e is not None
            if retry_counter >= n_retries:
                raise retry_raise_e
            delay = delay_method(retry_counter)
            time.sleep(delay)

    assert response is not None
    return response


def __check_response(
    response: requests.Response,
    url: str,
    file_elements: FILE_ELEMENTS_TYPE | None,
) -> None:
    if response.status_code != 200:
        raise __parse_server_exception(response, url, file_elements=file_elements)
    if "Content-Encoding" not in response.headers or response.headers["Content-Encoding"] != "gzip":
        logging.warning(f"Received uncompressed content from OpenML for {url}.")


def __parse_server_exception(
    response: requests.Response,
    url: str,
    file_elements: FILE_ELEMENTS_TYPE | None,
) -> OpenMLServerError:
    if response.status_code == requests.codes.URI_TOO_LONG:
        raise OpenMLServerError(f"URI too long! ({url})")

    # OpenML has a sophisticated error system where information about failures is provided,
    # in the response body itself.
    # First, we need to parse it out.
    try:
        server_exception = xmltodict.parse(response.text)
    except xml.parsers.expat.ExpatError as e:
        raise e
    except Exception as e:
        # If we failed to parse it out, then something has gone wrong in the body we have sent back
        # from the server and there is little extra information we can capture.
        raise OpenMLServerError(
            f"Unexpected server error when calling {url}. Please contact the developers!\n"
            f"Status code: {response.status_code}\n{response.text}",
        ) from e

    # Now we can parse out the specific error codes that we return. These
    # are in addition to the typical HTTP error codes, but encode more
    # specific informtion. You can find these codes here:
    # https://github.com/openml/OpenML/blob/develop/openml_OS/views/pages/api_new/v1/xml/pre.php
    server_error = server_exception["oml:error"]
    code = int(server_error["oml:code"])
    message = server_error["oml:message"]
    additional_information = server_error.get("oml:additional_information")
    if code in [111, 372, 512, 500, 482, 542, 674]:
        if additional_information:
            full_message = f"{message} - {additional_information}"
        else:
            full_message = message

        # 512 for runs, 372 for datasets, 500 for flows
        # 482 for tasks, 542 for evaluations, 674 for setups
        # 111 for dataset descriptions
        return OpenMLServerNoResult(code=code, message=full_message, url=url)

    # 163: failure to validate flow XML (https://www.openml.org/api_docs#!/flow/post_flow)
    if code in [163] and file_elements is not None and "description" in file_elements:
        # file_elements['description'] is the XML file description of the flow
        full_message = "\n{}\n{} - {}".format(
            file_elements["description"],
            message,
            additional_information,
        )
    else:
        full_message = f"{message} - {additional_information}"

    if code in [
        102,  # flow/exists post
        137,  # dataset post
        350,  # dataset/42 delete
        310,  # flow/<something> post
        320,  # flow/42 delete
        400,  # run/42 delete
        460,  # task/42 delete
    ]:
        msg = f"The API call {url} requires authentication via an API key."
        return OpenMLAuthenticationError(message=msg)

    return OpenMLServerException(code=code, message=full_message, url=url)
