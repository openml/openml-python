# License: BSD 3-Clause

import time
import hashlib
import logging
import math
import pathlib
import random
import requests
import urllib.parse
import xml
import xmltodict
from typing import Dict, Optional, Union

import minio

from . import config
from .exceptions import (
    OpenMLServerError,
    OpenMLServerException,
    OpenMLServerNoResult,
    OpenMLHashException,
)


def _perform_api_call(call, request_method, data=None, file_elements=None):
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
    return_code : int
        HTTP return code
    return_value : str
        Return value of the OpenML server
    """
    url = config.server
    if not url.endswith("/"):
        url += "/"
    url += call

    url = url.replace("=", "%3d")
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
        "%.7fs taken for [%s] request for the URL %s", time.time() - start, request_method, url,
    )
    return response.text


def _download_minio_file(
    source: str, destination: Union[str, pathlib.Path], exists_ok: bool = True,
) -> None:
    """ Download file ``source`` from a MinIO Bucket and store it at ``destination``.

    Parameters
    ----------
    source : Union[str, pathlib.Path]
        URL to a file in a MinIO bucket.
    destination : str
        Path to store the file to, if a directory is provided the original filename is used.
    exists_ok : bool, optional (default=True)
        If False, raise FileExists if a file already exists in ``destination``.

    """
    destination = pathlib.Path(destination)
    parsed_url = urllib.parse.urlparse(source)

    # expect path format: /BUCKET/path/to/file.ext
    bucket, object_name = parsed_url.path[1:].split("/", maxsplit=1)
    if destination.is_dir():
        destination = pathlib.Path(destination, object_name)
    if destination.is_file() and not exists_ok:
        raise FileExistsError(f"File already exists in {destination}.")

    client = minio.Minio(endpoint=parsed_url.netloc, secure=False)

    try:
        client.fget_object(
            bucket_name=bucket, object_name=object_name, file_path=str(destination),
        )
    except minio.error.S3Error as e:
        if e.message.startswith("Object does not exist"):
            raise FileNotFoundError(f"Object at '{source}' does not exist.") from e
        # e.g. permission error, or a bucket does not exist (which is also interpreted as a
        # permission error on minio level).
        raise FileNotFoundError("Bucket does not exist or is private.") from e


def _download_text_file(
    source: str,
    output_path: Optional[str] = None,
    md5_checksum: str = None,
    exists_ok: bool = True,
    encoding: str = "utf8",
) -> Optional[str]:
    """ Download the text file at `source` and store it in `output_path`.

    By default, do nothing if a file already exists in `output_path`.
    The downloaded file can be checked against an expected md5 checksum.

    Parameters
    ----------
    source : str
        url of the file to be downloaded
    output_path : str, (optional)
        full path, including filename, of where the file should be stored. If ``None``,
        this function returns the downloaded file as string.
    md5_checksum : str, optional (default=None)
        If not None, should be a string of hexidecimal digits of the expected digest value.
    exists_ok : bool, optional (default=True)
        If False, raise an FileExistsError if there already exists a file at `output_path`.
    encoding : str, optional (default='utf8')
        The encoding with which the file should be stored.
    """
    if output_path is not None:
        try:
            with open(output_path, encoding=encoding):
                if exists_ok:
                    return None
                else:
                    raise FileExistsError
        except FileNotFoundError:
            pass

    logging.info("Starting [%s] request for the URL %s", "get", source)
    start = time.time()
    response = __read_url(source, request_method="get", md5_checksum=md5_checksum)
    downloaded_file = response.text

    if output_path is None:
        logging.info(
            "%.7fs taken for [%s] request for the URL %s", time.time() - start, "get", source,
        )
        return downloaded_file

    else:
        with open(output_path, "w", encoding=encoding) as fh:
            fh.write(downloaded_file)

        logging.info(
            "%.7fs taken for [%s] request for the URL %s", time.time() - start, "get", source,
        )

        del downloaded_file
        return None


def _file_id_to_url(file_id, filename=None):
    """
     Presents the URL how to download a given file id
     filename is optional
    """
    openml_url = config.server.split("/api/")
    url = openml_url[0] + "/data/download/%s" % file_id
    if filename is not None:
        url += "/" + filename
    return url


def _read_url_files(url, data=None, file_elements=None):
    """do a post request to url with data
    and sending file_elements as files"""

    data = {} if data is None else data
    data["api_key"] = config.apikey
    if file_elements is None:
        file_elements = {}
    # Using requests.post sets header 'Accept-encoding' automatically to
    # 'gzip,deflate'
    response = _send_request(request_method="post", url=url, data=data, files=file_elements,)
    return response


def __read_url(url, request_method, data=None, md5_checksum=None):
    data = {} if data is None else data
    if config.apikey:
        data["api_key"] = config.apikey
    return _send_request(
        request_method=request_method, url=url, data=data, md5_checksum=md5_checksum
    )


def __is_checksum_equal(downloaded_file, md5_checksum=None):
    if md5_checksum is None:
        return True
    md5 = hashlib.md5()
    md5.update(downloaded_file.encode("utf-8"))
    md5_checksum_download = md5.hexdigest()
    if md5_checksum == md5_checksum_download:
        return True
    return False


def _send_request(request_method, url, data, files=None, md5_checksum=None):
    n_retries = max(1, config.connection_n_retries)

    response = None
    with requests.Session() as session:
        # Start at one to have a non-zero multiplier for the sleep
        for retry_counter in range(1, n_retries + 1):
            try:
                if request_method == "get":
                    response = session.get(url, params=data)
                elif request_method == "delete":
                    response = session.delete(url, params=data)
                elif request_method == "post":
                    response = session.post(url, data=data, files=files)
                else:
                    raise NotImplementedError()
                __check_response(response=response, url=url, file_elements=files)
                if request_method == "get" and not __is_checksum_equal(response.text, md5_checksum):
                    raise OpenMLHashException(
                        "Checksum of downloaded file is unequal to the expected checksum {} "
                        "when downloading {}.".format(md5_checksum, url)
                    )
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
                OpenMLServerException,
                xml.parsers.expat.ExpatError,
                OpenMLHashException,
            ) as e:
                if isinstance(e, OpenMLServerException):
                    if e.code not in [107]:
                        # 107: database connection error
                        raise
                elif isinstance(e, xml.parsers.expat.ExpatError):
                    if request_method != "get" or retry_counter >= n_retries:
                        raise OpenMLServerError(
                            "Unexpected server error when calling {}. Please contact the "
                            "developers!\nStatus code: {}\n{}".format(
                                url, response.status_code, response.text,
                            )
                        )
                if retry_counter >= n_retries:
                    raise
                else:

                    def robot(n: int) -> float:
                        wait = (1 / (1 + math.exp(-(n * 0.5 - 4)))) * 60
                        variation = random.gauss(0, wait / 10)
                        return max(1.0, wait + variation)

                    def human(n: int) -> float:
                        return max(1.0, n)

                    delay = {"human": human, "robot": robot}[config.retry_policy](retry_counter)
                    time.sleep(delay)
    if response is None:
        raise ValueError("This should never happen!")
    return response


def __check_response(response, url, file_elements):
    if response.status_code != 200:
        raise __parse_server_exception(response, url, file_elements=file_elements)
    elif (
        "Content-Encoding" not in response.headers or response.headers["Content-Encoding"] != "gzip"
    ):
        logging.warning("Received uncompressed content from OpenML for {}.".format(url))


def __parse_server_exception(
    response: requests.Response, url: str, file_elements: Dict,
) -> OpenMLServerError:

    if response.status_code == 414:
        raise OpenMLServerError("URI too long! ({})".format(url))
    try:
        server_exception = xmltodict.parse(response.text)
    except xml.parsers.expat.ExpatError:
        raise
    except Exception:
        # OpenML has a sophisticated error system
        # where information about failures is provided. try to parse this
        raise OpenMLServerError(
            "Unexpected server error when calling {}. Please contact the developers!\n"
            "Status code: {}\n{}".format(url, response.status_code, response.text)
        )

    server_error = server_exception["oml:error"]
    code = int(server_error["oml:code"])
    message = server_error["oml:message"]
    additional_information = server_error.get("oml:additional_information")
    if code in [372, 512, 500, 482, 542, 674]:
        if additional_information:
            full_message = "{} - {}".format(message, additional_information)
        else:
            full_message = message

        # 512 for runs, 372 for datasets, 500 for flows
        # 482 for tasks, 542 for evaluations, 674 for setups
        return OpenMLServerNoResult(code=code, message=full_message,)
    # 163: failure to validate flow XML (https://www.openml.org/api_docs#!/flow/post_flow)
    if code in [163] and file_elements is not None and "description" in file_elements:
        # file_elements['description'] is the XML file description of the flow
        full_message = "\n{}\n{} - {}".format(
            file_elements["description"], message, additional_information,
        )
    else:
        full_message = "{} - {}".format(message, additional_information)
    return OpenMLServerException(code=code, message=full_message, url=url)
