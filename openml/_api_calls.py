# License: BSD 3-Clause

import time
import hashlib
import logging
import requests
import xmltodict
from typing import Dict, Optional

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
        response = __read_url_files(url, data=data, file_elements=file_elements)
    else:
        response = __read_url(url, request_method, data)

    __check_response(response, url, file_elements)

    logging.info(
        "%.7fs taken for [%s] request for the URL %s", time.time() - start, request_method, url,
    )
    return response.text


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
    response = __read_url(source, request_method="get")
    __check_response(response, source, None)
    downloaded_file = response.text

    if md5_checksum is not None:
        md5 = hashlib.md5()
        md5.update(downloaded_file.encode("utf-8"))
        md5_checksum_download = md5.hexdigest()
        if md5_checksum != md5_checksum_download:
            raise OpenMLHashException(
                "Checksum {} of downloaded file is unequal to the expected checksum {}.".format(
                    md5_checksum_download, md5_checksum
                )
            )

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


def __check_response(response, url, file_elements):
    if response.status_code != 200:
        raise __parse_server_exception(response, url, file_elements=file_elements)
    elif (
        "Content-Encoding" not in response.headers or response.headers["Content-Encoding"] != "gzip"
    ):
        logging.warning("Received uncompressed content from OpenML for {}.".format(url))


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


def __read_url_files(url, data=None, file_elements=None):
    """do a post request to url with data
    and sending file_elements as files"""

    data = {} if data is None else data
    data["api_key"] = config.apikey
    if file_elements is None:
        file_elements = {}
    # Using requests.post sets header 'Accept-encoding' automatically to
    # 'gzip,deflate'
    response = __send_request(request_method="post", url=url, data=data, files=file_elements,)
    return response


def __read_url(url, request_method, data=None):
    data = {} if data is None else data
    if config.apikey is not None:
        data["api_key"] = config.apikey

    return __send_request(request_method=request_method, url=url, data=data)


def __send_request(
    request_method, url, data, files=None,
):
    n_retries = config.connection_n_retries
    response = None
    with requests.Session() as session:
        # Start at one to have a non-zero multiplier for the sleep
        for i in range(1, n_retries + 1):
            try:
                if request_method == "get":
                    response = session.get(url, params=data)
                elif request_method == "delete":
                    response = session.delete(url, params=data)
                elif request_method == "post":
                    response = session.post(url, data=data, files=files)
                else:
                    raise NotImplementedError()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.SSLError,) as e:
                if i == n_retries:
                    raise e
                else:
                    time.sleep(0.1 * i)
    if response is None:
        raise ValueError("This should never happen!")
    return response


def __parse_server_exception(
    response: requests.Response, url: str, file_elements: Dict,
) -> OpenMLServerError:

    if response.status_code == 414:
        raise OpenMLServerError("URI too long! ({})".format(url))
    try:
        server_exception = xmltodict.parse(response.text)
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
