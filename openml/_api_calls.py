# License: BSD 3-Clause

import time
from typing import Dict
import requests
import warnings

import xmltodict

from . import config
from .exceptions import (OpenMLServerError, OpenMLServerException,
                         OpenMLServerNoResult)


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

    url = url.replace('=', '%3d')

    if file_elements is not None:
        if request_method != 'post':
            raise ValueError('request method must be post when file elements '
                             'are present')
        return _read_url_files(url, data=data, file_elements=file_elements)
    return _read_url(url, request_method, data)


def _file_id_to_url(file_id, filename=None):
    """
     Presents the URL how to download a given file id
     filename is optional
    """
    openml_url = config.server.split('/api/')
    url = openml_url[0] + '/data/download/%s' % file_id
    if filename is not None:
        url += '/' + filename
    return url


def _read_url_files(url, data=None, file_elements=None):
    """do a post request to url with data
    and sending file_elements as files"""

    data = {} if data is None else data
    data['api_key'] = config.apikey
    if file_elements is None:
        file_elements = {}
    # Using requests.post sets header 'Accept-encoding' automatically to
    # 'gzip,deflate'
    response = send_request(
        request_method='post',
        url=url,
        data=data,
        files=file_elements,
    )
    if response.status_code != 200:
        raise _parse_server_exception(response, url, file_elements=file_elements)
    if 'Content-Encoding' not in response.headers or \
            response.headers['Content-Encoding'] != 'gzip':
        warnings.warn('Received uncompressed content from OpenML for {}.'
                      .format(url))
    return response.text


def _read_url(url, request_method, data=None):
    data = {} if data is None else data
    if config.apikey is not None:
        data['api_key'] = config.apikey

    response = send_request(request_method=request_method, url=url, data=data)
    if response.status_code != 200:
        raise _parse_server_exception(response, url, file_elements=None)
    if 'Content-Encoding' not in response.headers or \
            response.headers['Content-Encoding'] != 'gzip':
        warnings.warn('Received uncompressed content from OpenML for {}.'
                      .format(url))
    return response.text


def send_request(
    request_method,
    url,
    data,
    files=None,
):
    n_retries = config.connection_n_retries
    response = None
    with requests.Session() as session:
        # Start at one to have a non-zero multiplier for the sleep
        for i in range(1, n_retries + 1):
            try:
                if request_method == 'get':
                    response = session.get(url, params=data)
                elif request_method == 'delete':
                    response = session.delete(url, params=data)
                elif request_method == 'post':
                    response = session.post(url, data=data, files=files)
                else:
                    raise NotImplementedError()
                break
            except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.SSLError,
            ) as e:
                if i == n_retries:
                    raise e
                else:
                    time.sleep(0.1 * i)
    if response is None:
        raise ValueError('This should never happen!')
    return response


def _parse_server_exception(
    response: requests.Response,
    url: str,
    file_elements: Dict,
) -> OpenMLServerError:
    # OpenML has a sophisticated error system
    # where information about failures is provided. try to parse this
    try:
        server_exception = xmltodict.parse(response.text)
    except Exception:
        raise OpenMLServerError(
            'Unexpected server error when calling {}. Please contact the developers!\n'
            'Status code: {}\n{}'.format(url, response.status_code, response.text))

    server_error = server_exception['oml:error']
    code = int(server_error['oml:code'])
    message = server_error['oml:message']
    additional_information = server_error.get('oml:additional_information')
    if code in [372, 512, 500, 482, 542, 674]:
        if additional_information:
            full_message = '{} - {}'.format(message, additional_information)
        else:
            full_message = message

        # 512 for runs, 372 for datasets, 500 for flows
        # 482 for tasks, 542 for evaluations, 674 for setups
        return OpenMLServerNoResult(
            code=code,
            message=full_message,
        )
    # 163: failure to validate flow XML (https://www.openml.org/api_docs#!/flow/post_flow)
    if code in [163] and file_elements is not None and 'description' in file_elements:
        # file_elements['description'] is the XML file description of the flow
        full_message = '\n{}\n{} - {}'.format(
            file_elements['description'],
            message,
            additional_information,
        )
    else:
        full_message = '{} - {}'.format(message, additional_information)
    return OpenMLServerException(
        code=code,
        message=full_message,
        url=url
    )
