import time
import requests
import warnings

import xmltodict

from . import config
from .exceptions import (OpenMLServerError, OpenMLServerException,
                         OpenMLServerNoResult)


def _perform_api_call(call, data=None, file_elements=None):
    """
    Perform an API call at the OpenML server.

    Parameters
    ----------
    call : str
        The API call. For example data/list
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
        return _read_url_files(url, data=data, file_elements=file_elements)
    return _read_url(url, data)


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
        raise _parse_server_exception(response, url=url)
    if 'Content-Encoding' not in response.headers or \
            response.headers['Content-Encoding'] != 'gzip':
        warnings.warn('Received uncompressed content from OpenML for %s.' % url)
    return response.text


def _read_url(url, data=None):

    data = {} if data is None else data
    if config.apikey is not None:
        data['api_key'] = config.apikey

    if len(data) == 0 or (len(data) == 1 and 'api_key' in data):
        response = send_request(
            request_method='get', url=url, data=data,
        )

    else:
        # Using requests.post sets header 'Accept-encoding' automatically to
        #  'gzip,deflate'
        response = send_request(
            request_method='post', url=url, data=data,
        )

    if response.status_code != 200:
        raise _parse_server_exception(response, url=url)
    if 'Content-Encoding' not in response.headers or \
            response.headers['Content-Encoding'] != 'gzip':
        warnings.warn('Received uncompressed content from OpenML for %s.' % url)
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


def _parse_server_exception(response, url=None):
    # OpenML has a sopisticated error system
    # where information about failures is provided. try to parse this
    try:
        server_exception = xmltodict.parse(response.text)
    except Exception:
        raise OpenMLServerError(('Unexpected server error. Please '
                                 'contact the developers!\nStatus code: '
                                 '%d\n' % response.status_code) + response.text)

    code = int(server_exception['oml:error']['oml:code'])
    message = server_exception['oml:error']['oml:message']
    additional = None
    if 'oml:additional_information' in server_exception['oml:error']:
        additional = server_exception['oml:error']['oml:additional_information']
    if code in [372, 512, 500, 482, 542, 674]:
        # 512 for runs, 372 for datasets, 500 for flows
        # 482 for tasks, 542 for evaluations, 674 for setups
        return OpenMLServerNoResult(code, message, additional)
    return OpenMLServerException(
        code=code,
        message=message,
        additional=additional,
        url=url
    )
