import os
import requests
import arff

from . import config
from .exceptions import OpenMLServerError


def _perform_api_call(call, data=None, file_dictionary=None,
                      file_elements=None, add_authentication=True):
    """
    Perform an API call at the OpenML server.
    return self._read_url(url, data=data, filePath=filePath,
    def _read_url(self, url, add_authentication=False, data=None, filePath=None):

    Parameters
    ----------
    call : str
        The API call. For example data/list
    data : dict (default=None)
        Dictionary containing data which will be sent to the OpenML
        server via a POST request.
    **kwargs
        Further arguments which are appended as GET arguments.

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
    if file_dictionary is not None or file_elements is not None:
        return _read_url_files(url, data=data, file_dictionary=file_dictionary,
                               file_elements=file_elements)
    return _read_url(url, data=data)


def _read_url_files(url, data=None, file_dictionary=None, file_elements=None):
    """do a post request to url with data None, file content of
    file_dictionary and sending file_elements as files"""
    if data is None:
        data = {}
    data['api_key'] = config.apikey
    if file_elements is None:
        file_elements = {}
    if file_dictionary is not None:
        for key, path in file_dictionary.items():
            path = os.path.abspath(path)
            if os.path.exists(path):
                try:
                    if key is 'dataset':
                        # check if arff is valid?
                        decoder = arff.ArffDecoder()
                        with open(path) as fh:
                            decoder.decode(fh, encode_nominal=True)
                except:
                    raise ValueError("The file you have provided is not a valid arff file")

                file_elements[key] = open(path, 'rb')

            else:
                raise ValueError("File doesn't exist")
    response = requests.post(url, data=data, files=file_elements)
    if response.status_code != 200:
        raise OpenMLServerError(response.text)
    return response.status_code, response.text


def _read_url(url, data=None):
    if data is None:
        data = {}
    data['api_key'] = config.apikey

    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise OpenMLServerError(response.text)
    return response.status_code, response.text
