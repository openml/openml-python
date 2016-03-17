import os
import requests
import arff

from . import config


"""
Provides an interface to the OpenML server.

All parameters of the APIConnector can be either specified in a config
file or when creating this object. The config file must be placed in a
directory ``.openml`` inside the users home directory and have the name
``config``. If one of the parameters is specified by passing it to the
constructor of this class, it will override the value specified in the
configuration file.

Parameters
----------
cache_directory : string, optional (default=None)
    A local directory which will be used for caching. If this is not set, a
    directory '.openml/cache' in the users home directory will be used.
    If either directory does not exist, it will be created.

apikey : string, optional (default=None)
    Your OpenML API key which will be used to authenticate you at the OpenML
    server.

server : string, optional (default=None)
    The OpenML server to connect to.

verbosity : int, optional (default=None)

configure_logger : bool (default=True)
    Whether the python logging module should be configured by the openml
    package. If set to true, this is a very basic configuration,
    which only prints to the standard output. This is only recommended
    for testing or small problems. It is set to True to adhere to the
    `specifications of the OpenML client API
    <https://github.com/openml/OpenML/wiki/Client-API>`_.
    When the openml module is used as a library, it is recommended that
    the main application controls the logging level, e.g. see
    `here <http://pieces.openpolitics.com
    /2012/04/python-logging-best-practices/>`_.

private_directory : str, optional (default=None)
    A local directory which can be accessed through the OpenML package.
    Useful to access private datasets through the same interface.

Raises
------
ValueError
    If apikey is neither specified in the config nor given as an argument.
OpenMLServerError
    If the OpenML server returns an unexptected response.

Notes
-----
Testing the API calls in Firefox is possible with the Firefox AddOn
HTTPRequestor.

"""


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
    return response.status_code, response.text


def _read_url(url, data=None):
    if data is None:
        data = {}
    data['api_key'] = config.apikey

    response = requests.post(url, data=data)
    return response.status_code, response.text
