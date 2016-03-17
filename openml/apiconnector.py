from collections import OrderedDict
import logging
import os
import re
import sys
#import tempfile
import requests
import arff
import xmltodict

if sys.version_info[0] < 3:
    import ConfigParser as configparser
    from StringIO import StringIO
else:
    import configparser
    from io import StringIO

from .exceptions import OpenMLCacheException
from .entities.split import OpenMLSplit

logger = logging.getLogger(__name__)

OPENML_URL = "http://api_new.openml.org/v1/"


class APIConnector(object):
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
    def __init__(self, cache_directory=None, apikey=None,
                 server=None, verbosity=None, configure_logger=True,
                 private_directory=None):
        # The .openml directory is necessary, just try to create it (EAFP)
        try:
            os.mkdir(os.path.expanduser('~/.openml'))
        except (IOError, OSError):
            # TODO add debug information
            pass

        # Set all variables in the configuration object
        self.config = self._parse_config()
        if cache_directory is not None:
            self.config.set('FAKE_SECTION', 'cachedir', cache_directory)
        if apikey is not None:
            self.config.set('FAKE_SECTION', 'apikey', apikey)
        if server is not None:
            self.config.set('FAKE_SECTION', 'server', server)
        if verbosity is not None:
            self.config.set('FAKE_SECTION', 'verbosity', verbosity)
        if private_directory is not None:
            self.config.set('FAKE_SECTION', 'private_directory', private_directory)

        if configure_logger:
            verbosity = self.config.getint('FAKE_SECTION', 'verbosity')
            if verbosity <= 0:
                level = logging.ERROR
            elif verbosity == 1:
                level = logging.INFO
            elif verbosity >= 2:
                level = logging.DEBUG
            logging.basicConfig(
                format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                       'message)s', datefmt='%H:%M:%S', level=level)

        # Set up the cache directories
        self.cache_dir = self.config.get('FAKE_SECTION', 'cachedir')
        self.dataset_cache_dir = os.path.join(self.cache_dir, "datasets")
        self.task_cache_dir = os.path.join(self.cache_dir, "tasks")
        self.run_cache_dir = os.path.join(self.cache_dir, 'runs')

        # Set up the private directory
        self.private_directory = self.config.get('FAKE_SECTION',
                                                 'private_directory')
        self._private_directory_datasets = os.path.join(
            self.private_directory, "datasets")
        self._private_directory_tasks = os.path.join(
            self.private_directory, "tasks")
        self._private_directory_runs = os.path.join(
            self.private_directory, "runs")

        for dir_ in [self.cache_dir, self.dataset_cache_dir,
                     self.task_cache_dir, self.run_cache_dir,
                     self.private_directory,
                     self._private_directory_datasets,
                     self._private_directory_tasks,
                     self._private_directory_runs]:
            if not os.path.exists(dir_) and not os.path.isdir(dir_):
                os.mkdir(dir_)

    def _parse_config(self):
        defaults = {'apikey': '',
                    'server': OPENML_URL,
                    'verbosity': 0,
                    'cachedir': os.path.expanduser('~/.openml/cache'),
                    'private_directory': os.path.expanduser('~/.openml/private')}

        config_file = os.path.expanduser('~/.openml/config')
        config = configparser.RawConfigParser(defaults=defaults)

        if not os.path.exists(config_file):
            # Create an empty config file if there was none so far
            fh = open(config_file, "w")
            fh.close()
            logger.info("Could not find a configuration file at %s. Going to "
                        "create an empty file there." % config_file)

        try:
            # Cheat the ConfigParser module by adding a fake section header
            config_file_ = StringIO()
            config_file_.write("[FAKE_SECTION]\n")
            with open(config_file) as fh:
                for line in fh:
                    config_file_.write(line)
            config_file_.seek(0)
            config.readfp(config_file_)
        except OSError as e:
            logging.info("Error opening file %s: %s" %
                         config_file, e.message)
        return config

    # -> OpenMLSplit
    def get_cached_splits(self):
        splits = OrderedDict()
        for task_cache_dir in [self.task_cache_dir,
                               self._private_directory_tasks]:
            directory_content = os.listdir(task_cache_dir)
            directory_content.sort()

            for filename in directory_content:
                match = re.match(r"(tid)_([0-9]*)\.arff", filename)
                if match:
                    tid = match.group(2)
                    tid = int(tid)

                    splits[tid] = self.get_cached_task(tid)

        return splits

    # -> OpenMLSplit
    def get_cached_split(self, tid):
        for task_cache_dir in [self.task_cache_dir,
                               self._private_directory_tasks]:
            try:
                split_file = os.path.join(task_cache_dir,
                                          "tid_%d.arff" % int(tid))
                split = OpenMLSplit.from_arff_file(split_file)
                return split

            except (OSError, IOError):
                continue

        raise OpenMLCacheException("Split file for tid %d not "
                                   "cached" % tid)

    def _perform_api_call(self, call, data=None, file_dictionary=None,
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
        url = self.config.get("FAKE_SECTION", "server")
        if not url.endswith("/"):
            url += "/"
        url += call
        if file_dictionary is not None or file_elements is not None:
            return self._read_url_files(url, data=data,
                                        file_dictionary=file_dictionary,
                                        file_elements=file_elements)
        return self._read_url(url, data=data)

    def _read_url_files(self, url, data=None, file_dictionary=None, file_elements=None):
        """do a post request to url with data None, file content of
        file_dictionary and sending file_elements as files"""
        if data is None:
            data = {}
        data['api_key'] = self.config.get('FAKE_SECTION', 'apikey')
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

    def _read_url(self, url, data=None):
        if data is None:
            data = {}
        data['api_key'] = self.config.get('FAKE_SECTION', 'apikey')

        response = requests.post(url, data=data)
        return response.status_code, response.text

    # -> OpenMLFlow
    def upload_flow(self, description, flow):
        """
        The 'description' is binary data of an XML file according to the XSD Schema (OUTDATED!):
        https://github.com/openml/website/blob/master/openml_OS/views/pages/rest_api/xsd/openml.implementation.upload.xsd

        (optional) file_path is the absolute path to the file that is the flow (eg. a script)
        """
        data = {'description': description, 'source': flow}
        return_code, dataset_xml = self._perform_api_call(
            "/flow/", data=data)
        return return_code, dataset_xml

    # -> OpenMLFlow
    def check_flow_exists(self, name, version):
        """Retrieves the flow id of the flow uniquely identified by name+version.

        Returns flow id if such a flow exists,
        returns -1 if flow does not exists,
        returns -2 if there was not a well-formed response from the server
        http://www.openml.org/api_docs/#!/flow/get_flow_exists_name_version
        """
        # Perhaps returns the -1/-2 business with proper raising of exceptions?

        if not (type(name) is str and len(name) > 0):
            raise ValueError('Parameter \'name\' should be a non-empty string')
        if not (type(version) is str and len(version) > 0):
            raise ValueError('Parameter \'version\' should be a non-empty string')

        return_code, xml_response = self._perform_api_call(
            "/flow/exists/%s/%s" % (name, version))
        if return_code != 200:
            # fixme raise appropriate error
            raise ValueError("api call failed: %s" % xml_response)
        xml_dict = xmltodict.parse(xml_response)
        flow_id = xml_dict['oml:flow_exists']['oml:id']
        return return_code, xml_response, flow_id
