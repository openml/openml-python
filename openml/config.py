"""
Store module level information like the API key, cache directory and the server
"""

# License: BSD 3-Clause

import logging
import logging.handlers
import os
from typing import cast

from io import StringIO
import configparser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def configure_logging(console_output_level: int, file_output_level: int):
    """ Sets the OpenML logger to DEBUG, with attached Stream- and FileHandler. """
    # Verbosity levels as defined (https://github.com/openml/OpenML/wiki/Client-API-Standards)
    # don't match Python values directly:
    verbosity_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    openml_logger = logging.getLogger('openml')
    openml_logger.setLevel(logging.DEBUG)
    message_format = '[%(levelname)s] [%(asctime)s:%(name)s] %(message)s'
    output_formatter = logging.Formatter(message_format, datefmt='%H:%M:%S')

    console_stream = logging.StreamHandler()
    console_stream.setFormatter(output_formatter)
    console_stream.setLevel(verbosity_map[console_output_level])

    one_mb = 2**20
    log_path = os.path.join(cache_directory, 'openml_python.log')
    file_stream = logging.handlers.RotatingFileHandler(log_path, maxBytes=one_mb, backupCount=1)
    file_stream.setLevel(verbosity_map[file_output_level])
    file_stream.setFormatter(output_formatter)

    openml_logger.addHandler(console_stream)
    openml_logger.addHandler(file_stream)
    return console_stream, file_stream


# Default values (see also https://github.com/openml/OpenML/wiki/Client-API-Standards)
_defaults = {
    'apikey': None,
    'server': "https://www.openml.org/api/v1/xml",
    'verbosity': 0,  # WARNING
    'file_verbosity': 2,  # DEBUG
    'cachedir': os.path.expanduser(os.path.join('~', '.openml', 'cache')),
    'avoid_duplicate_runs': 'True',
    'connection_n_retries': 2,
}

config_file = os.path.expanduser(os.path.join('~', '.openml', 'config'))

# Default values are actually added here in the _setup() function which is
# called at the end of this module
server = str(_defaults['server'])  # so mypy knows it is a string
server_base_url = server[:-len('/api/v1/xml')]
apikey = _defaults['apikey']
# The current cache directory (without the server name)
cache_directory = str(_defaults['cachedir'])  # so mypy knows it is a string
avoid_duplicate_runs = True if _defaults['avoid_duplicate_runs'] == 'True' else False

# Number of retries if the connection breaks
connection_n_retries = _defaults['connection_n_retries']


class ConfigurationForExamples:
    """ Allows easy switching to and from a test configuration, used for examples. """
    _last_used_server = None
    _last_used_key = None
    _start_last_called = False
    _test_server = "https://test.openml.org/api/v1/xml"
    _test_apikey = "c0c42819af31e706efe1f4b88c23c6c1"

    @classmethod
    def start_using_configuration_for_example(cls):
        """ Sets the configuration to connect to the test server with valid apikey.

        To configuration as was before this call is stored, and can be recovered
        by using the `stop_use_example_configuration` method.
        """
        global server
        global apikey

        if cls._start_last_called and server == cls._test_server and apikey == cls._test_apikey:
            # Method is called more than once in a row without modifying the server or apikey.
            # We don't want to save the current test configuration as a last used configuration.
            return

        cls._last_used_server = server
        cls._last_used_key = apikey
        cls._start_last_called = True

        # Test server key for examples
        server = cls._test_server
        apikey = cls._test_apikey

    @classmethod
    def stop_using_configuration_for_example(cls):
        """ Return to configuration as it was before `start_use_example_configuration`. """
        if not cls._start_last_called:
            # We don't want to allow this because it will (likely) result in the `server` and
            # `apikey` variables being set to None.
            raise RuntimeError("`stop_use_example_configuration` called without a saved config."
                               "`start_use_example_configuration` must be called first.")

        global server
        global apikey

        server = cls._last_used_server
        apikey = cls._last_used_key
        cls._start_last_called = False


def _setup():
    """Setup openml package. Called on first import.

    Reads the config file and sets up apikey, server, cache appropriately.
    key and server can be set by the user simply using
    openml.config.apikey = THEIRKEY
    openml.config.server = SOMESERVER
    We could also make it a property but that's less clear.
    """
    global apikey
    global server
    global cache_directory
    global avoid_duplicate_runs
    global connection_n_retries

    # read config file, create cache directory
    try:
        os.mkdir(os.path.expanduser(os.path.join('~', '.openml')))
    except FileExistsError:
        # For other errors, we want to propagate the error as openml does not work without cache
        pass

    config = _parse_config()
    apikey = config.get('FAKE_SECTION', 'apikey')
    server = config.get('FAKE_SECTION', 'server')

    short_cache_dir = config.get('FAKE_SECTION', 'cachedir')
    cache_directory = os.path.expanduser(short_cache_dir)

    # create the cache subdirectory
    try:
        os.mkdir(cache_directory)
    except FileExistsError:
        # For other errors, we want to propagate the error as openml does not work without cache
        pass

    avoid_duplicate_runs = config.getboolean('FAKE_SECTION',
                                             'avoid_duplicate_runs')
    connection_n_retries = config.get('FAKE_SECTION', 'connection_n_retries')
    if connection_n_retries > 20:
        raise ValueError(
            'A higher number of retries than 20 is not allowed to keep the '
            'server load reasonable'
        )


def _parse_config():
    """Parse the config file, set up defaults.
    """

    config = configparser.RawConfigParser(defaults=_defaults)

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
        config.read_file(config_file_)
    except OSError as e:
        logger.info("Error opening file %s: %s", config_file, e.message)
    return config


def get_cache_directory():
    """Get the current cache directory.

    Returns
    -------
    cachedir : string
        The current cache directory.

    """
    url_suffix = urlparse(server).netloc
    reversed_url_suffix = os.sep.join(url_suffix.split('.')[::-1])
    if not cache_directory:
        _cachedir = _defaults(cache_directory)
    else:
        _cachedir = cache_directory
    _cachedir = os.path.join(_cachedir, reversed_url_suffix)
    return _cachedir


def set_cache_directory(cachedir):
    """Set module-wide cache directory.

    Sets the cache directory into which to download datasets, tasks etc.

    Parameters
    ----------
    cachedir : string
         Path to use as cache directory.

    See also
    --------
    get_cache_directory
    """

    global cache_directory
    cache_directory = cachedir


start_using_configuration_for_example = (
    ConfigurationForExamples.start_using_configuration_for_example
)
stop_using_configuration_for_example = (
    ConfigurationForExamples.stop_using_configuration_for_example
)

__all__ = [
    'get_cache_directory',
    'set_cache_directory',
    'start_using_configuration_for_example',
    'stop_using_configuration_for_example',
]

_setup()

_console_log_level = cast(int, _defaults['verbosity'])
_file_log_level = cast(int, _defaults['file_verbosity'])
console_log, file_log = configure_logging(_console_log_level, _file_log_level)
