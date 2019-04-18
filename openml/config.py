"""
Store module level information like the API key, cache directory and the server
"""
import logging
import os

from io import StringIO
import configparser
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s] [%(asctime)s:%(name)s] %('
           'message)s', datefmt='%H:%M:%S')

# Default values!
_defaults = {
    'apikey': None,
    'server': "https://www.openml.org/api/v1/xml",
    'verbosity': 0,
    'cachedir': os.path.expanduser(os.path.join('~', '.openml', 'cache')),
    'avoid_duplicate_runs': 'True',
    'connection_n_retries': 2,
}

config_file = os.path.expanduser(os.path.join('~', '.openml', 'config'))

# Default values are actually added here in the _setup() function which is
# called at the end of this module
server = _defaults['server']
apikey = _defaults['apikey']
# The current cache directory (without the server name)
cache_directory = _defaults['cachedir']
avoid_duplicate_runs = True if _defaults['avoid_duplicate_runs'] == 'True' else False

# Number of retries if the connection breaks
connection_n_retries = _defaults['connection_n_retries']

# variables to keep track of last used settings for when `use_example_configuration` is activated
_last_used_key = None
_last_used_server = None


def start_use_example_configuration():
    global server
    global apikey
    global _last_used_server
    global _last_used_key
    _last_used_server = server
    _last_used_key = apikey

    # Test server key for examples
    server = "https://test.openml.org/api/v1/xml"
    apikey = "c0c42819af31e706efe1f4b88c23c6c1"


def stop_use_example_configuration():
    global server
    global apikey
    global _last_used_server
    global _last_used_key

    server = _last_used_server
    apikey = _last_used_key


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
    except (IOError, OSError):
        # TODO add debug information
        pass
    config = _parse_config()
    apikey = config.get('FAKE_SECTION', 'apikey')
    server = config.get('FAKE_SECTION', 'server')

    short_cache_dir = config.get('FAKE_SECTION', 'cachedir')
    cache_directory = os.path.expanduser(short_cache_dir)

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
        logging.info("Error opening file %s: %s", config_file, e.message)
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


__all__ = [
    'get_cache_directory', 'set_cache_directory'
]

_setup()
