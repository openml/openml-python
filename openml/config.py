"""
Stores module level information like the API key, cache directory and the server.
"""
import logging
import os

from six import StringIO
from six.moves import configparser


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s] [%(asctime)s:%(name)s] %('
           'message)s', datefmt='%H:%M:%S')

config_file = os.path.expanduser('~/.openml/config')
server = "https://www.openml.org/api/v1/xml"
apikey = ""
cachedir = ""


def _setup():
    """Setup openml package. Called on first import.

    Reads the config file and sets up apikey, server, cache appropriately.
    key and server can be set by the user simply using
    openml.config.apikey = THEIRKEY
    openml.config.server = SOMESERVER
    The cache dir needs to be set up calling set_cache_directory
    because it needs some setup.
    We could also make it a property but that's less clear.
    """
    global apikey
    global server
    global avoid_duplicate_runs
    # read config file, create cache directory
    try:
        os.mkdir(os.path.expanduser('~/.openml'))
    except (IOError, OSError):
        # TODO add debug information
        pass
    config = _parse_config()
    apikey = config.get('FAKE_SECTION', 'apikey')
    server = config.get('FAKE_SECTION', 'server')
    cache_dir = config.get('FAKE_SECTION', 'cachedir')
    avoid_duplicate_runs = config.getboolean('FAKE_SECTION', 'avoid_duplicate_runs')
    set_cache_directory(cache_dir)


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

    global _cachedir
    _cachedir = cachedir

    # Set up the cache directories
    dataset_cache_dir = os.path.join(cachedir, "datasets")
    task_cache_dir = os.path.join(cachedir, "tasks")
    run_cache_dir = os.path.join(cachedir, 'runs')
    lock_dir = os.path.join(cachedir, 'locks')

    for dir_ in [
        cachedir, dataset_cache_dir, task_cache_dir, run_cache_dir, lock_dir,
    ]:
        if not os.path.exists(dir_) and not os.path.isdir(dir_):
            os.mkdir(dir_)


def _parse_config():
    """Parse the config file, set up defaults.
    """
    defaults = {'apikey': apikey,
                'server': server,
                'verbosity': 0,
                'cachedir': os.path.expanduser('~/.openml/cache'),
                'avoid_duplicate_runs': 'True'}

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


def get_cache_directory():
    """Get the current cache directory.

    Returns
    -------
    cachedir : string
        The current cache directory.

    See also
    --------
    set_cache_directory
    """
    return _cachedir


__all__ = ["set_cache_directory", 'get_cache_directory']

_setup()
