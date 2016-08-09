"""
Stores module level information like the API key, cache director, private
directory and the server.
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s] [%(asctime)s:%(name)s] %('
           'message)s', datefmt='%H:%M:%S')

server = "http://www.openml.org/api/v1/xml"
apikey = ""
cachedir = ""
privatedir = ""

if sys.version_info[0] < 3:
    import ConfigParser as configparser
    from StringIO import StringIO
else:
    import configparser
    from io import StringIO


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
    # read config file, create cache directory
    try:
        os.mkdir(os.path.expanduser('~/.openml'))
    except (IOError, OSError):
        # TODO add debug information
        pass
    config = _parse_config()
    apikey = config.get('FAKE_SECTION', 'apikey')
    server = config.get('FAKE_SECTION', 'server')
    private_dir = config.get('FAKE_SECTION', 'private_directory')
    cache_dir = config.get('FAKE_SECTION', 'cachedir')
    set_cache_directory(cache_dir, private_dir)


def set_cache_directory(cachedir, privatedir=None):
    """Set module-wide cache directory.

    Sets the cache directory into which to download datasets, tasks etc.
    Also sets the private directory for storing local datasets.

    Parameters
    ----------
    cachedir : string
        Path to use as cache directory.

    privatedir : string
        Path containing private datasets, tasks, etc.

    See also
    --------
    get_cache_directory
    get_private_directory
    """
    if privatedir is None:
        privatedir = cachedir

    global _cachedir
    global _privatedir
    _cachedir = cachedir
    _privatedir = privatedir

    # Set up the cache directories
    dataset_cache_dir = os.path.join(cachedir, "datasets")
    task_cache_dir = os.path.join(cachedir, "tasks")
    run_cache_dir = os.path.join(cachedir, 'runs')

    # Set up the private directory
    _private_directory_datasets = os.path.join(
        privatedir, "datasets")
    _private_directory_tasks = os.path.join(
        privatedir, "tasks")
    _private_directory_runs = os.path.join(
        privatedir, "runs")

    for dir_ in [cachedir, dataset_cache_dir,
                 task_cache_dir, run_cache_dir,
                 privatedir,
                 _private_directory_datasets,
                 _private_directory_tasks,
                 _private_directory_runs]:
        if not os.path.exists(dir_) and not os.path.isdir(dir_):
            os.mkdir(dir_)


def _parse_config():
    """Parse the config file, set up defaults.
    """
    defaults = {'apikey': apikey,
                'server': server,
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


def get_cache_directory():
    """Get the current cache directory.

    Returns
    -------
    cachedir : string
        The current cache directory.

    See also
    --------
    set_cache_directory
    get_private_directory
    """
    return _cachedir


def get_private_directory():
    """Get the current private directory.

    Returns
    -------
    privatecir : string
        The current private directory.

    See also
    --------
    set_cache_directory
    get_cache_directory
    """
    return _privatedir

__all__ = ["set_cache_directory", 'get_cache_directory', 'get_private_directory']

_setup()
