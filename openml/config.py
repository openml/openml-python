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
    # read config file, create cache directory
    try:
        os.mkdir(os.path.expanduser('~/.openml'))
    except (IOError, OSError):
        # TODO add debug information
        pass
    config = _parse_config()
    set_api_key(config.get('FAKE_SECTION', 'apikey'))
    set_server(config.get('FAKE_SECTION', 'server'))
    private_dir = config.get('FAKE_SECTION', 'private_directory')
    cache_dir = config.get('FAKE_SECTION', 'cachedir')
    set_cache_directory(cache_dir, private_dir)
    print(config)


def _setup_cache_directories(cache_dir, private_dir):
    # Set up the cache directories
    dataset_cache_dir = os.path.join(cache_dir, "datasets")
    task_cache_dir = os.path.join(cache_dir, "tasks")
    run_cache_dir = os.path.join(cache_dir, 'runs')

    # Set up the private directory
    _private_directory_datasets = os.path.join(
        private_dir, "datasets")
    _private_directory_tasks = os.path.join(
        private_dir, "tasks")
    _private_directory_runs = os.path.join(
        private_dir, "runs")

    for dir_ in [cache_dir, dataset_cache_dir,
                 task_cache_dir, run_cache_dir,
                 private_dir,
                 _private_directory_datasets,
                 _private_directory_tasks,
                 _private_directory_runs]:
        if not os.path.exists(dir_) and not os.path.isdir(dir_):
            os.mkdir(dir_)


def set_cache_directory(cache_dir, private_dir):
    global CACHEDIR
    global PRIVATEDIR
    CACHEDIR = cache_dir
    PRIVATEDIR = private_dir
    _setup_cache_directories(cache_dir, private_dir)


def set_api_key(apikey):
    global APIKEY
    APIKEY = apikey


def set_server(url):
    global OPENML_URL
    OPENML_URL = url


def _parse_config():
    defaults = {'apikey': APIKEY,
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

__all__ = ["set_cache_directory", "set_api_key"]

_setup()
