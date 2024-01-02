"""
Store module level information like the API key, cache directory and the server
"""

# License: BSD 3-Clause

import logging
import logging.handlers
import os
from pathlib import Path
import platform
from typing import Dict, Optional, Tuple, Union, cast
import warnings

from io import StringIO
import configparser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
openml_logger = logging.getLogger("openml")
console_handler = None
file_handler = None  # type: Optional[logging.Handler]


def _create_log_handlers(create_file_handler: bool = True) -> None:
    """Creates but does not attach the log handlers."""
    global console_handler, file_handler
    if console_handler is not None or file_handler is not None:
        logger.debug("Requested to create log handlers, but they are already created.")
        return

    message_format = "[%(levelname)s] [%(asctime)s:%(name)s] %(message)s"
    output_formatter = logging.Formatter(message_format, datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(output_formatter)

    if create_file_handler:
        one_mb = 2**20
        log_path = os.path.join(_root_cache_directory, "openml_python.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=one_mb, backupCount=1, delay=True
        )
        file_handler.setFormatter(output_formatter)


def _convert_log_levels(log_level: int) -> Tuple[int, int]:
    """Converts a log level that's either defined by OpenML/Python to both specifications."""
    # OpenML verbosity level don't match Python values directly:
    openml_to_python = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    python_to_openml = {
        logging.DEBUG: 2,
        logging.INFO: 1,
        logging.WARNING: 0,
        logging.CRITICAL: 0,
        logging.ERROR: 0,
    }
    # Because the dictionaries share no keys, we use `get` to convert as necessary:
    openml_level = python_to_openml.get(log_level, log_level)
    python_level = openml_to_python.get(log_level, log_level)
    return openml_level, python_level


def _set_level_register_and_store(handler: logging.Handler, log_level: int) -> None:
    """Set handler log level, register it if needed, save setting to config file if specified."""
    oml_level, py_level = _convert_log_levels(log_level)
    handler.setLevel(py_level)

    if openml_logger.level > py_level or openml_logger.level == logging.NOTSET:
        openml_logger.setLevel(py_level)

    if handler not in openml_logger.handlers:
        openml_logger.addHandler(handler)


def set_console_log_level(console_output_level: int) -> None:
    """Set console output to the desired level and register it with openml logger if needed."""
    global console_handler
    _set_level_register_and_store(cast(logging.Handler, console_handler), console_output_level)


def set_file_log_level(file_output_level: int) -> None:
    """Set file output to the desired level and register it with openml logger if needed."""
    global file_handler
    _set_level_register_and_store(cast(logging.Handler, file_handler), file_output_level)


# Default values (see also https://github.com/openml/OpenML/wiki/Client-API-Standards)
_defaults = {
    "apikey": "",
    "server": "https://www.openml.org/api/v1/xml",
    "cachedir": (
        os.environ.get(
            "XDG_CACHE_HOME",
            os.path.join(
                "~",
                ".cache",
                "openml",
            ),
        )
        if platform.system() == "Linux"
        else os.path.join("~", ".openml")
    ),
    "avoid_duplicate_runs": "True",
    "retry_policy": "human",
    "connection_n_retries": "5",
}

# Default values are actually added here in the _setup() function which is
# called at the end of this module
server = str(_defaults["server"])  # so mypy knows it is a string


def get_server_base_url() -> str:
    """Return the base URL of the currently configured server.

    Turns ``"https://www.openml.org/api/v1/xml"`` in ``"https://www.openml.org/"``

    Returns
    =======
    str
    """
    return server.split("/api")[0]


apikey = _defaults["apikey"]
# The current cache directory (without the server name)
_root_cache_directory = str(_defaults["cachedir"])  # so mypy knows it is a string
avoid_duplicate_runs = True if _defaults["avoid_duplicate_runs"] == "True" else False

retry_policy = _defaults["retry_policy"]
connection_n_retries = int(_defaults["connection_n_retries"])


def set_retry_policy(value: str, n_retries: Optional[int] = None) -> None:
    global retry_policy
    global connection_n_retries
    default_retries_by_policy = dict(human=5, robot=50)

    if value not in default_retries_by_policy:
        raise ValueError(
            f"Detected retry_policy '{value}' but must be one of "
            f"{list(default_retries_by_policy.keys())}"
        )
    if n_retries is not None and not isinstance(n_retries, int):
        raise TypeError(f"`n_retries` must be of type `int` or `None` but is `{type(n_retries)}`.")
    if isinstance(n_retries, int) and n_retries < 1:
        raise ValueError(f"`n_retries` is '{n_retries}' but must be positive.")

    retry_policy = value
    connection_n_retries = default_retries_by_policy[value] if n_retries is None else n_retries


class ConfigurationForExamples:
    """Allows easy switching to and from a test configuration, used for examples."""

    _last_used_server = None
    _last_used_key = None
    _start_last_called = False
    _test_server = "https://test.openml.org/api/v1/xml"
    _test_apikey = "c0c42819af31e706efe1f4b88c23c6c1"

    @classmethod
    def start_using_configuration_for_example(cls) -> None:
        """Sets the configuration to connect to the test server with valid apikey.

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
        warnings.warn(
            "Switching to the test server {} to not upload results to the live server. "
            "Using the test server may result in reduced performance of the API!".format(server)
        )

    @classmethod
    def stop_using_configuration_for_example(cls) -> None:
        """Return to configuration as it was before `start_use_example_configuration`."""
        if not cls._start_last_called:
            # We don't want to allow this because it will (likely) result in the `server` and
            # `apikey` variables being set to None.
            raise RuntimeError(
                "`stop_use_example_configuration` called without a saved config."
                "`start_use_example_configuration` must be called first."
            )

        global server
        global apikey

        server = cast(str, cls._last_used_server)
        apikey = cast(str, cls._last_used_key)
        cls._start_last_called = False


def determine_config_file_path() -> Path:
    if platform.system() == "Linux":
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path("~") / ".config" / "openml"))
    else:
        config_dir = Path("~") / ".openml"
    # Still use os.path.expanduser to trigger the mock in the unit test
    config_dir = Path(os.path.expanduser(config_dir))
    return config_dir / "config"


def _setup(config: Optional[Dict[str, Union[str, int, bool]]] = None) -> None:
    """Setup openml package. Called on first import.

    Reads the config file and sets up apikey, server, cache appropriately.
    key and server can be set by the user simply using
    openml.config.apikey = THEIRKEY
    openml.config.server = SOMESERVER
    We could also make it a property but that's less clear.
    """
    global apikey
    global server
    global _root_cache_directory
    global avoid_duplicate_runs

    config_file = determine_config_file_path()
    config_dir = config_file.parent

    # read config file, create directory for config file
    if not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir, exist_ok=True)
            cache_exists = True
        except PermissionError:
            cache_exists = False
    else:
        cache_exists = True

    if config is None:
        config = cast(Dict[str, Union[str, int, bool]], _parse_config(config_file))
    config = cast(Dict[str, Union[str, int, bool]], config)

    avoid_duplicate_runs = bool(config.get("avoid_duplicate_runs"))

    apikey = cast(str, config["apikey"])
    server = cast(str, config["server"])
    short_cache_dir = cast(str, config["cachedir"])

    tmp_n_retries = config["connection_n_retries"]
    if tmp_n_retries is not None:
        n_retries = int(tmp_n_retries)
    else:
        n_retries = None

    set_retry_policy(cast(str, config["retry_policy"]), n_retries)

    _root_cache_directory = os.path.expanduser(short_cache_dir)
    # create the cache subdirectory
    if not os.path.exists(_root_cache_directory):
        try:
            os.makedirs(_root_cache_directory, exist_ok=True)
        except PermissionError:
            openml_logger.warning(
                "No permission to create openml cache directory at %s! This can result in "
                "OpenML-Python not working properly." % _root_cache_directory
            )

    if cache_exists:
        _create_log_handlers()
    else:
        _create_log_handlers(create_file_handler=False)
        openml_logger.warning(
            "No permission to create OpenML directory at %s! This can result in OpenML-Python "
            "not working properly." % config_dir
        )


def set_field_in_config_file(field: str, value: str) -> None:
    """Overwrites the `field` in the configuration file with the new `value`."""
    if field not in _defaults:
        raise ValueError(f"Field '{field}' is not valid and must be one of '{_defaults.keys()}'.")

    globals()[field] = value
    config_file = determine_config_file_path()
    config = _parse_config(str(config_file))
    with open(config_file, "w") as fh:
        for f in _defaults.keys():
            # We can't blindly set all values based on globals() because when the user
            # sets it through config.FIELD it should not be stored to file.
            # There doesn't seem to be a way to avoid writing defaults to file with configparser,
            # because it is impossible to distinguish from an explicitly set value that matches
            # the default value, to one that was set to its default because it was omitted.
            value = config.get("FAKE_SECTION", f)
            if f == field:
                value = globals()[f]
            fh.write(f"{f} = {value}\n")


def _parse_config(config_file: Union[str, Path]) -> Dict[str, str]:
    """Parse the config file, set up defaults."""
    config = configparser.RawConfigParser(defaults=_defaults)

    # The ConfigParser requires a [SECTION_HEADER], which we do not expect in our config file.
    # Cheat the ConfigParser module by adding a fake section header
    config_file_ = StringIO()
    config_file_.write("[FAKE_SECTION]\n")
    try:
        with open(config_file) as fh:
            for line in fh:
                config_file_.write(line)
    except FileNotFoundError:
        logger.info("No config file found at %s, using default configuration.", config_file)
    except OSError as e:
        logger.info("Error opening file %s: %s", config_file, e.args[0])
    config_file_.seek(0)
    config.read_file(config_file_)
    config_as_dict = {key: value for key, value in config.items("FAKE_SECTION")}
    return config_as_dict


def get_config_as_dict() -> Dict[str, Union[str, int, bool]]:
    config = dict()  # type: Dict[str, Union[str, int, bool]]
    config["apikey"] = apikey
    config["server"] = server
    config["cachedir"] = _root_cache_directory
    config["avoid_duplicate_runs"] = avoid_duplicate_runs
    config["connection_n_retries"] = connection_n_retries
    config["retry_policy"] = retry_policy
    return config


def get_cache_directory() -> str:
    """Get the current cache directory.

    This gets the cache directory for the current server relative
    to the root cache directory that can be set via
    ``set_root_cache_directory()``. The cache directory is the
    ``root_cache_directory`` with additional information on which
    subdirectory to use based on the server name. By default it is
    ``root_cache_directory / org / openml / www`` for the standard
    OpenML.org server and is defined as
    ``root_cache_directory / top-level domain / second-level domain /
    hostname``
    ```

    Returns
    -------
    cachedir : string
        The current cache directory.

    """
    url_suffix = urlparse(server).netloc
    reversed_url_suffix = os.sep.join(url_suffix.split(".")[::-1])
    _cachedir = os.path.join(_root_cache_directory, reversed_url_suffix)
    return _cachedir


def set_root_cache_directory(root_cache_directory: str) -> None:
    """Set module-wide base cache directory.

    Sets the root cache directory, wherin the cache directories are
    created to store content from different OpenML servers. For example,
    by default, cached data for the standard OpenML.org server is stored
    at ``root_cache_directory / org / openml / www``, and the general
    pattern is ``root_cache_directory / top-level domain / second-level
    domain / hostname``.

    Parameters
    ----------
    root_cache_directory : string
         Path to use as cache directory.

    See also
    --------
    get_cache_directory
    """

    global _root_cache_directory
    _root_cache_directory = root_cache_directory


start_using_configuration_for_example = (
    ConfigurationForExamples.start_using_configuration_for_example
)
stop_using_configuration_for_example = ConfigurationForExamples.stop_using_configuration_for_example


__all__ = [
    "get_cache_directory",
    "set_root_cache_directory",
    "start_using_configuration_for_example",
    "stop_using_configuration_for_example",
    "get_config_as_dict",
]

_setup()
