"""Store module level information like the API key, cache directory and the server"""

# License: BSD 3-Clause
from __future__ import annotations

import configparser
import logging
import logging.handlers
import os
import platform
import shutil
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
openml_logger = logging.getLogger("openml")


def _resolve_default_cache_dir() -> Path:
    user_defined_cache_dir = os.environ.get("OPENML_CACHE_DIR")
    if user_defined_cache_dir is not None:
        return Path(user_defined_cache_dir)

    if platform.system().lower() != "linux":
        return Path("~", ".openml")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home is None:
        return Path("~", ".cache", "openml")

    cache_dir = Path(xdg_cache_home) / "openml"
    if cache_dir.exists():
        return cache_dir

    heuristic_dir_for_backwards_compat = Path(xdg_cache_home) / "org" / "openml"
    if not heuristic_dir_for_backwards_compat.exists():
        return cache_dir

    root_dir_to_delete = Path(xdg_cache_home) / "org"
    openml_logger.warning(
        "An old cache directory was found at '%s'. This directory is no longer used by "
        "OpenML-Python. To silence this warning you would need to delete the old cache "
        "directory. The cached files will then be located in '%s'.",
        root_dir_to_delete,
        cache_dir,
    )
    return Path(xdg_cache_home)


@dataclass
class OpenMLConfig:
    """Dataclass storing the OpenML configuration."""

    apikey: str = ""
    server: str = "https://www.openml.org/api/v1/xml"
    cachedir: Path = field(default_factory=_resolve_default_cache_dir)
    avoid_duplicate_runs: bool = False
    retry_policy: Literal["human", "robot"] = "human"
    connection_n_retries: int = 5
    show_progress: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "apikey" and value is not None and not isinstance(value, str):
            raise ValueError("apikey must be a string or None")

        super().__setattr__(name, value)


class OpenMLConfigManager:
    """The OpenMLConfigManager manages the configuration of the openml-python package."""

    def __init__(self) -> None:
        self.console_handler: logging.StreamHandler | None = None
        self.file_handler: logging.handlers.RotatingFileHandler | None = None

        self.OPENML_CACHE_DIR_ENV_VAR = "OPENML_CACHE_DIR"
        self.OPENML_SKIP_PARQUET_ENV_VAR = "OPENML_SKIP_PARQUET"
        self._TEST_SERVER_NORMAL_USER_KEY = "normaluser"

        self._user_path = Path("~").expanduser().absolute()

        self._config: OpenMLConfig = OpenMLConfig()
        # for legacy test `test_non_writable_home`
        self._defaults: dict[str, Any] = OpenMLConfig().__dict__.copy()
        self._root_cache_directory: Path = self._config.cachedir

        self.logger = logger
        self.openml_logger = openml_logger

        self._examples = self.ConfigurationForExamples(self)

        self._setup()

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._config, name):
            return getattr(self._config, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    _FIELDS = {  # noqa: RUF012
        "apikey",
        "server",
        "cachedir",
        "avoid_duplicate_runs",
        "retry_policy",
        "connection_n_retries",
        "show_progress",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        # during __init__ before _config exists
        if name in {
            "_config",
            "_root_cache_directory",
            "console_handler",
            "file_handler",
            "logger",
            "openml_logger",
            "_examples",
            "OPENML_CACHE_DIR_ENV_VAR",
            "OPENML_SKIP_PARQUET_ENV_VAR",
            "_TEST_SERVER_NORMAL_USER_KEY",
            "_user_path",
        }:
            return object.__setattr__(self, name, value)

        if name in self._FIELDS:
            # write into dataclass, not manager (prevents shadowing)
            if name == "cachedir":
                object.__setattr__(self, "_root_cache_directory", Path(value))
            object.__setattr__(self, "_config", replace(self._config, **{name: value}))
            return None

        object.__setattr__(self, name, value)
        return None

    def _create_log_handlers(self, create_file_handler: bool = True) -> None:  # noqa: FBT002
        if self.console_handler is not None or self.file_handler is not None:
            self.logger.debug("Requested to create log handlers, but they are already created.")
            return

        message_format = "[%(levelname)s] [%(asctime)s:%(name)s] %(message)s"
        output_formatter = logging.Formatter(message_format, datefmt="%H:%M:%S")

        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(output_formatter)

        if create_file_handler:
            one_mb = 2**20
            log_path = self._root_cache_directory / "openml_python.log"
            self.file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=one_mb,
                backupCount=1,
                delay=True,
            )
            self.file_handler.setFormatter(output_formatter)

    def _convert_log_levels(self, log_level: int) -> tuple[int, int]:
        openml_to_python = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        python_to_openml = {
            logging.DEBUG: 2,
            logging.INFO: 1,
            logging.WARNING: 0,
            logging.CRITICAL: 0,
            logging.ERROR: 0,
        }
        openml_level = python_to_openml.get(log_level, log_level)
        python_level = openml_to_python.get(log_level, log_level)
        return openml_level, python_level

    def _set_level_register_and_store(self, handler: logging.Handler, log_level: int) -> None:
        _oml_level, py_level = self._convert_log_levels(log_level)
        handler.setLevel(py_level)

        if self.openml_logger.level > py_level or self.openml_logger.level == logging.NOTSET:
            self.openml_logger.setLevel(py_level)

        if handler not in self.openml_logger.handlers:
            self.openml_logger.addHandler(handler)

    def set_console_log_level(self, console_output_level: int) -> None:
        """Set the log level for console output."""
        assert self.console_handler is not None
        self._set_level_register_and_store(self.console_handler, console_output_level)

    def set_file_log_level(self, file_output_level: int) -> None:
        """Set the log level for file output."""
        assert self.file_handler is not None
        self._set_level_register_and_store(self.file_handler, file_output_level)

    def get_server_base_url(self) -> str:
        """Get the base URL of the OpenML server (i.e., without /api)."""
        domain, _ = self._config.server.split("/api", maxsplit=1)
        return domain.replace("api", "www")

    def set_retry_policy(
        self, value: Literal["human", "robot"], n_retries: int | None = None
    ) -> None:
        """Set the retry policy for server connections."""
        default_retries_by_policy = {"human": 5, "robot": 50}

        if value not in default_retries_by_policy:
            raise ValueError(
                f"Detected retry_policy '{value}' but must be one of "
                f"{list(default_retries_by_policy.keys())}",
            )
        if n_retries is not None and not isinstance(n_retries, int):
            raise TypeError(
                f"`n_retries` must be of type `int` or `None` but is `{type(n_retries)}`."
            )

        if isinstance(n_retries, int) and n_retries < 1:
            raise ValueError(f"`n_retries` is '{n_retries}' but must be positive.")

        self._config = replace(
            self._config,
            retry_policy=value,
            connection_n_retries=(
                default_retries_by_policy[value] if n_retries is None else n_retries
            ),
        )

    def _handle_xdg_config_home_backwards_compatibility(self, xdg_home: str) -> Path:
        config_dir = Path(xdg_home) / "openml"

        backwards_compat_config_file = Path(xdg_home) / "config"
        if not backwards_compat_config_file.exists():
            return config_dir

        try:
            self._parse_config(backwards_compat_config_file)
        except Exception:  # noqa: BLE001
            return config_dir

        correct_config_location = config_dir / "config"
        try:
            shutil.copy(backwards_compat_config_file, correct_config_location)
            self.openml_logger.warning(
                "An openml configuration file was found at the old location "
                f"at {backwards_compat_config_file}. We have copied it to the new "
                f"location at {correct_config_location}. "
                "\nTo silence this warning please verify that the configuration file "
                f"at {correct_config_location} is correct and delete the file at "
                f"{backwards_compat_config_file}."
            )
            return config_dir
        except Exception as e:  # noqa: BLE001
            self.openml_logger.warning(
                "While attempting to perform a backwards compatible fix, we "
                f"failed to copy the openml config file at "
                f"{backwards_compat_config_file}' to {correct_config_location}"
                f"\n{type(e)}: {e}",
                "\n\nTo silence this warning, please copy the file "
                "to the new location and delete the old file at "
                f"{backwards_compat_config_file}.",
            )
            return backwards_compat_config_file

    def determine_config_file_path(self) -> Path:
        """Determine the path to the openml configuration file."""
        if platform.system().lower() == "linux":
            xdg_home = os.environ.get("XDG_CONFIG_HOME")
            if xdg_home is not None:
                config_dir = self._handle_xdg_config_home_backwards_compatibility(xdg_home)
            else:
                config_dir = Path("~", ".config", "openml")
        else:
            config_dir = Path("~") / ".openml"

        config_dir = Path(config_dir).expanduser().resolve()
        return config_dir / "config"

    def _parse_config(self, config_file: str | Path) -> dict[str, Any]:
        config_file = Path(config_file)
        config = configparser.RawConfigParser(defaults=OpenMLConfig().__dict__)  # type: ignore

        config_file_ = StringIO()
        config_file_.write("[FAKE_SECTION]\n")
        try:
            with config_file.open("r") as fh:
                for line in fh:
                    config_file_.write(line)
        except FileNotFoundError:
            self.logger.info(
                "No config file found at %s, using default configuration.", config_file
            )
        except OSError as e:
            self.logger.info("Error opening file %s: %s", config_file, e.args[0])
        config_file_.seek(0)
        config.read_file(config_file_)
        configuration = dict(config.items("FAKE_SECTION"))
        for boolean_field in ["avoid_duplicate_runs", "show_progress"]:
            if isinstance(config["FAKE_SECTION"][boolean_field], str):
                configuration[boolean_field] = config["FAKE_SECTION"].getboolean(boolean_field)  # type: ignore
        return configuration  # type: ignore

    def start_using_configuration_for_example(self) -> None:
        """Sets the configuration to connect to the test server with valid apikey."""
        return self._examples.start_using_configuration_for_example()

    def stop_using_configuration_for_example(self) -> None:
        """Store the configuration as it was before `start_use_example_configuration`."""
        return self._examples.stop_using_configuration_for_example()

    def _setup(self, config: dict[str, Any] | None = None) -> None:
        config_file = self.determine_config_file_path()
        config_dir = config_file.parent

        try:
            if not config_dir.exists():
                config_dir.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            self.openml_logger.warning(
                f"No permission to create OpenML directory at {config_dir}!"
                " This can result in OpenML-Python not working properly."
            )

        if config is None:
            config = self._parse_config(config_file)

        self._config = replace(
            self._config,
            apikey=config["apikey"],
            server=config["server"],
            show_progress=config["show_progress"],
            avoid_duplicate_runs=config["avoid_duplicate_runs"],
            retry_policy=config["retry_policy"],
            connection_n_retries=int(config["connection_n_retries"]),
        )

        user_defined_cache_dir = os.environ.get(self.OPENML_CACHE_DIR_ENV_VAR)
        if user_defined_cache_dir is not None:
            short_cache_dir = Path(user_defined_cache_dir)
        else:
            short_cache_dir = Path(config["cachedir"])

        self._root_cache_directory = short_cache_dir.expanduser().resolve()
        self._config = replace(self._config, cachedir=self._root_cache_directory)

        try:
            cache_exists = self._root_cache_directory.exists()
            if not cache_exists:
                self._root_cache_directory.mkdir(exist_ok=True, parents=True)
            self._create_log_handlers()
        except PermissionError:
            self.openml_logger.warning(
                f"No permission to create OpenML directory at {self._root_cache_directory}!"
                " This can result in OpenML-Python not working properly."
            )
            self._create_log_handlers(create_file_handler=False)

    def set_field_in_config_file(self, field: str, value: Any) -> None:
        """Set a field in the configuration file."""
        if not hasattr(OpenMLConfig(), field):
            raise ValueError(
                f"Field '{field}' is not valid and must be one of "
                f"'{OpenMLConfig().__dict__.keys()}'."
            )

        self._config = replace(self._config, **{field: value})
        config_file = self.determine_config_file_path()
        existing = self._parse_config(config_file)
        with config_file.open("w") as fh:
            for f in OpenMLConfig().__dict__:
                v = value if f == field else existing.get(f)
                if v is not None:
                    fh.write(f"{f} = {v}\n")

    def get_config_as_dict(self) -> dict[str, Any]:
        """Get the current configuration as a dictionary."""
        return self._config.__dict__.copy()

    def get_cache_directory(self) -> str:
        """Get the cache directory for the current server."""
        url_suffix = urlparse(self._config.server).netloc
        reversed_url_suffix = os.sep.join(url_suffix.split(".")[::-1])  # noqa: PTH118
        return os.path.join(self._root_cache_directory, reversed_url_suffix)  # noqa: PTH118

    def set_root_cache_directory(self, root_cache_directory: str | Path) -> None:
        """Set the root cache directory."""
        self._root_cache_directory = Path(root_cache_directory)
        self._config = replace(self._config, cachedir=self._root_cache_directory)

    @contextmanager
    def overwrite_config_context(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Overwrite the current configuration within a context manager."""
        existing_config = self.get_config_as_dict()
        merged_config = {**existing_config, **config}

        self._setup(merged_config)
        yield merged_config
        self._setup(existing_config)

    class ConfigurationForExamples:
        """Allows easy switching to and from a test configuration, used for examples."""

        _last_used_server = None
        _last_used_key = None
        _start_last_called = False

        def __init__(self, manager: OpenMLConfigManager):
            self._manager = manager
            self._test_apikey = manager._TEST_SERVER_NORMAL_USER_KEY
            self._test_server = "https://test.openml.org/api/v1/xml"

        def start_using_configuration_for_example(self) -> None:
            """Sets the configuration to connect to the test server with valid apikey.

            To configuration as was before this call is stored, and can be recovered
            by using the `stop_use_example_configuration` method.
            """
            if (
                self._start_last_called
                and self._manager._config.server == self._test_server
                and self._manager._config.apikey == self._test_apikey
            ):
                # Method is called more than once in a row without modifying the server or apikey.
                # We don't want to save the current test configuration as a last used configuration.
                return

            self._last_used_server = self._manager._config.server
            self._last_used_key = self._manager._config.apikey
            type(self)._start_last_called = True

            # Test server key for examples
            self._manager._config = replace(
                self._manager._config,
                server=self._test_server,
                apikey=self._test_apikey,
            )
            warnings.warn(
                f"Switching to the test server {self._test_server} to not upload results to "
                "the live server. Using the test server may result in reduced performance of the "
                "API!",
                stacklevel=2,
            )

        def stop_using_configuration_for_example(self) -> None:
            """Return to configuration as it was before `start_use_example_configuration`."""
            if not type(self)._start_last_called:
                # We don't want to allow this because it will (likely) result in the `server` and
                # `apikey` variables being set to None.
                raise RuntimeError(
                    "`stop_use_example_configuration` called without a saved config."
                    "`start_use_example_configuration` must be called first.",
                )

            self._manager._config = replace(
                self._manager._config,
                server=cast("str", self._last_used_server),
                apikey=cast("str", self._last_used_key),
            )
            type(self)._start_last_called = False


_config = OpenMLConfigManager()


def __getattr__(name: str) -> Any:
    return getattr(_config, name)
