from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

openml_logger = logging.getLogger("openml")

# Default values (see also https://github.com/openml/OpenML/wiki/Client-API-Standards)
_user_path = Path("~").expanduser().absolute()


def _resolve_default_cache_dir() -> Path:
    user_defined_cache_dir = os.environ.get("OPENML_CACHE_DIR")
    if user_defined_cache_dir is not None:
        return Path(user_defined_cache_dir)

    if platform.system().lower() != "linux":
        return _user_path / ".openml"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home is None:
        return Path("~", ".cache", "openml")

    # This is the proper XDG_CACHE_HOME directory, but
    # we unfortunately had a problem where we used XDG_CACHE_HOME/org,
    # we check heuristically if this old directory still exists and issue
    # a warning if it does. There's too much data to move to do this for the user.

    # The new cache directory exists
    cache_dir = Path(xdg_cache_home) / "openml"
    if cache_dir.exists():
        return cache_dir

    # The old cache directory *does not* exist
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
