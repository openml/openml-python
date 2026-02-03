from __future__ import annotations

from pathlib import Path

from openml.__version__ import __version__


class MinIOClient:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.headers: dict[str, str] = {"user-agent": f"openml-python/{__version__}"}
