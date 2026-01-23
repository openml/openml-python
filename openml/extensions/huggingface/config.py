"""Configuration for HuggingFace Hub integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openml


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub integration.

    Attributes
    ----------
    cache_dir : Path
        Directory to cache downloaded models from HuggingFace.
    default_commit_message : str
        Default commit message when uploading to HuggingFace.
    model_filename : str
        Filename for serialized model in HuggingFace repos.
    metadata_filename : str
        Filename for OpenML metadata in HuggingFace repos.
    """

    cache_dir: Path | None = None
    default_commit_message: str = "Upload from OpenML"
    model_filename: str = "model.pkl"
    metadata_filename: str = "openml_metadata.json"
    readme_filename: str = "README.md"

    def __post_init__(self) -> None:
        """Initialize cache directory."""
        if self.cache_dir is None:
            # Use OpenML cache directory + huggingface subdirectory
            self.cache_dir = Path(openml.config.get_cache_directory()) / "huggingface"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config = HuggingFaceConfig()


def get_config() -> HuggingFaceConfig:
    """Get the current HuggingFace integration configuration.

    Returns
    -------
    HuggingFaceConfig
        Current configuration object.
    """
    return _config


def set_cache_directory(path: str | Path) -> None:
    """Set the cache directory for HuggingFace downloads.

    Parameters
    ----------
    path : str or Path
        Path to cache directory.
    """
    _config.cache_dir = Path(path)
    _config.cache_dir.mkdir(parents=True, exist_ok=True)


def reset_config() -> None:
    """Reset configuration to defaults.

    Note: This recreates the configuration by reinitializing fields.
    """
    _config.cache_dir = None
    _config.default_commit_message = "Upload from OpenML"
    _config.model_filename = "model.pkl"
    _config.metadata_filename = "openml_metadata.json"
    _config.readme_filename = "README.md"
    _config.__post_init__()
