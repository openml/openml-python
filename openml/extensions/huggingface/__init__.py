"""
HuggingFace Hub integration for OpenML.
Enables bidirectional model sharing between OpenML and HuggingFace Hub.
"""

from .functions import (
    download_flow_from_huggingface,
    upload_flow_to_huggingface,
)

__all__ = [
    "download_flow_from_huggingface",
    "upload_flow_to_huggingface",
]
