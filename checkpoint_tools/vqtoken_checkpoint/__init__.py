"""Streaming tools for the released VQToken vLLM checkpoint conversion."""

from importlib import metadata as importlib_metadata
from pathlib import Path

from .converter import (
    PUBLIC_PROFILE,
    PUBLIC_SOURCE_REVISION,
    PUBLIC_SOURCE_SHA256,
    PUBLIC_SOURCE_SIZE,
    ConversionError,
    ModelShapeProfile,
    convert_checkpoint,
    map_tensor_name,
    read_safetensors_layout,
)

_DISTRIBUTION_NAME = "vqtoken-vllm-checkpoint-tools"
_METADATA_SENTINEL = "share/vqtoken-vllm-checkpoint-tools/metadata/config.json"


def get_metadata_dir() -> Path:
    """Return the source-tree or installed metadata-template directory."""

    source_tree = Path(__file__).resolve().parents[1] / "metadata"
    if (source_tree / "config.json").is_file():
        return source_tree

    try:
        distribution = importlib_metadata.distribution(_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError as error:
        raise FileNotFoundError("VQToken metadata templates are not installed") from error

    for installed_file in distribution.files or ():
        if installed_file.as_posix().endswith(_METADATA_SENTINEL):
            metadata_dir = Path(distribution.locate_file(installed_file)).parent
            if (metadata_dir / "config.json").is_file():
                return metadata_dir

    raise FileNotFoundError("VQToken metadata templates are missing from the package")


__all__ = [
    "PUBLIC_PROFILE",
    "PUBLIC_SOURCE_REVISION",
    "PUBLIC_SOURCE_SHA256",
    "PUBLIC_SOURCE_SIZE",
    "ConversionError",
    "ModelShapeProfile",
    "convert_checkpoint",
    "get_metadata_dir",
    "map_tensor_name",
    "read_safetensors_layout",
]
