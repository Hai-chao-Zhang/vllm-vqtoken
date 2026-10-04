"""Convert the released VQToken checkpoint to a vLLM-serving layout.

The conversion is deliberately header-only: tensor names in the safetensors
JSON header are rewritten, while the tensor data region is copied byte for
byte in bounded chunks.  No tensor is materialized in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PUBLIC_SOURCE_REPO = "haichaozhang/VQ-Token-llava-ov-0.5b"
PUBLIC_SOURCE_REVISION = "ab20666864dbe71c931a9c1236190e90493f02fe"
PUBLIC_SOURCE_SHA256 = "593c6ffe2df774aaf2baa357e56bb1c33379584ae5a66d7c7cea5608a12bd5c7"
PUBLIC_SOURCE_SIZE = 2_081_132_000
CONVERSION_SCHEMA_VERSION = 2
COPY_CHUNK_SIZE = 16 * 1024 * 1024
MAX_HEADER_SIZE = 128 * 1024 * 1024

_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class ConversionError(ValueError):
    """Raised when a checkpoint cannot be converted without ambiguity."""


@dataclass(frozen=True)
class ModelShapeProfile:
    """Dimensions needed to validate the public VQToken adapter."""

    hidden_size: int = 896
    query_dim: int = 729
    feed_forward_size: int = 2048
    projector_input_size: int = 1152
    vocab_size: int = 151_936
    patch_size: int = 14
    vision_config_num_hidden_layers: int = 27
    vision_feature_layer: int = -2
    vision_checkpoint_layer_count: int = 26
    language_layer_count: int = 24

    def attention_shapes(self) -> dict[str, list[int]]:
        prefix = "model.cross_attention."
        layer = f"{prefix}transformer_decoder.layers.0."
        hidden = self.hidden_size
        feed_forward = self.feed_forward_size
        one_dimensional = [hidden]
        return {
            f"{prefix}to_q_proj.weight": [hidden, self.query_dim],
            f"{prefix}to_q_proj.bias": one_dimensional,
            f"{layer}self_attn.in_proj_weight": [3 * hidden, hidden],
            f"{layer}self_attn.in_proj_bias": [3 * hidden],
            f"{layer}self_attn.out_proj.weight": [hidden, hidden],
            f"{layer}self_attn.out_proj.bias": one_dimensional,
            f"{layer}multihead_attn.in_proj_weight": [3 * hidden, hidden],
            f"{layer}multihead_attn.in_proj_bias": [3 * hidden],
            f"{layer}multihead_attn.out_proj.weight": [hidden, hidden],
            f"{layer}multihead_attn.out_proj.bias": one_dimensional,
            f"{layer}linear1.weight": [feed_forward, hidden],
            f"{layer}linear1.bias": [feed_forward],
            f"{layer}linear2.weight": [hidden, feed_forward],
            f"{layer}linear2.bias": one_dimensional,
            f"{layer}norm1.weight": one_dimensional,
            f"{layer}norm1.bias": one_dimensional,
            f"{layer}norm2.weight": one_dimensional,
            f"{layer}norm2.bias": one_dimensional,
            f"{layer}norm3.weight": one_dimensional,
            f"{layer}norm3.bias": one_dimensional,
        }

    def required_shapes(self) -> dict[str, list[int]]:
        hidden = self.hidden_size
        projector_input = self.projector_input_size
        required = self.attention_shapes()
        required.update(
            {
                "model.mm_projector.0.weight": [hidden, projector_input],
                "model.mm_projector.0.bias": [hidden],
                "model.mm_projector.2.weight": [hidden, hidden],
                "model.mm_projector.2.bias": [hidden],
                "model.image_newline": [hidden],
                "model.embed_tokens.weight": [self.vocab_size, hidden],
                "model.norm.weight": [hidden],
                "lm_head.weight": [self.vocab_size, hidden],
                (
                    "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"
                ): [projector_input, 3, self.patch_size, self.patch_size],
                (
                    "model.vision_tower.vision_tower.vision_model.embeddings."
                    "position_embedding.weight"
                ): [self.query_dim, projector_input],
            }
        )
        return required


PUBLIC_PROFILE = ModelShapeProfile()


@dataclass(frozen=True)
class SafetensorsLayout:
    path: Path
    header: dict[str, Any]
    header_prefix: bytes
    raw_header: bytes
    data_start: int
    data_size: int


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"duplicate JSON key in safetensors header: {key!r}")
        result[key] = value
    return result


def _tensor_nbytes(dtype: str, shape: Sequence[int]) -> int:
    if dtype not in _DTYPE_SIZES:
        raise ConversionError(f"unsupported safetensors dtype: {dtype!r}")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ConversionError(f"invalid tensor dimension: {dimension!r}")
        elements *= dimension
    return elements * _DTYPE_SIZES[dtype]


def _validate_header(header: Mapping[str, Any], data_size: int) -> None:
    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        ):
            raise ConversionError("safetensors __metadata__ must map strings to strings")

    intervals: list[tuple[int, int, str]] = []
    tensor_count = 0
    for name, specification in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(name, str) or not name:
            raise ConversionError("tensor names must be non-empty strings")
        if not isinstance(specification, dict):
            raise ConversionError(f"tensor specification is not an object: {name!r}")
        if set(specification) != {"dtype", "shape", "data_offsets"}:
            raise ConversionError(f"unexpected tensor specification fields: {name!r}")
        dtype = specification["dtype"]
        shape = specification["shape"]
        offsets = specification["data_offsets"]
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ConversionError(f"invalid dtype or shape for tensor: {name!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ConversionError(f"invalid data_offsets for tensor: {name!r}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ConversionError(f"out-of-range data_offsets for tensor: {name!r}")
        expected_size = _tensor_nbytes(dtype, shape)
        if end - start != expected_size:
            raise ConversionError(
                f"byte length mismatch for {name!r}: offsets contain {end - start}, "
                f"shape and dtype require {expected_size}"
            )
        intervals.append((start, end, name))

    if tensor_count == 0:
        raise ConversionError("safetensors checkpoint contains no tensors")

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise ConversionError(f"tensor data has a {relation} before {name!r}")
        cursor = end
    if cursor != data_size:
        raise ConversionError("tensor data does not cover the complete safetensors data region")


def read_safetensors_layout(path: str | os.PathLike[str]) -> SafetensorsLayout:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ConversionError(f"checkpoint is not a regular file: {checkpoint_path}")
    file_size = checkpoint_path.stat().st_size
    if file_size < 9:
        raise ConversionError("file is too small to be a safetensors checkpoint")

    with checkpoint_path.open("rb") as checkpoint:
        prefix = checkpoint.read(8)
        if len(prefix) != 8:
            raise ConversionError("truncated safetensors header length")
        (header_size,) = struct.unpack("<Q", prefix)
        if header_size < 2 or header_size > MAX_HEADER_SIZE:
            raise ConversionError(f"unsafe safetensors header size: {header_size}")
        if 8 + header_size >= file_size:
            raise ConversionError("safetensors header leaves no tensor data")
        raw_header = checkpoint.read(header_size)
        if len(raw_header) != header_size:
            raise ConversionError("truncated safetensors header")

    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError("invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise ConversionError("safetensors header must be a JSON object")

    data_start = 8 + header_size
    data_size = file_size - data_start
    _validate_header(header, data_size)
    return SafetensorsLayout(
        path=checkpoint_path,
        header=header,
        header_prefix=prefix,
        raw_header=raw_header,
        data_start=data_start,
        data_size=data_size,
    )


def map_tensor_name(name: str) -> tuple[str, str]:
    """Return ``(target_name, mapping_rule)`` for one released tensor."""

    mappings = (
        (
            "model.vision_tower.vision_tower.",
            "model.vision_tower.",
            "embedded_siglip",
        ),
        (
            "model.mm_projector.0.",
            "model.multi_modal_projector.linear_1.",
            "projector_linear_1",
        ),
        (
            "model.mm_projector.2.",
            "model.multi_modal_projector.linear_2.",
            "projector_linear_2",
        ),
        (
            "model.cross_attention.",
            "model.vq_attention.",
            "vq_attention",
        ),
    )
    for source_prefix, target_prefix, rule in mappings:
        if name.startswith(source_prefix):
            suffix = name[len(source_prefix) :]
            if not suffix:
                raise ConversionError(f"tensor name has no suffix: {name!r}")
            return target_prefix + suffix, rule

    if name == "model.image_newline":
        return name, "image_newline"
    if name.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")):
        return "model.language_model." + name[len("model.") :], "language_model"
    if name.startswith("lm_head.") and len(name) > len("lm_head."):
        return name, "lm_head"

    raise ConversionError(f"no reviewed mapping rule for tensor: {name!r}")


def build_name_map(
    names: Iterable[str],
    mapper: Callable[[str], tuple[str, str]] = map_tensor_name,
) -> tuple[dict[str, str], dict[str, int]]:
    name_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}
    rule_counts: dict[str, int] = {}
    for source_name in names:
        target_name, rule = mapper(source_name)
        if target_name in reverse_map:
            raise ConversionError(
                "tensor-name collision: "
                f"{source_name!r} and {reverse_map[target_name]!r} both map to {target_name!r}"
            )
        name_map[source_name] = target_name
        reverse_map[target_name] = source_name
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
    if len(name_map) != len(reverse_map):
        raise ConversionError("tensor mapping is not one-to-one")
    return name_map, rule_counts


def _tensor_specs(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: specification for name, specification in header.items() if name != "__metadata__"}


def _validate_layer_indices(
    tensors: Mapping[str, Any],
    *,
    prefix: str,
    expected_count: int,
    label: str,
) -> None:
    indices: set[int] = set()
    malformed: list[str] = []
    for name in tensors:
        if not name.startswith(prefix):
            continue
        segment = name[len(prefix) :].partition(".")[0]
        if (
            not segment
            or not segment.isascii()
            or not segment.isdigit()
            or segment != str(int(segment))
        ):
            malformed.append(name)
            continue
        indices.add(int(segment))

    if malformed:
        raise ConversionError(f"invalid {label} layer tensor names: {sorted(malformed)}")

    expected = set(range(expected_count))
    if indices != expected:
        missing = sorted(expected - indices)
        unexpected = sorted(indices - expected)
        raise ConversionError(
            f"{label} layer index set mismatch: expected 0..{expected_count - 1}; "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_source_shapes(
    header: Mapping[str, Any], profile: ModelShapeProfile = PUBLIC_PROFILE
) -> None:
    tensors = _tensor_specs(header)
    expected_attention = profile.attention_shapes()
    actual_attention = {name for name in tensors if name.startswith("model.cross_attention.")}
    if actual_attention != set(expected_attention):
        missing = sorted(set(expected_attention) - actual_attention)
        unexpected = sorted(actual_attention - set(expected_attention))
        raise ConversionError(
            "released VQ-Attention key set mismatch: "
            f"expected exactly 20 tensors; missing={missing}, unexpected={unexpected}"
        )

    expected_vllm_vision_layers = (
        profile.vision_config_num_hidden_layers + profile.vision_feature_layer + 1
    )
    if expected_vllm_vision_layers != profile.vision_checkpoint_layer_count:
        raise ConversionError(
            "inconsistent vision layer profile: "
            f"config declares {profile.vision_config_num_hidden_layers} layers and selects "
            f"feature layer {profile.vision_feature_layer}, so vLLM expects "
            f"{expected_vllm_vision_layers} checkpoint layers, not "
            f"{profile.vision_checkpoint_layer_count}"
        )

    _validate_layer_indices(
        tensors,
        prefix="model.vision_tower.vision_tower.vision_model.encoder.layers.",
        expected_count=profile.vision_checkpoint_layer_count,
        label="SigLIP checkpoint",
    )
    _validate_layer_indices(
        tensors,
        prefix="model.layers.",
        expected_count=profile.language_layer_count,
        label="language-model checkpoint",
    )

    for name, expected_shape in profile.required_shapes().items():
        if name not in tensors:
            raise ConversionError(f"required released tensor is missing: {name!r}")
        actual_shape = tensors[name].get("shape")
        if actual_shape != expected_shape:
            raise ConversionError(
                f"shape mismatch for {name!r}: expected {expected_shape}, got {actual_shape}"
            )


def _encode_header(header: Mapping[str, Any]) -> tuple[bytes, bytes]:
    compact = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(compact)) % 8
    raw_header = compact + (b" " * padding)
    return struct.pack("<Q", len(raw_header)), raw_header


def _rewrite_header(header: Mapping[str, Any], name_map: Mapping[str, str]) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    if "__metadata__" in header:
        rewritten["__metadata__"] = header["__metadata__"]
    for source_name, specification in header.items():
        if source_name == "__metadata__":
            continue
        rewritten[name_map[source_name]] = specification
    if len(_tensor_specs(rewritten)) != len(name_map):
        raise ConversionError("rewritten header lost one or more tensors")
    return rewritten


def _write_json_atomically(path: Path, payload: Mapping[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise ConversionError(f"refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _hash_checkpoint_and_data(layout: SafetensorsLayout) -> tuple[str, str]:
    checkpoint_hasher = hashlib.sha256()
    data_hasher = hashlib.sha256()
    with layout.path.open("rb") as checkpoint:
        header = checkpoint.read(layout.data_start)
        if header != layout.header_prefix + layout.raw_header:
            raise ConversionError("checkpoint changed after its header was validated")
        checkpoint_hasher.update(header)
        read_data = 0
        while True:
            chunk = checkpoint.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            read_data += len(chunk)
            checkpoint_hasher.update(chunk)
            data_hasher.update(chunk)
    if read_data != layout.data_size:
        raise ConversionError(f"hashed {read_data} data bytes, expected {layout.data_size}")
    return checkpoint_hasher.hexdigest(), data_hasher.hexdigest()


def convert_checkpoint(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    profile: ModelShapeProfile = PUBLIC_PROFILE,
    expected_source_sha256: str | None = None,
    expected_source_size: int | None = None,
    source_repo: str | None = None,
    source_revision: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Convert one safetensors file without materializing tensor data."""

    source_path = Path(source)
    destination_path = Path(destination)
    manifest_output_path = Path(manifest_path)
    resolved_paths = {
        source_path.resolve(),
        destination_path.resolve(),
        manifest_output_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ConversionError("source, destination, and manifest must be different files")
    if (source_repo is None) != (source_revision is None):
        raise ConversionError("source_repo and source_revision must be provided together")
    source_verified = expected_source_sha256 is not None and expected_source_size is not None
    if source_repo is not None and not source_verified:
        raise ConversionError(
            "source repository provenance requires both expected_source_sha256 "
            "and expected_source_size"
        )
    if destination_path.exists() and not force:
        raise ConversionError(f"refusing to overwrite existing checkpoint: {destination_path}")
    if manifest_output_path.exists() and not force:
        raise ConversionError(f"refusing to overwrite existing manifest: {manifest_output_path}")

    layout = read_safetensors_layout(source_path)
    if expected_source_size is not None and layout.path.stat().st_size != expected_source_size:
        raise ConversionError(
            f"source size mismatch: expected {expected_source_size}, "
            f"got {layout.path.stat().st_size}"
        )
    validate_source_shapes(layout.header, profile)
    source_tensors = _tensor_specs(layout.header)
    name_map, rule_counts = build_name_map(source_tensors)
    rewritten_header = _rewrite_header(layout.header, name_map)
    output_prefix, output_header = _encode_header(rewritten_header)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_name)
    source_hasher = hashlib.sha256()
    data_hasher = hashlib.sha256()
    try:
        with layout.path.open("rb") as source_file, os.fdopen(descriptor, "wb") as output_file:
            current_header = source_file.read(layout.data_start)
            if current_header != layout.header_prefix + layout.raw_header:
                raise ConversionError("source checkpoint changed after its header was validated")
            source_hasher.update(layout.header_prefix)
            source_hasher.update(layout.raw_header)
            output_file.write(output_prefix)
            output_file.write(output_header)

            copied = 0
            while True:
                chunk = source_file.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                copied += len(chunk)
                source_hasher.update(chunk)
                data_hasher.update(chunk)
                output_file.write(chunk)
            if copied != layout.data_size:
                raise ConversionError(f"copied {copied} data bytes, expected {layout.data_size}")
            output_file.flush()
            os.fsync(output_file.fileno())

        source_digest = source_hasher.hexdigest()
        if expected_source_sha256 is not None and source_digest != expected_source_sha256:
            raise ConversionError(
                f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_digest}"
            )

        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    try:
        converted_layout = read_safetensors_layout(destination_path)
        target_tensors = _tensor_specs(converted_layout.header)
        if set(target_tensors) != set(name_map.values()):
            raise ConversionError("post-write target tensor set does not match the name map")
        output_digest, output_data_digest = _hash_checkpoint_and_data(converted_layout)
        source_data_digest = data_hasher.hexdigest()
        if output_data_digest != source_data_digest:
            raise ConversionError("converted tensor data is not byte-identical to the source")
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise

    manifest: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "conversion": "safetensors-header-rename-and-bytewise-data-copy",
        "source": {
            "repository": source_repo,
            "revision": source_revision,
            "verified": source_verified,
            "sha256_checked": expected_source_sha256 is not None,
            "size_checked": expected_source_size is not None,
            "file": source_path.name,
            "size": layout.data_start + layout.data_size,
            "sha256": source_digest,
            "header_sha256": hashlib.sha256(layout.raw_header).hexdigest(),
        },
        "output": {
            "file": destination_path.name,
            "size": destination_path.stat().st_size,
            "sha256": output_digest,
            "header_sha256": hashlib.sha256(output_header).hexdigest(),
        },
        "tensor_data": {
            "size": layout.data_size,
            "sha256": source_data_digest,
            "byte_identical": True,
        },
        "validation": {
            "tensor_count": len(source_tensors),
            "vq_attention_tensor_count": 20,
            "profile": asdict(profile),
            "one_to_one_mapping": len(name_map) == len(set(name_map.values())),
            "rule_counts": dict(sorted(rule_counts.items())),
        },
        "tensor_name_mapping": dict(sorted(name_map.items())),
    }
    try:
        _write_json_atomically(manifest_output_path, manifest, force=force)
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    return manifest


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded released model.safetensors")
    parser.add_argument("destination", type=Path, help="converted output model.safetensors")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--unverified-source",
        action="store_true",
        help=(
            "disable the pinned public file size and SHA-256 checks; intended only "
            "for development fixtures"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    expected_sha256 = None if arguments.unverified_source else PUBLIC_SOURCE_SHA256
    expected_size = None if arguments.unverified_source else PUBLIC_SOURCE_SIZE
    manifest = convert_checkpoint(
        arguments.source,
        arguments.destination,
        arguments.manifest,
        expected_source_sha256=expected_sha256,
        expected_source_size=expected_size,
        source_repo=None if arguments.unverified_source else PUBLIC_SOURCE_REPO,
        source_revision=(None if arguments.unverified_source else PUBLIC_SOURCE_REVISION),
        force=arguments.force,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.destination),
                "output_sha256": manifest["output"]["sha256"],
                "manifest": str(arguments.manifest),
                "tensor_count": manifest["validation"]["tensor_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
