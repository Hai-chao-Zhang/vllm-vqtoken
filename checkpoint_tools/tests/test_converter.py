from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from vqtoken_checkpoint.converter import (
    PUBLIC_SOURCE_REPO,
    PUBLIC_SOURCE_REVISION,
    ConversionError,
    ModelShapeProfile,
    build_name_map,
    convert_checkpoint,
    map_tensor_name,
    read_safetensors_layout,
)

TINY_PROFILE = ModelShapeProfile(
    hidden_size=4,
    query_dim=3,
    feed_forward_size=8,
    projector_input_size=5,
    vocab_size=7,
    patch_size=2,
    vision_config_num_hidden_layers=3,
    vision_feature_layer=-2,
    vision_checkpoint_layer_count=2,
    language_layer_count=2,
)


def _write_synthetic_checkpoint(
    path: Path,
    *,
    omit: set[str] | None = None,
    shape_overrides: dict[str, list[int]] | None = None,
    extra_shapes: dict[str, list[int]] | None = None,
) -> bytes:
    omit = omit or set()
    shapes = TINY_PROFILE.required_shapes()
    for layer_index in range(TINY_PROFILE.language_layer_count):
        shapes[f"model.layers.{layer_index}.input_layernorm.weight"] = [TINY_PROFILE.hidden_size]
    for layer_index in range(TINY_PROFILE.vision_checkpoint_layer_count):
        shapes[
            "model.vision_tower.vision_tower.vision_model.encoder.layers."
            f"{layer_index}.layer_norm1.weight"
        ] = [TINY_PROFILE.projector_input_size]
    shapes.update(extra_shapes or {})
    shapes.update(shape_overrides or {})

    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        if name in omit:
            continue
        elements = 1
        for dimension in shape:
            elements *= dimension
        start = len(data)
        data.extend(bytes([(index * 17 + 3) % 256]) * elements)
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [start, len(data)],
        }

    compact_header = json.dumps(header, separators=(",", ":")).encode()
    compact_header += b" " * ((-len(compact_header)) % 8)
    path.write_bytes(struct.pack("<Q", len(compact_header)) + compact_header + data)
    return bytes(data)


class ConverterTests(unittest.TestCase):
    def test_streaming_conversion_preserves_tensor_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            destination = root / "model.safetensors"
            manifest_path = root / "conversion-manifest.json"
            expected_data = _write_synthetic_checkpoint(source)
            source_before = source.read_bytes()

            manifest = convert_checkpoint(
                source,
                destination,
                manifest_path,
                profile=TINY_PROFILE,
                expected_source_sha256=hashlib.sha256(source_before).hexdigest(),
                expected_source_size=len(source_before),
            )

            self.assertEqual(source.read_bytes(), source_before)
            converted_layout = read_safetensors_layout(destination)
            self.assertEqual(destination.read_bytes()[converted_layout.data_start :], expected_data)
            source_layout = read_safetensors_layout(source)
            self.assertNotEqual(source_layout.data_start, converted_layout.data_start)
            self.assertEqual(
                manifest["tensor_data"]["sha256"],
                hashlib.sha256(expected_data).hexdigest(),
            )
            self.assertTrue(manifest["tensor_data"]["byte_identical"])
            self.assertTrue(manifest["source"]["verified"])
            self.assertTrue(manifest["source"]["sha256_checked"])
            self.assertTrue(manifest["source"]["size_checked"])
            self.assertIsNone(manifest["source"]["repository"])
            self.assertIsNone(manifest["source"]["revision"])
            self.assertTrue(manifest["validation"]["one_to_one_mapping"])
            self.assertEqual(manifest["validation"]["vq_attention_tensor_count"], 20)
            self.assertEqual(
                manifest["validation"]["profile"]["vision_checkpoint_layer_count"],
                2,
            )
            self.assertEqual(
                manifest["validation"]["profile"]["language_layer_count"],
                2,
            )
            self.assertEqual(
                manifest["tensor_name_mapping"]["model.cross_attention.to_q_proj.weight"],
                "model.vq_attention.to_q_proj.weight",
            )
            self.assertIn(
                "model.multi_modal_projector.linear_1.weight",
                converted_layout.header,
            )
            self.assertIn(
                "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
                converted_layout.header,
            )
            self.assertIn(
                "model.language_model.embed_tokens.weight",
                converted_layout.header,
            )
            self.assertIn("model.image_newline", converted_layout.header)
            self.assertIn("lm_head.weight", converted_layout.header)
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_rejects_missing_attention_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            missing = "model.cross_attention.to_q_proj.bias"
            _write_synthetic_checkpoint(source, omit={missing})
            with self.assertRaisesRegex(ConversionError, "expected exactly 20"):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=TINY_PROFILE,
                )

    def test_rejects_wrong_attention_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            tensor = "model.cross_attention.to_q_proj.weight"
            _write_synthetic_checkpoint(source, shape_overrides={tensor: [4, 4]})
            with self.assertRaisesRegex(ConversionError, "shape mismatch"):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=TINY_PROFILE,
                )

    def test_rejects_missing_or_extra_vision_layer_indices(self) -> None:
        layer_prefix = "model.vision_tower.vision_tower.vision_model.encoder.layers."
        scenarios = {
            "missing": {
                "omit": {f"{layer_prefix}1.layer_norm1.weight"},
                "extra_shapes": None,
                "message": r"SigLIP checkpoint layer index set mismatch.*missing=\[1\]",
            },
            "extra": {
                "omit": None,
                "extra_shapes": {f"{layer_prefix}2.layer_norm1.weight": [5]},
                "message": r"SigLIP checkpoint layer index set mismatch.*unexpected=\[2\]",
            },
        }
        for name, scenario in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                source = root / "source.safetensors"
                _write_synthetic_checkpoint(
                    source,
                    omit=scenario["omit"],
                    extra_shapes=scenario["extra_shapes"],
                )
                with self.assertRaisesRegex(ConversionError, scenario["message"]):
                    convert_checkpoint(
                        source,
                        root / "output.safetensors",
                        root / "manifest.json",
                        profile=TINY_PROFILE,
                    )

    def test_rejects_missing_or_extra_language_layer_indices(self) -> None:
        scenarios = {
            "missing": {
                "omit": {"model.layers.1.input_layernorm.weight"},
                "extra_shapes": None,
                "message": r"language-model checkpoint layer index set mismatch.*missing=\[1\]",
            },
            "extra": {
                "omit": None,
                "extra_shapes": {"model.layers.2.input_layernorm.weight": [4]},
                "message": r"language-model checkpoint layer index set mismatch.*unexpected=\[2\]",
            },
        }
        for name, scenario in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                source = root / "source.safetensors"
                _write_synthetic_checkpoint(
                    source,
                    omit=scenario["omit"],
                    extra_shapes=scenario["extra_shapes"],
                )
                with self.assertRaisesRegex(ConversionError, scenario["message"]):
                    convert_checkpoint(
                        source,
                        root / "output.safetensors",
                        root / "manifest.json",
                        profile=TINY_PROFILE,
                    )

    def test_rejects_inconsistent_vision_layer_profile(self) -> None:
        inconsistent_profile = ModelShapeProfile(
            hidden_size=4,
            query_dim=3,
            feed_forward_size=8,
            projector_input_size=5,
            vocab_size=7,
            patch_size=2,
            vision_config_num_hidden_layers=4,
            vision_feature_layer=-2,
            vision_checkpoint_layer_count=2,
            language_layer_count=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            _write_synthetic_checkpoint(source)
            with self.assertRaisesRegex(ConversionError, "inconsistent vision layer profile"):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=inconsistent_profile,
                )

    def test_rejects_noncanonical_layer_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            _write_synthetic_checkpoint(
                source,
                extra_shapes={"model.layers.01.input_layernorm.weight": [4]},
            )
            with self.assertRaisesRegex(
                ConversionError,
                "invalid language-model checkpoint layer tensor names",
            ):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=TINY_PROFILE,
                )

    def test_rejects_unreviewed_tensor_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            _write_synthetic_checkpoint(
                source,
                extra_shapes={"model.extra_module.weight": [1]},
            )
            with self.assertRaisesRegex(ConversionError, "no reviewed mapping rule"):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=TINY_PROFILE,
                )

    def test_collision_detection_is_explicit(self) -> None:
        def colliding_mapper(name: str) -> tuple[str, str]:
            return "same.target", f"rule_for_{name}"

        with self.assertRaisesRegex(ConversionError, "collision"):
            build_name_map(["first", "second"], mapper=colliding_mapper)

    def test_public_mapping_rules_are_exact(self) -> None:
        cases = {
            "model.vision_tower.vision_tower.encoder.weight": (
                "model.vision_tower.encoder.weight",
                "embedded_siglip",
            ),
            "model.mm_projector.0.weight": (
                "model.multi_modal_projector.linear_1.weight",
                "projector_linear_1",
            ),
            "model.mm_projector.2.bias": (
                "model.multi_modal_projector.linear_2.bias",
                "projector_linear_2",
            ),
            "model.cross_attention.to_q_proj.weight": (
                "model.vq_attention.to_q_proj.weight",
                "vq_attention",
            ),
            "model.image_newline": ("model.image_newline", "image_newline"),
            "model.layers.0.self_attn.q_proj.weight": (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "language_model",
            ),
            "lm_head.weight": ("lm_head.weight", "lm_head"),
        }
        for source, target_and_rule in cases.items():
            with self.subTest(source=source):
                self.assertEqual(map_tensor_name(source), target_and_rule)

    def test_pinned_hash_mismatch_leaves_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            destination = root / "output.safetensors"
            manifest = root / "manifest.json"
            _write_synthetic_checkpoint(source)
            with self.assertRaisesRegex(ConversionError, "SHA-256 mismatch"):
                convert_checkpoint(
                    source,
                    destination,
                    manifest,
                    profile=TINY_PROFILE,
                    expected_source_sha256="0" * 64,
                )
            self.assertFalse(destination.exists())
            self.assertFalse(manifest.exists())

    def test_unverified_conversion_does_not_claim_public_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            destination = root / "output.safetensors"
            manifest_path = root / "manifest.json"
            _write_synthetic_checkpoint(source)

            manifest = convert_checkpoint(
                source,
                destination,
                manifest_path,
                profile=TINY_PROFILE,
            )

            self.assertFalse(manifest["source"]["verified"])
            self.assertFalse(manifest["source"]["sha256_checked"])
            self.assertFalse(manifest["source"]["size_checked"])
            self.assertIsNone(manifest["source"]["repository"])
            self.assertIsNone(manifest["source"]["revision"])

    def test_repository_provenance_requires_content_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            _write_synthetic_checkpoint(source)

            with self.assertRaisesRegex(
                ConversionError,
                "source repository provenance requires",
            ):
                convert_checkpoint(
                    source,
                    root / "output.safetensors",
                    root / "manifest.json",
                    profile=TINY_PROFILE,
                    source_repo=PUBLIC_SOURCE_REPO,
                    source_revision=PUBLIC_SOURCE_REVISION,
                )

    def test_refuses_to_overwrite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            destination = root / "output.safetensors"
            manifest = root / "manifest.json"
            _write_synthetic_checkpoint(source)
            destination.write_bytes(b"keep")
            with self.assertRaisesRegex(ConversionError, "refusing to overwrite"):
                convert_checkpoint(
                    source,
                    destination,
                    manifest,
                    profile=TINY_PROFILE,
                )
            self.assertEqual(destination.read_bytes(), b"keep")

    def test_requires_three_distinct_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.safetensors"
            destination = root / "output.safetensors"
            _write_synthetic_checkpoint(source)
            with self.assertRaisesRegex(ConversionError, "must be different"):
                convert_checkpoint(
                    source,
                    destination,
                    destination,
                    profile=TINY_PROFILE,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
