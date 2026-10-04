from __future__ import annotations

import json
import unittest
from pathlib import Path

from vqtoken_checkpoint import get_metadata_dir
from vqtoken_checkpoint.converter import PUBLIC_PROFILE

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"


class MetadataTemplateTests(unittest.TestCase):
    def test_metadata_bundle_is_discoverable(self) -> None:
        self.assertEqual(get_metadata_dir().resolve(), METADATA.resolve())

    def test_config_matches_released_fixed_vq_profile(self) -> None:
        config = json.loads((METADATA / "config.json").read_text())
        self.assertEqual(
            config["architectures"],
            ["VQTokenLlavaOnevisionForConditionalGeneration"],
        )
        self.assertEqual(config["model_type"], "llava_onevision")
        self.assertEqual(config["vision_feature_layer"], PUBLIC_PROFILE.vision_feature_layer)
        self.assertEqual(config["vision_feature_select_strategy"], "full")
        self.assertEqual(config["vqtoken_mode"], "attention")
        self.assertEqual(config["vqtoken_selection_method"], "fixed")
        self.assertEqual(config["vqtoken_num_clusters"], 32)
        self.assertEqual(config["vqtoken_max_frames"], 32)
        self.assertEqual(config["vqtoken_query_dim"], 729)
        self.assertEqual(config["vqtoken_attention_heads"], 8)
        self.assertEqual(config["vqtoken_seed"], 0)
        self.assertNotIn("vqtoken_kmeans_seed", config)
        self.assertFalse(config["vqtoken_canonicalize"])
        self.assertEqual(config["transformers_version"], "5.17.0")

        text = config["text_config"]
        self.assertEqual(text["model_type"], "qwen2")
        self.assertEqual(text["hidden_size"], 896)
        self.assertEqual(text["vocab_size"], 151936)
        self.assertFalse(text["tie_word_embeddings"])
        self.assertTrue(text["use_cache"])
        self.assertFalse(config["tie_word_embeddings"])

        vision = config["vision_config"]
        self.assertEqual(vision["model_type"], "siglip_vision_model")
        self.assertEqual(vision["hidden_size"], 1152)
        self.assertEqual(vision["image_size"], 384)
        self.assertEqual(vision["patch_size"], 14)
        # The public tower declares 27 layers and removes the last one. With
        # vision_feature_layer=-2, vLLM therefore constructs the 26 layers
        # (indices 0-25) present in the released checkpoint.
        self.assertEqual(
            vision["num_hidden_layers"],
            PUBLIC_PROFILE.vision_config_num_hidden_layers,
        )
        self.assertEqual(
            vision["num_hidden_layers"] + config["vision_feature_layer"] + 1,
            PUBLIC_PROFILE.vision_checkpoint_layer_count,
        )
        self.assertEqual(text["num_hidden_layers"], PUBLIC_PROFILE.language_layer_count)

    def test_processor_token_contract_is_consistent(self) -> None:
        config = json.loads((METADATA / "config.json").read_text())
        processor = json.loads((METADATA / "processor_config.json").read_text())
        image = json.loads((METADATA / "preprocessor_config.json").read_text())
        video = json.loads((METADATA / "video_preprocessor_config.json").read_text())
        compatibility_video = json.loads(
            (METADATA / "video_processor" / "preprocessor_config.json").read_text()
        )

        self.assertEqual(config["image_token_index"], 151646)
        self.assertEqual(config["video_token_index"], 151647)
        self.assertEqual(processor["image_token"], "<image>")
        self.assertEqual(processor["video_token"], "<video>")
        self.assertEqual(processor["num_image_tokens"], 729)
        self.assertEqual(image["size"], {"height": 384, "width": 384})
        self.assertEqual(video["size"], image["size"])
        self.assertEqual(compatibility_video["size"], image["size"])
        self.assertFalse(video["do_sample_frames"])
        self.assertEqual(len(config["image_grid_pinpoints"]), 36)
        self.assertEqual(config["image_grid_pinpoints"], image["image_grid_pinpoints"])

    def test_model_card_requires_hash_substitution(self) -> None:
        template = (METADATA / "MODEL_CARD.template.md").read_text()
        for placeholder in (
            "{{SOURCE_REVISION}}",
            "{{SOURCE_SHA256}}",
            "{{OUTPUT_SHA256}}",
            "{{DATA_SHA256}}",
        ):
            self.assertIn(placeholder, template)
        self.assertIn("released fixed-budget path", template)
        self.assertNotIn("unpublished", template)


if __name__ == "__main__":
    unittest.main()
