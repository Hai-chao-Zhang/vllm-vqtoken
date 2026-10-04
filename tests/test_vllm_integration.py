# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import vllm  # noqa: F401
except ImportError:
    if os.environ.get("VQTOKEN_REQUIRE_VLLM") == "1":
        raise
    pytest.skip(
        "vLLM is not installed; set VQTOKEN_REQUIRE_VLLM=1 to make this fatal",
        allow_module_level=True,
    )

from vllm_vqtoken.model import (  # noqa: E402
    VQTokenLlavaOnevisionForConditionalGeneration,
    VQTokenProcessingInfo,
)


def _config():
    return SimpleNamespace(
        vqtoken_mode="attention",
        vqtoken_selection_method="fixed",
        vqtoken_num_clusters=32,
        vqtoken_query_dim=729,
        vqtoken_attention_heads=8,
        vqtoken_max_iterations=50,
        vqtoken_canonicalize=False,
        vqtoken_seed=0,
        vision_feature_layer=-2,
        vision_feature_select_strategy="full",
        vision_config=SimpleNamespace(
            image_size=384,
            patch_size=14,
            num_hidden_layers=27,
        ),
    )


def test_profile_uses_all_frames_and_model_requests_sequential_video(monkeypatch):
    info = object.__new__(VQTokenProcessingInfo)
    monkeypatch.setattr(info, "get_hf_config", _config)

    assert (
        info.get_num_video_tokens(
            image_width=384,
            image_height=384,
            num_frames=32,
        )
        == 33
    )
    assert info.get_num_frames_with_most_features(8192, {"video": 1}) == 32
    assert (
        VQTokenLlavaOnevisionForConditionalGeneration.requires_sequential_video_encoding
        is True
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "model.vq_attention.to_q_proj.weight",
            "vq_attention.to_q_proj.weight",
        ),
        (
            "model.cross_attention.to_q_proj.weight",
            "vq_attention.to_q_proj.weight",
        ),
        (
            "model.vision_tower.embeddings.patch_embedding.weight",
            "vision_tower.embeddings.patch_embedding.weight",
        ),
        (
            "model.multi_modal_projector.linear_1.weight",
            "multi_modal_projector.linear_1.weight",
        ),
        (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
        ),
        ("lm_head.weight", "language_model.lm_head.weight"),
    ],
)
def test_hf_weight_prefix_mapping(source: str, target: str):
    mapper = VQTokenLlavaOnevisionForConditionalGeneration.hf_to_vllm_mapper

    # ``map_name`` became public after vLLM 0.30.0. ``apply_list`` exercises
    # the same supported mapping path across the stable release and newer
    # revisions without reaching into the private ``_map_name`` helper.
    assert mapper.apply_list([source]) == [target]


def test_full_checkpoint_loads_on_meta_when_requested():
    """Validate every converted weight without allocating the model on CPU.

    This is intentionally opt-in because the public checkpoint is about 2 GB.
    It checks topology and weight routing, not CUDA inference correctness.
    """

    checkpoint_dir_value = os.environ.get("VQTOKEN_CHECKPOINT_DIR")
    if checkpoint_dir_value is None:
        pytest.skip("set VQTOKEN_CHECKPOINT_DIR to run the full weight-load test")

    checkpoint_dir = Path(checkpoint_dir_value).resolve()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if not checkpoint_path.is_file():
        pytest.fail(f"missing converted checkpoint: {checkpoint_path}")

    import torch
    from safetensors import safe_open
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_vqtoken import register

    register()
    model_config = ModelConfig(
        model=str(checkpoint_dir),
        tokenizer=str(checkpoint_dir),
        dtype="bfloat16",
        enforce_eager=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=32),
        device_config=DeviceConfig(device="cpu"),
    )

    try:
        with tempfile.TemporaryDirectory(prefix="vqtoken-gloo-") as temp_dir:
            init_path = Path(temp_dir) / "init"
            with set_current_vllm_config(vllm_config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"file://{init_path}",
                    local_rank=0,
                    backend="gloo",
                )
                initialize_model_parallel(1, 1)
                with (
                    set_default_torch_dtype(model_config.dtype),
                    torch.device("meta"),
                ):
                    model = VQTokenLlavaOnevisionForConditionalGeneration(
                        vllm_config=vllm_config
                    )

                with safe_open(
                    checkpoint_path,
                    framework="pt",
                    device="cpu",
                ) as checkpoint:
                    source_names = list(checkpoint.keys())
                    loaded_names = model.load_weights(
                        (name, checkpoint.get_tensor(name)) for name in source_names
                    )

        parameter_names = set(dict(model.named_parameters()))
        assert len(source_names) == 737
        assert len(parameter_names) == 511
        assert loaded_names == parameter_names
        assert sum(name.startswith("vq_attention.") for name in parameter_names) == 20
        assert len(model.vision_tower.vision_model.encoder.layers) == 26
        assert model.vision_tower.vision_model.post_layernorm is None
    finally:
        cleanup_dist_env_and_memory()
