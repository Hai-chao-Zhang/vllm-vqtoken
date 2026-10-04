# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_vqtoken.semantics import (
    get_num_video_tokens,
    get_profile_frame_count,
    get_query_dim,
    validate_vqtoken_config,
)


def _config(**overrides):
    values = {
        "vqtoken_mode": "attention",
        "vqtoken_selection_method": "fixed",
        "vqtoken_num_clusters": 32,
        "vqtoken_canonicalize": False,
        "vqtoken_seed": 0,
        "vqtoken_max_iterations": 50,
        "vqtoken_attention_heads": 8,
        "vqtoken_query_dim": 729,
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "full",
        "vision_config": SimpleNamespace(
            image_size=384,
            patch_size=14,
            num_hidden_layers=27,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fixed_budget_includes_one_newline_token():
    config = _config()

    assert get_num_video_tokens(config, num_frames=32) == 33
    assert get_profile_frame_count(config) == 32


def test_more_frames_than_clusters_is_rejected():
    with pytest.raises(ValueError, match="num_frames <= 32"):
        get_num_video_tokens(_config(), num_frames=33)


@pytest.mark.parametrize("selection_method", ["elbow", "silhouette"])
def test_nonfixed_selection_is_rejected(selection_method: str):
    with pytest.raises(ValueError, match="static VQToken placeholder count"):
        validate_vqtoken_config(_config(vqtoken_selection_method=selection_method))


def test_released_cluster_ids_cannot_be_canonicalized():
    with pytest.raises(ValueError, match="vqtoken_canonicalize=false"):
        validate_vqtoken_config(_config(vqtoken_canonicalize=True))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vision_feature_layer", -1, "vision_feature_layer=-2"),
        (
            "vision_feature_select_strategy",
            "default",
            "vision_feature_select_strategy='full'",
        ),
    ],
)
def test_checkpoint_vision_feature_contract(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_vqtoken_config(_config(**{field: value}))


def test_checkpoint_vision_depth_contract():
    config = _config()
    config.vision_config.num_hidden_layers = 26

    with pytest.raises(ValueError, match="num_hidden_layers=27"):
        validate_vqtoken_config(config)


def test_query_width_must_match_patch_grid():
    assert get_query_dim(_config()) == 729
    with pytest.raises(ValueError, match="must match the vision patch grid"):
        get_query_dim(_config(vqtoken_query_dim=728))
