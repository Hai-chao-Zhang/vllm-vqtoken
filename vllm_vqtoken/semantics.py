# SPDX-License-Identifier: Apache-2.0

"""Configuration rules shared by the plugin and lightweight tests."""

from __future__ import annotations

from typing import Any

DEFAULT_NUM_CLUSTERS = 32
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_ATTENTION_HEADS = 8


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def get_num_clusters(config: object) -> int:
    """Read the static codebook size, accepting the released legacy name."""

    value = getattr(
        config,
        "vqtoken_num_clusters",
        getattr(config, "vqtoken_max_clusters", DEFAULT_NUM_CLUSTERS),
    )
    return _positive_int(value, "vqtoken_num_clusters")


def get_max_iterations(config: object) -> int:
    """Read the fixed spherical K-Means iteration budget."""

    value = getattr(config, "vqtoken_max_iterations", DEFAULT_MAX_ITERATIONS)
    return _positive_int(value, "vqtoken_max_iterations")


def get_attention_heads(config: object) -> int:
    """Read the number of heads in the released VQ-Attention module."""

    value = getattr(config, "vqtoken_attention_heads", DEFAULT_ATTENTION_HEADS)
    return _positive_int(value, "vqtoken_attention_heads")


def get_query_dim(config: object) -> int:
    """Validate the assignment-map width against the SigLIP patch grid."""

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise ValueError("vision_config is required for VQToken")
    image_size = _positive_int(
        getattr(vision_config, "image_size", None),
        "vision_config.image_size",
    )
    patch_size = _positive_int(
        getattr(vision_config, "patch_size", None),
        "vision_config.patch_size",
    )
    patch_grid_dim = (image_size // patch_size) ** 2
    query_dim = _positive_int(
        getattr(config, "vqtoken_query_dim", patch_grid_dim),
        "vqtoken_query_dim",
    )
    if query_dim != patch_grid_dim:
        raise ValueError(
            f"vqtoken_query_dim must match the vision patch grid "
            f"({patch_grid_dim}), got {query_dim}"
        )
    return query_dim


def validate_vqtoken_config(config: object) -> None:
    """Fail closed for semantics not represented by a static vLLM prompt."""

    mode = getattr(config, "vqtoken_mode", "attention")
    if mode != "attention":
        raise ValueError(
            "only the released VQToken attention mode is supported; "
            f"got vqtoken_mode={mode!r}"
        )

    selection_method = getattr(config, "vqtoken_selection_method", "fixed")
    if selection_method != "fixed":
        raise ValueError(
            "vLLM requires a static VQToken placeholder count; "
            "set vqtoken_selection_method='fixed'"
        )

    canonicalize = getattr(config, "vqtoken_canonicalize", False)
    if canonicalize is not False:
        raise ValueError(
            "the released attention checkpoint requires vqtoken_canonicalize=false"
        )

    seed = getattr(config, "vqtoken_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise ValueError("the released fixed-K path requires vqtoken_seed=0")

    vision_feature_layer = getattr(config, "vision_feature_layer", None)
    if vision_feature_layer != -2:
        raise ValueError("the released checkpoint requires vision_feature_layer=-2")

    strategy = getattr(config, "vision_feature_select_strategy", None)
    if strategy != "full":
        raise ValueError(
            "the released checkpoint requires vision_feature_select_strategy='full'"
        )

    vision_config = getattr(config, "vision_config", None)
    num_hidden_layers = getattr(vision_config, "num_hidden_layers", None)
    if num_hidden_layers != 27:
        raise ValueError(
            "the released checkpoint requires vision_config.num_hidden_layers=27; "
            "vLLM then selects layer -2 and constructs the 26 stored layers"
        )

    get_num_clusters(config)
    get_max_iterations(config)
    get_attention_heads(config)
    get_query_dim(config)


def get_num_video_tokens(config: object, num_frames: int) -> int:
    """Return the exact static placeholder count for one sampled video."""

    validate_vqtoken_config(config)
    num_frames = _positive_int(num_frames, "num_frames")
    num_clusters = get_num_clusters(config)
    if num_frames > num_clusters:
        raise ValueError(
            f"VQToken received {num_frames} frames, but the released "
            f"attention path requires num_frames <= {num_clusters}"
        )

    return num_clusters + 1


def get_profile_frame_count(config: object) -> int:
    """Profile the maximum released frame count to reserve worst-case memory."""

    validate_vqtoken_config(config)
    return get_num_clusters(config)


__all__ = [
    "DEFAULT_ATTENTION_HEADS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_NUM_CLUSTERS",
    "get_attention_heads",
    "get_max_iterations",
    "get_num_clusters",
    "get_num_video_tokens",
    "get_profile_frame_count",
    "get_query_dim",
    "validate_vqtoken_config",
]
