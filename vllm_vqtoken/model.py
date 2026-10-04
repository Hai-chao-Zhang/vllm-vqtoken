# SPDX-License-Identifier: Apache-2.0

"""vLLM model adapter for released fixed-K VQToken."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.llava_onevision import (
    LlavaOnevisionDummyInputsBuilder,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionMultiModalProcessor,
    LlavaOnevisionProcessingInfo,
    LlavaOnevisionVideoPixelInputs,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from .algorithms import VQTokenAttention, fixed_spherical_kmeans
from .semantics import (
    get_attention_heads,
    get_max_iterations,
    get_num_clusters,
    get_num_video_tokens,
    get_profile_frame_count,
    get_query_dim,
    validate_vqtoken_config,
)


class VQTokenProcessingInfo(LlavaOnevisionProcessingInfo):
    """LLaVA-OneVision processing with a static VQToken video budget."""

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
    ) -> int:
        del image_width, image_height
        return get_num_video_tokens(self.get_hf_config(), num_frames)

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        del seq_len, mm_counts
        return get_profile_frame_count(self.get_hf_config())


@MULTIMODAL_REGISTRY.register_processor(
    LlavaOnevisionMultiModalProcessor,
    info=VQTokenProcessingInfo,
    dummy_inputs=LlavaOnevisionDummyInputsBuilder,
)
class VQTokenLlavaOnevisionForConditionalGeneration(
    LlavaOnevisionForConditionalGeneration
):
    """LLaVA-OneVision with released fixed-K VQToken video compression."""

    requires_sequential_video_encoding = True

    hf_to_vllm_mapper = (
        LlavaOnevisionForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(
            orig_to_new_prefix={
                "model.vq_attention.": "vq_attention.",
                "model.cross_attention.": "vq_attention.",
            }
        )
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        validate_vqtoken_config(config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        query_dim = get_query_dim(config)

        base_tower_names = list(self._tower_model_names)
        with self._mark_tower_model(vllm_config, {"video"}):
            self.vq_attention = VQTokenAttention(
                query_dim=query_dim,
                context_dim=config.text_config.hidden_size,
                num_heads=get_attention_heads(config),
            )
        vq_tower_names = list(self._tower_model_names)
        self._tower_model_names = list(
            dict.fromkeys([*base_tower_names, *vq_tower_names])
        )

        self.vqtoken_query_dim = query_dim
        self.vqtoken_num_clusters = get_num_clusters(config)
        self.vqtoken_max_iterations = get_max_iterations(config)

    def _compress_video_features(self, video_features: torch.Tensor) -> torch.Tensor:
        if video_features.shape[1] != self.vqtoken_query_dim:
            raise ValueError(
                "projected video features have "
                f"{video_features.shape[1]} patches per frame, but "
                f"vqtoken_query_dim={self.vqtoken_query_dim}"
            )
        assignment_maps, codebook = fixed_spherical_kmeans(
            video_features,
            self.vqtoken_num_clusters,
            max_iterations=self.vqtoken_max_iterations,
            seed=0,
        )
        return self.vq_attention(
            assignment_maps.to(dtype=video_features.dtype),
            codebook,
        )

    def _vqtoken_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        features = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )
        return self.multi_modal_projector(features)

    def _process_video_pixels(
        self,
        inputs: LlavaOnevisionVideoPixelInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        video_pixels = inputs["pixel_values_videos"]

        if isinstance(video_pixels, torch.Tensor):
            total_videos, num_frames, channels, height, width = video_pixels.shape
            get_num_video_tokens(self.config, num_frames)
            pixels_flat = video_pixels.reshape(
                total_videos * num_frames,
                channels,
                height,
                width,
            )
            features_flat = self._vqtoken_pixels_to_features(pixels_flat)
            features = features_flat.reshape(
                total_videos,
                num_frames,
                features_flat.shape[1],
                features_flat.shape[2],
            )
            compressed = torch.stack(
                [self._compress_video_features(video) for video in features]
            )
            image_newline = self.image_newline[None, None, :].expand(
                total_videos,
                -1,
                -1,
            )
            return torch.cat((compressed, image_newline), dim=1)

        frames_per_video = [len(video) for video in video_pixels]
        for num_frames in frames_per_video:
            get_num_video_tokens(self.config, num_frames)

        pixels_flat = torch.cat(video_pixels, dim=0)
        features_flat = self._vqtoken_pixels_to_features(pixels_flat)
        videos = torch.split(features_flat, frames_per_video)
        image_newline = self.image_newline[None, None, :]

        return [
            torch.cat(
                (
                    self._compress_video_features(video).unsqueeze(0),
                    image_newline,
                ),
                dim=1,
            )
            for video in videos
        ]


__all__ = [
    "VQTokenLlavaOnevisionForConditionalGeneration",
    "VQTokenProcessingInfo",
]
