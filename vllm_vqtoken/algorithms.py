# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Haichao Zhang

"""Released fixed-K clustering and learned VQ-Attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_initial_centroids(
    tokens: torch.Tensor,
    num_clusters: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Initialize spherical K-Means centers with K-Means++."""

    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    first_idx = torch.randint(
        tokens.shape[0],
        (1,),
        device=tokens.device,
        generator=generator,
    ).item()
    centroids = [tokens[first_idx]]
    closest_squared_distance = (tokens - centroids[0]).square().sum(dim=1)

    for _ in range(1, num_clusters):
        total = closest_squared_distance.sum()
        if not torch.isfinite(total) or total <= 0:
            next_idx = torch.randint(
                tokens.shape[0],
                (1,),
                device=tokens.device,
                generator=generator,
            ).item()
        else:
            next_idx = torch.multinomial(
                closest_squared_distance / total,
                1,
                generator=generator,
            ).item()
        centroids.append(tokens[next_idx])
        new_squared_distance = (tokens - centroids[-1]).square().sum(dim=1)
        closest_squared_distance = torch.minimum(
            closest_squared_distance,
            new_squared_distance,
        )

    return torch.stack(centroids)


def _compute_cosine_distances(
    tokens: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    tokens = F.normalize(tokens, dim=-1, eps=1e-12)
    centroids = F.normalize(centroids, dim=-1, eps=1e-12)
    return 1 - tokens @ centroids.T


def _cluster_means(
    normalized_tokens: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    num_clusters, hidden_size = centroids.shape
    sums = torch.zeros(
        num_clusters,
        hidden_size,
        dtype=torch.float32,
        device=normalized_tokens.device,
    )
    sums.index_add_(0, labels, normalized_tokens)
    counts = torch.bincount(labels, minlength=num_clusters)
    means = sums / counts.clamp_min(1).to(sums.dtype)[:, None]
    means = torch.where(counts[:, None] > 0, means, centroids)
    return means.to(dtype=output_dtype)


def fixed_spherical_kmeans(
    video_features: torch.Tensor,
    num_clusters: int,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster a projected video with released fixed-K VQToken semantics.

    Args:
        video_features: Features shaped ``[frames, patches, hidden_size]``.
        num_clusters: Static codebook size.
        max_iterations: Maximum spherical K-Means updates.
        tolerance: Early-stop threshold for the centroid shift.
        seed: K-Means++ generator seed. Released checkpoints use zero.

    Returns:
        Assignment maps shaped ``[frames, patches]`` and a normalized codebook
        shaped ``[num_clusters, hidden_size]``.
    """

    if not isinstance(video_features, torch.Tensor) or video_features.ndim != 3:
        raise ValueError(
            "video_features must be a tensor with shape [frames, patches, hidden_size]"
        )
    if not video_features.is_floating_point():
        raise TypeError("video_features must use a floating-point dtype")
    if any(size == 0 for size in video_features.shape):
        raise ValueError("video_features must be non-empty in every dimension")
    if not torch.isfinite(video_features).all():
        raise ValueError("video_features must contain only finite values")
    if (
        isinstance(num_clusters, bool)
        or not isinstance(num_clusters, int)
        or num_clusters < 1
    ):
        raise ValueError("num_clusters must be a positive integer")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    num_frames, tokens_per_frame, hidden_size = video_features.shape
    num_tokens = num_frames * tokens_per_frame
    if num_clusters > num_tokens:
        raise ValueError(
            f"num_clusters ({num_clusters}) cannot exceed the number of "
            f"video tokens ({num_tokens})"
        )

    output_dtype = video_features.dtype
    normalized_tokens = F.normalize(
        video_features.reshape(num_tokens, hidden_size).to(dtype=torch.float32),
        dim=-1,
        eps=1e-12,
    )
    clustering_tokens = normalized_tokens.detach()
    centroids = _sample_initial_centroids(
        clustering_tokens,
        num_clusters,
        seed=seed,
    )

    for _ in range(max_iterations):
        labels = _compute_cosine_distances(
            clustering_tokens,
            centroids,
        ).argmin(dim=1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, labels, clustering_tokens)
        counts = torch.bincount(labels, minlength=num_clusters)
        means = sums / counts.clamp_min(1).to(sums.dtype)[:, None]
        new_centroids = torch.where(counts[:, None] > 0, means, centroids)
        shift = torch.linalg.vector_norm(centroids - new_centroids)
        centroids = new_centroids
        if shift <= tolerance:
            break

    labels = _compute_cosine_distances(
        clustering_tokens,
        centroids,
    ).argmin(dim=1)
    codebook = _cluster_means(
        normalized_tokens,
        labels,
        centroids,
        output_dtype,
    )

    return labels.reshape(num_frames, tokens_per_frame), codebook


class VQTokenAttention(nn.Module):
    """Learned assignment-map-to-codebook attention from VQToken."""

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "query_dim": query_dim,
            "context_dim": context_dim,
            "num_heads": num_heads,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if context_dim % num_heads != 0:
            raise ValueError("VQToken context dimension must be divisible by num_heads")

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.to_q_proj = nn.Linear(query_dim, context_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=1,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        assignment_maps: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        """Emit one compressed visual token per codebook entry."""

        if assignment_maps.ndim != 2 or codebook.ndim != 2:
            raise ValueError("assignment_maps and codebook must both be 2D")
        if assignment_maps.shape[1] != self.query_dim:
            raise ValueError(
                f"expected {self.query_dim} assignment values per frame, got "
                f"{assignment_maps.shape[1]}"
            )
        if codebook.shape[1] != self.context_dim:
            raise ValueError(
                f"expected codebook width {self.context_dim}, got {codebook.shape[1]}"
            )

        num_frames = assignment_maps.shape[0]
        num_clusters = codebook.shape[0]
        if num_frames == 0 or num_clusters == 0:
            raise ValueError("assignment_maps and codebook must be non-empty")
        if num_frames > num_clusters:
            raise ValueError(
                "VQToken requires num_frames <= vqtoken_num_clusters; "
                "reduce the sampled frame count or increase the static codebook"
            )
        if num_frames < num_clusters:
            repeats = (num_clusters + num_frames - 1) // num_frames
            assignment_maps = assignment_maps.repeat(repeats, 1)[:num_clusters]

        queries = self.to_q_proj(assignment_maps).unsqueeze(0)
        memory = codebook.unsqueeze(0)
        return self.transformer_decoder(tgt=queries, memory=memory).squeeze(0)


__all__ = ["VQTokenAttention", "fixed_spherical_kmeans"]
