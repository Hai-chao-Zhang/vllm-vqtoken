# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Haichao Zhang

import pytest
import torch
import torch.nn.functional as F

from vllm_vqtoken.algorithms import VQTokenAttention, fixed_spherical_kmeans


@pytest.mark.parametrize("num_frames", [1, 4, 8])
def test_fixed_spherical_kmeans_is_deterministic(num_frames: int):
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(num_frames, 9, 12, generator=generator)

    labels_1, codebook_1 = fixed_spherical_kmeans(
        features,
        6,
        max_iterations=10,
    )
    labels_2, codebook_2 = fixed_spherical_kmeans(
        features,
        6,
        max_iterations=10,
    )

    assert labels_1.shape == (num_frames, 9)
    assert codebook_1.shape == (6, 12)
    assert torch.equal(labels_1, labels_2)
    assert torch.equal(codebook_1, codebook_2)
    assert labels_1.min() >= 0
    assert labels_1.max() < 6
    assert torch.isfinite(codebook_1).all()


def test_codebook_contains_final_normalized_cluster_means():
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(4, 9, 8, generator=generator)
    labels, codebook = fixed_spherical_kmeans(features, 4)
    normalized = F.normalize(features.reshape(-1, 8).float(), dim=-1)
    flat_labels = labels.flatten()

    for cluster_idx in range(4):
        members = normalized[flat_labels == cluster_idx]
        assert members.shape[0] > 0
        torch.testing.assert_close(codebook[cluster_idx], members.mean(dim=0))


def test_attention_emits_one_token_per_cluster():
    attention = VQTokenAttention(query_dim=9, context_dim=12, num_heads=3)
    attention.eval()
    assignments = torch.arange(4 * 9, dtype=torch.float32).view(4, 9) % 6
    codebook = torch.randn(6, 12)

    with torch.inference_mode():
        output = attention(assignments, codebook)

    assert output.shape == (6, 12)
    assert torch.isfinite(output).all()


def test_attention_rejects_more_frames_than_clusters():
    attention = VQTokenAttention(query_dim=9, context_dim=12, num_heads=3)
    assignments = torch.zeros(7, 9)
    codebook = torch.zeros(6, 12)

    with pytest.raises(ValueError, match="num_frames <= vqtoken_num_clusters"):
        attention(assignments, codebook)


@pytest.mark.parametrize("num_heads", [True, 0, "8"])
def test_attention_validates_num_heads_type(num_heads):
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        VQTokenAttention(query_dim=9, context_dim=12, num_heads=num_heads)
