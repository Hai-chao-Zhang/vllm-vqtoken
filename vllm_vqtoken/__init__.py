# SPDX-License-Identifier: Apache-2.0

"""vLLM registration entry point for VQToken."""

from __future__ import annotations

MODEL_ARCHITECTURE = "VQTokenLlavaOnevisionForConditionalGeneration"
MODEL_ENTRY_POINT = "vllm_vqtoken.model:VQTokenLlavaOnevisionForConditionalGeneration"


def register() -> None:
    """Register the VQToken model lazily with vLLM."""

    from vllm import ModelRegistry

    if MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(MODEL_ARCHITECTURE, MODEL_ENTRY_POINT)


__all__ = ["MODEL_ARCHITECTURE", "MODEL_ENTRY_POINT", "register"]
