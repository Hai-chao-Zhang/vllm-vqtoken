---
license: cc-by-sa-4.0
library_name: vllm
pipeline_tag: video-text-to-text
base_model:
  - haichaozhang/VQ-Token-llava-ov-0.5b
tags:
  - llava-onevision
  - video-understanding
  - vllm
  - vqtoken
---

# VQToken LLaVA-OneVision 0.5B — vLLM serving layout

This is a tensor-name and metadata conversion of the released
[`haichaozhang/VQ-Token-llava-ov-0.5b`](https://huggingface.co/haichaozhang/VQ-Token-llava-ov-0.5b)
checkpoint for the vLLM `VQTokenLlavaOnevisionForConditionalGeneration`
out-of-tree plugin. Tensor payload bytes are unchanged.

## Provenance

- Source revision: `{{SOURCE_REVISION}}`
- Source `model.safetensors` SHA-256: `{{SOURCE_SHA256}}`
- Converted `model.safetensors` SHA-256: `{{OUTPUT_SHA256}}`
- Tensor-data-region SHA-256: `{{DATA_SHA256}}`
- Conversion manifest: `conversion-manifest.json`
- Tokenizer/processor reference revision:
  `llava-hf/llava-onevision-qwen2-0.5b-ov-hf@74dd0bf867a4cda7950c17663794267c60cf4b40`

## Serving scope

The initial vLLM implementation exposes only the released deterministic fixed
budget:

- 32 uniformly sampled video frames maximum;
- fixed spherical K-Means with K=32 and seed 0;
- trained one-layer VQ-Attention; and
- 32 compressed video tokens plus one newline token.

This artifact contains only the released fixed-budget path. Inputs with more
than 32 sampled frames must be resampled before serving.

## vLLM plugin

Install the out-of-tree plugin from
[`Hai-chao-Zhang/vllm-vqtoken`](https://github.com/Hai-chao-Zhang/vllm-vqtoken)
in the same environment as the pinned vLLM release. Enable it with
`VLLM_PLUGINS=vqtoken_model`, then pass this repository ID to the normal
`vllm serve` command. The plugin README contains the exact supported versions,
video sampling arguments, and OpenAI-compatible request format.

## Validation

Record exact commands, vLLM revision, hardware, prompt, sampling settings,
output hashes, and the weight-loader result here. Distinguish successful
vLLM/CUDA serving from any separate original-runtime parity evidence.

## License and citation

The converted weights retain the source checkpoint's CC-BY-SA-4.0 license.
The conversion tooling and vLLM integration code have their own repository
licenses.  Cite the VQToken paper and the upstream LLaVA-OneVision/Qwen/SigLIP
components as described by their original model cards.
