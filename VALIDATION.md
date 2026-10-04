# Validation record

This record separates release compatibility, real GPU serving, and claims that
have not been established. It covers only the released fixed-`K` VQToken path.

## vLLM 0.30.0 release compatibility

Date: 2026-10-04 (America/New_York)

- Runtime: official `vllm/vllm-openai-cpu:v0.30.0` image, materialized as an
  Apptainer SIF with SHA-256
  `b069913e6cb83409b553ab5cf9bf0f29db785b34150d188dfcec997ba7e6f1fe`.
- vLLM: `0.30.0`.
- Transformers: `5.16.1`, the version pinned by vLLM 0.30.0's test locks.
- Plugin artifact: the `vllm-vqtoken` `0.1.0` wheel, installed without a
  source-tree copy on `PYTHONPATH`.
- Converted model SHA-256:
  `83b33589aab6b8cc16a232cf4fd1319c7ba8c90c14c7fead1e485941a0de3a91`.
- Converted config SHA-256:
  `7c55988e1425dcac0eb711509a359257d516b914d8c0305ecd615ff1a6d66e3a`.
- Result: the `vllm.general_plugins` entry point was discovered, the VQToken
  architecture was registered, the Pixtral/LLaVA-OneVision dependency path
  imported, and the suite reported `28 passed`.
- The opt-in full-checkpoint test constructed the model on the meta device,
  read all 737 source tensors, matched all 511 internal parameters, loaded all
  20 VQ-Attention parameters, and confirmed the expected 26 stored vision
  layers.

The published v0.30.0 CPU image currently contains Transformers 5.17.0, which
is incompatible with that release's Pixtral import. The test therefore overlaid
only the upstream test-locked Transformers 5.16.1 wheel ahead of the immutable
image. The plugin's `vllm` optional dependency pins the same combination.

## CUDA and OpenAI-compatible API

Date: 2026-09-25 (America/New_York)

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB).
- Driver: `595.71.05`.
- PyTorch: `2.13.0+cu130`.
- vLLM source and precompiled extension:
  `da1aaec31fb366c0ded497c1fa4b79a2fcc99969`.
- Plugin development wheel SHA-256:
  `6d5fcb65428ff8cca3b336eedb2d5853a2abe5a71195af8712ccac3522fffe8e`.
- Its four runtime modules are byte-identical to the `0.1.0` release sources;
  only packaging, tests, and documentation changed afterward.
- Full suite: `27 passed` and the 1.96 GiB checkpoint loaded on CUDA.
- Offline inference: 10-second and 48-second videos were each sampled to 32
  frames including both timeline endpoints.
- Determinism: three greedy 48-second runs produced the same token-ID SHA-256,
  `40fbcaf26b7ad0892f869c8d7bcebbbdaec79ed8067b816ebda665be216debfa`.
- Native server: `POST /v1/chat/completions` with the 48-second MP4 returned
  HTTP 200; the server and response contained no 30-second rejection.

This GPU run predates the `0.1.0` packaging release, so it validates the same
fixed-`K` model implementation on the pinned vLLM development revision, not a
claim that the final wheel was GPU-tested against every supported vLLM build.

## Evidence boundary

The checks establish OOT registration, full checkpoint topology and weight
routing, real CUDA model materialization and deterministic inference, and a
native OpenAI-compatible video request. They do not establish token/logprob
parity with the original training repository, a throughput improvement, or
support for any path beyond the released fixed-`K` implementation.
