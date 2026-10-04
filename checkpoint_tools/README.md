# VQToken vLLM checkpoint conversion tools

This repository contains a reviewable, streaming conversion for the
**publicly released, access-gated**
[`haichaozhang/VQ-Token-llava-ov-0.5b`](https://huggingface.co/haichaozhang/VQ-Token-llava-ov-0.5b)
checkpoint. It prepares the tensor names and metadata expected by the
out-of-tree vLLM `VQTokenLlavaOnevisionForConditionalGeneration` plugin.

It does **not** contain model weights or download credentials. The converter
accepts only the released fixed-budget tensor namespaces and rejects anything
outside that reviewed profile: 32 spherical K-Means tokens plus one newline
token.

## Pinned source

- Repository: `haichaozhang/VQ-Token-llava-ov-0.5b`
- Revision: `ab20666864dbe71c931a9c1236190e90493f02fe`
- `model.safetensors` size: `2081132000` bytes
- `model.safetensors` SHA-256:
  `593c6ffe2df774aaf2baa357e56bb1c33379584ae5a66d7c7cea5608a12bd5c7`
- Weight license: retain the source model card's `CC-BY-SA-4.0` license and
  attribution in any converted repository.

The CLI checks both the pinned size and hash by default.  It has no network
code, so the caller must obtain the file through an authenticated Hugging Face
workflow without passing a token on the command line.

Development fixtures may opt out with `--unverified-source`. Their manifests
set `source.verified` to `false` and leave the repository and revision null;
they never inherit the released checkpoint's provenance.

## Conversion

```bash
python -m vqtoken_checkpoint \
  /path/to/source/model.safetensors \
  /path/to/staging/model.safetensors \
  --manifest /path/to/staging/conversion-manifest.json
```

The converter:

1. parses and validates the safetensors header with duplicate-key, bounds,
   shape/dtype byte-size, gap, and overlap checks;
2. requires exactly the 20 released VQ-Attention tensors and validates their
   full shapes for query dimension 729, hidden size 896, and FFN size 2048;
3. validates the exact SigLIP encoder layer indices 0-25 and Qwen2 language
   layer indices 0-23, plus the two-layer projector, newline, embedding, LM
   head, and key SigLIP-384 tensor shapes;
4. constructs a complete one-to-one name map and rejects collisions or any
   unreviewed tensor namespace;
5. writes a new header, then copies the data region in 16 MiB chunks without
   loading a tensor into RAM; and
6. emits source/output/header/data SHA-256 hashes plus the complete tensor map
   in `conversion-manifest.json`.

Existing destination and manifest files are not overwritten unless the caller
explicitly supplies `--force`.  A temporary file is atomically renamed only
after all source checks pass.

## Exact tensor-name mapping

| Released source prefix/name | vLLM-ready target prefix/name |
| --- | --- |
| `model.vision_tower.vision_tower.` | `model.vision_tower.` |
| `model.mm_projector.0.` | `model.multi_modal_projector.linear_1.` |
| `model.mm_projector.2.` | `model.multi_modal_projector.linear_2.` |
| `model.cross_attention.` | `model.vq_attention.` |
| `model.image_newline` | unchanged |
| `model.embed_tokens.`, `model.layers.`, `model.norm.` | insert `language_model.` after `model.` |
| `lm_head.` | unchanged |

No catch-all mapping exists. For example, a tensor under an unsupported
`model.extra_module.*` namespace causes conversion to stop.

## Metadata plan

The files in [`metadata/`](metadata/) are the serving templates.  They correct
the stale nested Llama/CLIP metadata in the released repository while leaving
the original repository unchanged:

- architecture: `VQTokenLlavaOnevisionForConditionalGeneration`;
- standard outer `model_type`: `llava_onevision`, allowing the normal
  Transformers LLaVA-OneVision config/processor path;
- Qwen2 0.5B: 24 layers, hidden size 896, vocab 151936, untied embeddings;
- embedded SigLIP-384: a 27-layer base declaration with layer `-2` selected,
  matching the 26 stored layers (indices 0-25), with hidden size 1152 and
  14-pixel patches;
- feature selection: layer `-2`, strategy `full`, yielding 729 patches/frame;
- public VQ path: learned attention, fixed K=32, query dimension 729, 8 heads,
  `vqtoken_seed=0`, 50 K-Means iterations, tolerance `1e-4`, no
  canonicalization; and
- processor tokens: `<image>` 151646 and `<video>` 151647.

The released `config.json` records `use_cache=false`, a training-time setting.
The serving template deliberately sets `text_config.use_cache=true`; vLLM owns
its runtime KV-cache implementation, and a serving checkpoint should not carry
the training-time cache disablement forward.

For a new, separate public Hugging Face serving repository:

1. place the converted `model.safetensors`, its conversion manifest, and these
   templates in an empty staging directory;
2. copy only `tokenizer.json`, `tokenizer_config.json`,
   `special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`,
   `chat_template.json`, and `generation_config.json` from
   `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` pinned at revision
   `74dd0bf867a4cda7950c17663794267c60cf4b40`;
3. do not copy that reference repository's model, ONNX, config, or processor
   files; the files here replace the latter three categories;
4. add a model card recording both pinned revisions, source and converted
   hashes, CC-BY-SA-4.0 weight licensing, and the fixed-K-only vLLM scope;
5. validate `AutoConfig` and `AutoProcessor` locally, then load every tensor
   through the vLLM registry with no missing or unexpected weights; and
6. run deterministic video inference against the original released runtime
   before publishing or citing parity.

The `video_processor/preprocessor_config.json` compatibility copy is retained
alongside the newer root `video_preprocessor_config.json` layout.

These JSON templates derive from
`llava-hf/llava-onevision-qwen2-0.5b-ov-hf` at revision
`74dd0bf867a4cda7950c17663794267c60cf4b40`. `processor_config.json` is copied
unchanged; the other configuration/processor templates are modified for the
released VQToken architecture, the vLLM serving layout, or the current
Transformers metadata format. See the repository-level third-party notice and
Apache-2.0 license text.

When installed from the standalone converter wheel, locate the packaged
templates without assuming a platform-specific `share/` prefix:

```python
from vqtoken_checkpoint import get_metadata_dir

metadata_dir = get_metadata_dir()
```

## Tests

The tests create a tiny, dependency-free safetensors fixture with the same
module topology and reduced dimensions:

```bash
python -m unittest discover -v
```

They cover streaming byte preservation, all mapping rules, exact attention
membership and shapes, exact vision/language layer index sets, metadata-to-weight
layer consistency, collision rejection, unreviewed namespaces, pinned-hash
failure cleanup, and overwrite protection.

## Current validation boundary

The pinned 2.08 GB released checkpoint has passed the converter's real-header
validation, including all 737 tensor names, the exact vision/language layer
index sets, and the fixed public shape profile.  A local streaming conversion
also preserved the tensor data byte-for-byte and produced output SHA-256
`83b33589aab6b8cc16a232cf4fd1319c7ba8c90c14c7fead1e485941a0de3a91`.

No converted Hugging Face repository has been published yet. On the pinned
vLLM revision, a meta-device construction streamed all 737 tensors into all
511 internal parameters with no missing or unexpected weights. The converted
checkpoint also passed real CUDA materialization and inference: 10-second and
48-second videos were uniformly sampled to 32 frames, three greedy long-video
runs produced identical token IDs, and the native OpenAI-compatible vLLM API
returned HTTP 200 for the 48-second input.

These checks validate the fixed-`K` conversion, model load, deterministic GPU
generation, and serving path. Representative original-runtime versus vLLM
token/logprob parity and a serving throughput benchmark have not yet been run.
