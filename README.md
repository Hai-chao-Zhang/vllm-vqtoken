# vLLM VQToken plugin

`vllm-vqtoken` registers the released fixed-`K` VQToken video path as an
out-of-tree model in vLLM. It uses vLLM's documented
`vllm.general_plugins` entry point and does not patch the vLLM source tree.

## Compatibility

Only the versions listed here are supported. The registration API is public,
but this adapter subclasses vLLM's internal LLaVA-OneVision implementation, so
new vLLM releases must be tested before the range is widened.

| vLLM revision | Validation |
| --- | --- |
| `0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`) | Official CPU image, `0.1.0` wheel discovery/registration, and 28 tests including full 737-tensor checkpoint load |
| `da1aaec31fb366c0ded497c1fa4b79a2fcc99969` | 27 tests, full CUDA checkpoint load, deterministic 10 s/48 s video inference, and OpenAI-compatible HTTP 200 |

The CUDA/API result used an NVIDIA RTX PRO 6000 Blackwell GPU, Python 3.12,
PyTorch `2.13.0+cu130`, and 32 uniformly sampled frames. It proves the native
vLLM serving path for the converted public fixed-`K` checkpoint; it is not an
original-runtime logprob-parity result or a throughput benchmark.
See [`VALIDATION.md`](VALIDATION.md) for the exact evidence boundary and the
stable-image dependency note.

## Install

On a supported Linux/CUDA host, install the first supported stable vLLM and
this plugin in the same environment:

```bash
git clone https://github.com/Hai-chao-Zhang/vllm-vqtoken.git
cd vllm-vqtoken
git checkout v0.1.0
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[vllm]' --torch-backend=auto
```

The base package deliberately has no mandatory `vllm` dependency, so
installing it cannot silently replace an existing CUDA-specific vLLM build.
The `vllm` extra pins the tested stable release and its upstream test-suite
Transformers version (`5.16.1`). This avoids an incompatible newer Transformers
release being selected by vLLM's intentionally broad runtime requirement.
The checkpoint's informational `transformers_version` field records the
conversion environment; it is not a serving dependency constraint.

For the validated development revision, make the precompiled-wheel commit
explicit; otherwise vLLM may install a different nightly extension:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout da1aaec31fb366c0ded497c1fa4b79a2fcc99969
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 \
VLLM_MAIN_CUDA_VERSION=13.0 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
VLLM_PRECOMPILED_WHEEL_COMMIT=da1aaec31fb366c0ded497c1fa4b79a2fcc99969 \
  uv pip install --python .venv/bin/python -e . --torch-backend=cu130
uv pip install --python .venv/bin/python \
  -e /absolute/path/to/vllm-vqtoken
```

## Checkpoint contract

The checkpoint must use the architecture
`VQTokenLlavaOnevisionForConditionalGeneration`, standard Hugging Face
LLaVA-OneVision processor metadata, and the released VQ-Attention weights.
The converted config contains:

```json
{
  "architectures": ["VQTokenLlavaOnevisionForConditionalGeneration"],
  "model_type": "llava_onevision",
  "vqtoken_mode": "attention",
  "vqtoken_selection_method": "fixed",
  "vqtoken_num_clusters": 32,
  "vqtoken_query_dim": 729,
  "vqtoken_attention_heads": 8,
  "vqtoken_max_iterations": 50,
  "vqtoken_canonicalize": false,
  "vqtoken_seed": 0,
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "full",
  "vision_config": {"num_hidden_layers": 27}
}
```

The 27-layer declaration is intentional: layer `-2` is selected, while the
released checkpoint stores the resulting 26 vision layers (indices 0-25).

The converted Hub repository is not publicly downloadable yet, so this README
uses a local checkpoint path rather than advertising a broken model ID. The
validated local artifact has model SHA-256
`83b33589aab6b8cc16a232cf4fd1319c7ba8c90c14c7fead1e485941a0de3a91`.
Users who have access to the released source checkpoint can reproduce that
artifact with the audited streaming converter in
[`checkpoint_tools/`](checkpoint_tools/).

## Serve with vLLM

Enable only the named plugin and start the normal vLLM OpenAI server:

```bash
export MODEL=/absolute/path/to/converted-vqtoken-checkpoint
export VLLM_PLUGINS=vqtoken_model
.venv/bin/vllm serve "$MODEL" \
  --served-model-name vqtoken \
  --runner generate \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"video":1,"image":0}' \
  --media-io-kwargs '{"video":{"num_frames":32}}'
```

Then call the ordinary OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="vqtoken",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.org/video.mp4"},
                },
            ],
        }
    ],
    max_completion_tokens=128,
)
print(response.choices[0].message.content)
```

There is no 30-second duration limit in this plugin. The media loader samples
32 frames over the full timeline. Each video produces 32 compressed tokens
plus one learned newline token, so keep `num_frames <= 32`.

Image inputs retain the standard LLaVA-OneVision path and are not compressed.
The conservative command above disables images only to match the validated
video-only API smoke.

## Released inference path

For each video the adapter:

1. runs every sampled frame through the LLaVA-OneVision vision tower;
2. projects all 729 SigLIP patch features per frame to the language width;
3. runs deterministic FP32 spherical K-Means with K-Means++ seed `0`;
4. passes raw assignment IDs and the codebook through the released one-layer
   VQ-Attention module; and
5. appends one learned newline embedding.

The plugin intentionally implements only this fixed-budget public path. vLLM
must know the exact multimodal placeholder count before encoder execution.

## Tests

Lightweight algorithm, semantics, and registration tests do not require vLLM:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

The vLLM integration module is skipped in that environment. For a release
gate, install the tested vLLM extra and make a missing vLLM installation fatal:

```bash
uv pip install --python .venv/bin/python -e '.[vllm,test]' --torch-backend=auto
VQTOKEN_REQUIRE_VLLM=1 \
VQTOKEN_CHECKPOINT_DIR=/absolute/path/to/converted-vqtoken-checkpoint \
  .venv/bin/python -m pytest -q
```

The opt-in checkpoint test constructs the full model on the meta device,
streams all tensors through vLLM's loader, and checks the vision depth and all
VQ-Attention parameters. This validates topology and weight routing, not CUDA
numerical execution; a release also requires the video and HTTP GPU smokes.

## Licensing

The plugin glue is Apache-2.0. `vllm_vqtoken/algorithms.py` and its algorithm
tests adapt the BSD-3-Clause VQToken reference implementation at revision
`4bcd5a01fa39480a2f18d303f55b555bd1b5de58`. Both license texts and the
third-party notice are included in source and wheel distributions. Converted
weights retain the source checkpoint's CC-BY-SA-4.0 license.
