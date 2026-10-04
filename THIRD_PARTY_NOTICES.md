# Third-party notices

Parts of `vllm_vqtoken/algorithms.py` and `tests/test_algorithms.py` are adapted
from the VQToken reference implementation:

- Source: <https://github.com/Hai-chao-Zhang/VQToken>
- Source revision: `4bcd5a01fa39480a2f18d303f55b555bd1b5de58`
- License: BSD-3-Clause
- Copyright (c) 2025, Haichao Zhang

The BSD-3-Clause license text is included in
`LICENSES/BSD-3-Clause.txt`.

The checkpoint metadata templates under `checkpoint_tools/metadata/` derive
from the LLaVA-OneVision configuration and processor files in:

- Source: <https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf>
- Source revision: `74dd0bf867a4cda7950c17663794267c60cf4b40`
- License: Apache-2.0

`checkpoint_tools/metadata/processor_config.json` is copied unchanged.
`config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`,
and `video_processor/preprocessor_config.json` are modified for the released
VQToken architecture, the vLLM serving layout, or the current Transformers
metadata format. The Apache-2.0 license text is included in
`LICENSES/Apache-2.0.txt`.
