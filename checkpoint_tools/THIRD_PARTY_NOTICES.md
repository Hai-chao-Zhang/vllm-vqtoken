# Third-party notices

The metadata templates under `metadata/` derive from the LLaVA-OneVision
configuration and processor files in:

- Source: <https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf>
- Source revision: `74dd0bf867a4cda7950c17663794267c60cf4b40`
- License: Apache-2.0

`metadata/processor_config.json` is copied unchanged. `metadata/config.json`,
`metadata/preprocessor_config.json`, `metadata/video_preprocessor_config.json`,
and `metadata/video_processor/preprocessor_config.json` are modified for the
released VQToken architecture, the vLLM serving layout, or the current
Transformers metadata format. The Apache-2.0 license text is included in
`APACHE-2.0.txt`.
