# Real-media smoke evidence

This directory records the earlier 0.2.0 engine checks. Current CLI-release verification is in [reports/cli](../cli/README.md).

These checks used an RTX 5060 Ti, CUDA PyTorch, and the pinned models recorded in `smoke-results.json`. The MP4 fixtures contain locally synthesized spoken reviews over a plain background, plus one silent clip. They exercise actual decoding and pretrained inference, not natural-video benchmark accuracy.

The positive and negative spoken reviews were classified correctly from text alone. Full native passes acquire text, audio, and four frames. The silent clip returns no sentiment prediction and records `no_linguistic_evidence`.

`smoke-results.json` retains measured outputs, fixture hashes, and calculated checks. Cold timings include model loading and initial quantization/cache construction. `verification.txt` records the current CLI/core test suite; `source-hashes.json` identifies the corresponding implementation.

Claude is optional. No key was supplied, so no paid call was made or claimed verified. The real-media controller uses a confidence-based baseline; the learned policy remains in the separate synthetic experiment.
