# CLI release verification

ModalEnough 0.3.0 was exercised from a different working directory through its installed command. A GPU batch processed positive, negative, and silent video fixtures, saved per-video reports and evidence, and produced CSV/Markdown summaries. Positive and negative fixtures used text, native audio, and frames; the silent clip abstained. Resume reused all three completed results without another attempt.

These fixtures contain locally synthesized English speech over plain video. They exercise decoding and actual pretrained inference; they do not establish accuracy on natural reviews. First-job timing includes loading the model cache, so it is much slower than subsequent jobs.

`verification.txt` records 102 passing tests and 93% statement coverage across the six CLI workflow modules. Tests exercise real FFmpeg decoding, interruption/recovery, failed inputs, source changes, corrupted results, output-directory protection, cross-process locking, report escaping, settings precedence, and command exit codes. Model/API boundaries are replaced in automated workflow tests; the GPU smoke checks use the actual local models.

Black, Ruff, and strict mypy checks passed for the CLI modules. Mypy also checked the Linux platform branch. The wheel was built, installed into another environment, and its command/configuration tested outside the checkout; every video module in the wheel matches the source bytes and no web UI is included. The separate environment reuses locally installed dependencies for its inference check; this is not a clean-machine dependency-install benchmark.

`evidence.json` records fixture hashes, measured decisions/timings, model revisions, coverage, and wheel identity. `source-hashes.json` identifies the implementation. CI now targets Windows and Linux; hosted CI results and live Claude connectivity are not claimed verified here. No paid request was made.
