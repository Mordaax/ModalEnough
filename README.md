# ModalEnough

ModalEnough analyses video reviews from the command line. Give it a clip or a folder; it produces a sentiment prediction or abstention, saves the evidence, and shows which inputs contributed to the answer.

Whisper transcribes speech and RoBERTa scores the words. When more evidence is needed, a quantized Qwen2.5-Omni model analyses audio windows and sampled frames on an NVIDIA GPU. A budget controls how far it can go. Processing is local by default; Claude is an optional, explicitly selected backend.

## Install

Use Python 3.11+ and put FFmpeg/FFprobe on PATH. Install CUDA-enabled PyTorch first if you want native audio and frame analysis; the [video guide](docs/video.md) includes the tested Windows command.

```bash
python -m venv .venv
# Activate .venv, then run from this checkout:
python -m pip install -e ".[video]"
modalenough config init
modalenough download-models
modalenough doctor
```

For CPU transcript processing, use `download-models --text-only`, then `doctor --text-only --device cpu`.

## Analyse reviews

```bash
modalenough predict review.mp4
modalenough batch ./reviews --recursive --output-dir ./results/run-01
modalenough batch --resume ./results/run-01
```

A batch keeps models loaded between videos, continues past failed clips, and saves its state after each job. Ctrl+C preserves completed work; resume retries the interrupted video. Failed or degraded jobs need `--retry-failed`. Input hashes prevent stale results from being reused after a file changes.

Each video gets a readable Markdown report, strict JSON, and its acquired evidence. The batch directory also contains CSV and Markdown summaries, a manifest, and an incremental event log. Use `--json` for machine-readable stdout; progress goes to stderr.

```bash
modalenough predict review.mp4 --text-only --device cpu --output result.json
modalenough predict review.mp4 --budget 0.55 --json
modalenough batch ./reviews --all --output-dir ./results/full-pass
```

Settings live in a user-level TOML file and work from any directory. `config show` displays effective values; command flags override environment settings and the file. See the [CLI reference](docs/cli.md) for paths, resume rules, exit codes, and troubleshooting.

## What the result means

The current video controller uses a confidence rule: text first, then audio and frames if enabled and affordable. Text costs 0.20 effort units, additional audio understanding 0.35, and frames 0.45. These units are neither dollars nor measured compute savings; transcription itself requires audio processing.

Use English videos up to three minutes long. Scores are uncalibrated, the text model was trained on tweets, and audio/frame sampling can miss details. A silent clip cannot establish a review opinion. Real-review benchmark accuracy has not been established. The learned acquisition policy and its results remain in the separate [synthetic experiment](docs/synthetic.md).

Claude requires `ANTHROPIC_API_KEY` and `--backend claude`. It receives acquired transcripts, frames, and any existing audio-score summary; speech recognition and native audio processing stay local. No API key is stored in configuration.

For contributors: `python -m pip install -e ".[dev,video]"`, then `python -m pytest -q`. Read the [video method](docs/video.md), [verification evidence](reports/video/README.md), and [prior-work review](docs/prior_work.md). Source is MIT; downloaded models keep their own licences, including Qwen's research licence.
