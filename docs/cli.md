# Using ModalEnough

Install the package once, activate its environment, then run `modalenough` from any directory. `python -m selective_multimodal.video.cli` exposes the same commands.

## Commands

| Command | Purpose |
|---|---|
| `config init` | Create a TOML file without overwriting an existing one |
| `config show` | Print effective settings as JSON; no API keys |
| `download-models` | Download pinned model revisions to the configured directory |
| `doctor` | Check FFmpeg, dependencies, CUDA, and model resources without loading weights |
| `predict VIDEO` | Save one analysis and print its decision and report paths |
| `batch PATH...` | Process files or folders serially, reusing loaded models |

`--help` lists options for each command. `--version` prints the installed version.

```bash
modalenough predict review.mp4 --output result.json
modalenough batch reviews/ --recursive --output-dir results/january
modalenough batch --resume results/january --retry-failed
```

Supported containers are MP4/M4V, MOV, MKV, WebM, and AVI. Clips must contain video, last at most 180 seconds, and fit within 7680 by 4320 pixels. Review speech should be English. Remote URLs are rejected.

## Settings and paths

By default, settings and results live under `%LOCALAPPDATA%/ModalEnough` on Windows or `$XDG_DATA_HOME/modalenough` on Linux (`~/.local/share/modalenough` when unset). Set `MODALENOUGH_HOME` to choose another location. Models use its `models/` directory; an existing editable checkout's `runs/models/` cache is reused when present.

```bash
modalenough --config /path/to/settings.toml config init
modalenough --config /path/to/settings.toml config show
modalenough predict review.mp4 --model-root /path/to/models --output-root /path/to/results
```

CLI flags override environment variables, then TOML, then defaults. Relative paths in TOML resolve against the settings file. Paths supplied in flags or environment variables resolve against the current directory. `--output-dir` selects one run's directory; `--output-root` selects the parent for automatically named runs. A new run requires a new or empty directory.

The TOML keys are `model_root`, `output_root`, `backend`, `device`, `budget`, `confidence`, `strategy`, `modalities`, and `claude_model`. Their environment equivalents use the `MODALENOUGH_` prefix, such as `MODALENOUGH_MODEL_ROOT` or `MODALENOUGH_DEVICE`. Use a comma-separated list for `MODALENOUGH_MODALITIES`; TOML uses an array. `.env` files are not loaded automatically.

```toml
model_root = "D:/models/modalenough"
output_root = "D:/review-results"
device = "auto"
backend = "local"
budget = 1.0
confidence = 0.85
strategy = "adaptive"
modalities = ["text", "audio", "vision"]
claude_model = "claude-sonnet-4-6"
```

`--text-only` enables just transcript sentiment. `--device cpu` makes that path CPU-only. `--all` processes every enabled affordable input, even after a confident score. `--modalities text,vision` skips the separate native-audio pass. Budget units and score thresholds are explained in the [method guide](video.md).

## Saved work and recovery

Every run contains `manifest.json`, `events.jsonl`, `summary.csv`, `summary.md`, and a directory per video. Each successful analysis has `result.json`, `report.md`, and any extracted audio or frames. Decoder/setup failures get `error.json` and a failure report. Your source video is never copied or changed.

`batch --resume DIRECTORY` restores the saved input list and settings when no explicit configuration or inference flags are supplied. Complete results are reused only when both the source hash and saved result hash match. A damaged result is recomputed; a missing Markdown report is regenerated from intact JSON. Changed video content gets a new job directory. Failed/degraded results remain recorded unless you add `--retry-failed`.

Recovery happens between videos. Interrupted videos restart from the beginning; already incurred paid requests may be charged again. A running native operation may finish before Python handles Ctrl+C. One OS lock protects a run directory and is released when the process exits, including after a crash. Never edit manifests by hand or run two processes against the same output directory.

A failed extra modality leaves a visibly **degraded** result, even if earlier evidence produced a prediction. A legitimate abstention is a completed analysis. CSV summaries escape formula prefixes in text fields. Reports may contain private speech and frames; keep their directory as private as the original review.

## Automation and errors

Use `--json --quiet` for scripts. Diagnostics and progress stay on stderr; stdout contains one JSON document. `predict --output FILE` additionally exports its result outside the managed run directory. The run already contains its own `result.json`.

| Exit code | Meaning |
|---|---|
| 0 | Completed prediction, abstention, or successful utility command |
| 1 | Operational failure, or at least one failed/degraded batch job |
| 2 | Invalid arguments/settings or failed setup diagnostics |
| 130 | Interrupted; completed jobs remain saved |

Start with `doctor` when setup fails. A CPU PyTorch build cannot use an NVIDIA card; install the CUDA wheel shown in the [video guide](video.md), or select `--text-only --device cpu`. Missing shards or tokenizer files require `download-models` again. Resume refuses changed inference configuration; start a new run to compare different settings.
