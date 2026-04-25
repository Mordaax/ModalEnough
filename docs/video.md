# Real video reviews

The video CLI accepts a local review clip, predicts positive, negative, or neutral sentiment from the evidence it acquires, and returns an explicit abstention when the evidence doesn't clear the selected threshold. Its input-use trail is part of the result, not a post-hoc explanation.

## Start here

Install from the repository root; the installed `modalenough` command then works from any directory. FFmpeg and FFprobe must be on PATH. The tested machine has Python 3.13, 32 GB RAM, an RTX 5060 Ti with 16 GB VRAM, and CUDA-enabled PyTorch 2.9.0.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev,video]"
.\.venv\Scripts\python.exe -m selective_multimodal.video.cli download-models
.\.venv\Scripts\python.exe -m selective_multimodal.video.cli predict review.mp4 --output runs/review.json
```

The pipeline accepts an English video up to three minutes long and writes its result as JSON. Use `--budget` to limit acquisition or `--all` to process every affordable input.

The pinned model downloads include roughly 12 GB for the original multimodal checkpoint, plus speech and text models. The first native audio/frame request creates a smaller NF4 cache with speech generation disabled. It can take several minutes on a slow drive. Both caches live in the configured model directory; an existing checkout's `runs/models/` is reused. No weights are committed. Later requests reuse the quantized cache, and a batch retains loaded models between predictions.

For CPU-only transcript inference, install the video extra, download with `--text-only`, and run `predict review.mp4 --device cpu --budget 0.2`. Native audio/image inference in this implementation requires CUDA. A CPU-only PyTorch wheel won't detect an NVIDIA card even when one is installed; check `torch.version.cuda` as well as `nvidia-smi`.

## What each acquisition does

Text extraction decodes the video's audio to mono 16 kHz and runs `Systran/faster-whisper-base.en`. Voice activity detection filters empty sections. The complete usable transcript is split into token chunks and classified by `cardiffnlp/twitter-roberta-base-sentiment-latest`; chunk scores are weighted by token count. The model is trained on tweets, so transfer to spoken product reviews is a limitation.

Audio understanding is a separate request. Qwen2.5-Omni-3B receives up to three non-silent 10-second windows from the beginning, middle, and end, together with any transcript already acquired. Window scores are averaged. This is direct audio input, rather than another transcription-only call.

Vision samples four evenly spaced frames, resized to 512 pixels wide. The model receives those images and the acquired transcript. If audio was acquired successfully, the same acquired windows are included in joint scoring; the new score therefore retains that evidence. This final pass repeats some audio encoding, and its elapsed time includes that work. The JSON's `used_inputs` and per-stage `scoring_inputs` record what contributed to the final scorer. `acquired` also includes failed attempts, which are charged.

Qwen scores come from next-token logits for the three label words, normalized over those labels. They are relative model scores. Neither those scores nor RoBERTa's outputs have been calibrated on a real review validation set, and a threshold of 0.85 does not mean 85% empirical accuracy. Images alone do not establish the reviewer's opinion; clips without usable linguistic evidence abstain.

## Acquisition and budgets

The real-media controller uses a confidence-based baseline. It starts with text, then considers audio and vision in that order, skipping missing, disabled, or unaffordable inputs. Each completed stage replaces the current score with its assessment of the evidence supplied to it. With native joint scoring, later stages receive the previously acquired context. `all` processes every enabled input within budget; `adaptive` stops as soon as a usable opinion clears the threshold. A failed stage is logged and charged, never retried for free.

Effort costs are 0.20 for text, 0.35 for additional audio understanding, and 0.45 for vision. They are configured units, not dollars, latency predictions, or measured encoder savings. Transcript extraction has an audio dependency, so the inputs are not physically independent. Actual processing times include decoding, model loading, and inference; initial cache construction makes cold and warm timings quite different.

The original learned acquisition policy is preserved in the [synthetic benchmark](synthetic.md). It cannot consume these encoders' outputs without matching real-data training. Do not load its checkpoint into the video path or describe the live controller as a policy trained on real reviews.

## Optional Claude backend

Set `ANTHROPIC_API_KEY`, optionally set `MODALENOUGH_CLAUDE_MODEL`, then pass `--backend claude`. `.env.example` is only a reference; nothing loads it automatically. Never put a real key in source control or paste it into a clip.

Claude receives the acquired transcript and, after a vision request, sampled JPEG frames. If a local audio assessment already exists, its score summary is supplied too. Whisper and native audio understanding run locally. Only Claude mode sends those acquired inputs to Anthropic and may incur API charges. The adapter records token usage, validates the returned JSON, and never claims that Claude directly heard the waveform. See [Anthropic's image-input documentation](https://platform.claude.com/docs/en/build-with-claude/vision).

## Files and privacy

Video inputs are local paths. FFprobe checks their duration and dimensions before inference; decoder subprocesses have bounded timeouts and restricted local formats/protocols. Remote URLs are rejected.

The CLI saves each result and acquired evidence in its run directory under the configured output root. Batch CSV/Markdown summaries and a manifest support file-level resume; see the [CLI reference](cli.md). `--output` also writes the result to your chosen JSON path. The original video is read without being copied or changed. Local mode sends no review content to a model API; selecting Claude sends only the acquired evidence described above. Keep generated transcripts and frames private when the source video is private.

## Verify it

```bash
python -m pytest -q
python -m selective_multimodal.video.cli predict review.mp4 --budget 0.55 --output runs/review.json
python -m selective_multimodal.video.cli predict review.mp4 --all
```

The development smoke fixtures are locally synthesized spoken positive and negative reviews over a plain video background. They exercise genuine decoding, speech recognition, pretrained inference, and acquisition paths. They are not natural review benchmarks and don't establish sentiment accuracy, robustness, or novelty. Saved smoke evidence identifies the fixture source and model revisions. The Claude integration needs an actual configured API key for a live paid-provider check; mocked protocol tests alone don't establish provider availability.

Model cards: [Whisper conversion](https://huggingface.co/Systran/faster-whisper-base.en), [RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B). Downloaded models keep their original licences; the Qwen 3B checkpoint uses the Qwen research licence.
