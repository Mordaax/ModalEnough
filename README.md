# ModalEnough

ModalEnough predicts sentiment from a video review and records which inputs it needed. It starts with a speech transcript, then can request native audio analysis or sampled video frames if the evidence is still uncertain. Every stage has an effort cost; the pipeline can stop early or abstain.

Whisper transcribes speech, RoBERTa classifies the words, and a quantized Qwen2.5-Omni model handles audio and frames on an NVIDIA GPU. Processing runs locally by default. Claude is an optional backend for acquired transcripts and images.

## Install

Use Python 3.11+ and install FFmpeg with `ffmpeg` and `ffprobe` on PATH.

```bash
python -m venv .venv
# Activate .venv, then:
python -m pip install -e ".[dev,video]"
python -m selective_multimodal.video.cli download-models
```

Native audio/frame inference needs CUDA-enabled PyTorch. The tested setup uses an RTX 5060 Ti, PyTorch 2.9.0, and CUDA 12.8; the [video guide](docs/video.md) has the Windows command and CPU-only text option.

## Analyse a review

```bash
python -m selective_multimodal.video.cli predict review.mp4 --budget 0.55 --output runs/review.json
python -m selective_multimodal.video.cli predict review.mp4 --all
```

Use an English video up to three minutes long. The JSON output contains the sentiment or abstention, class scores, acquired inputs, transcript, frame timestamps, stage timings, costs, and stopping reason. Extracted evidence stays under ignored `runs/` directories.

Text costs 0.20 effort units, additional audio understanding 0.35, and frames 0.45. These are budget units, not dollars or measured compute savings. Text includes audio decoding and transcription. The first native request prepares a smaller model cache, so cold and warm timings differ considerably.

To use Claude, set `ANTHROPIC_API_KEY` in your environment and pass `--backend claude`. Only acquired transcripts, frames, and any existing audio-model summary are sent to Anthropic. Speech processing stays local. `.env.example` lists settings without loading them automatically; never commit a real key.

## Evaluation

```bash
python -m pytest -q
```

The video pipeline currently uses confidence-based acquisition. Its scores have not been calibrated on real reviews, and benchmark sentiment accuracy has not been established. The separate learned-policy experiment and its saved results are documented in the [synthetic benchmark guide](docs/synthetic.md).

The text model was trained on tweets, audio uses up to three 10-second windows, and vision samples four frames. Sarcasm, mixed opinions, language differences, and transcription errors can mislead the system. Silent footage alone produces an abstention.

See [video setup and method](docs/video.md), [smoke evidence](reports/video/README.md), and the [prior-work review](docs/prior_work.md). Source is MIT; downloaded models retain their own licences, including Qwen2.5-Omni-3B's research licence.
