"""Read-only setup checks; no encoder weights are loaded and no API is called."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Settings
from .pipeline import COSTS


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _weights(root: Path) -> bool:
    if any(
        _nonempty(root / name) for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            try:
                mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                if not isinstance(mapping, dict) or not all(
                    isinstance(v, str) for v in mapping.values()
                ):
                    return False
                shards = set(mapping.values())
                return bool(shards) and all(
                    Path(shard).name == shard and _nonempty(root / shard)
                    for shard in shards
                )
            except (OSError, ValueError, KeyError, TypeError):
                return False
    return False


def model_check(model_root: Path, key: str) -> dict[str, Any]:
    """Check required config/tokenizer resources and every indexed weight shard."""
    from .backends import MODELS

    root = model_root / MODELS[key]["directory"]
    if key == "multimodal":
        cached = model_root / ("qwen-omni-nf4-" + MODELS[key]["revision"][:8])
        if (cached / "modalenough-ready.json").is_file():
            root = cached
    required = ["config.json"]
    if key == "speech":
        required += ["model.bin", "tokenizer.json"]
    else:
        required += ["vocab.json", "merges.txt"]
    if key == "multimodal":
        required += ["preprocessor_config.json", "tokenizer_config.json", "spk_dict.pt"]
    missing = [name for name in required if not _nonempty(root / name)]
    if key != "speech" and not _weights(root):
        missing.append("model weights or indexed shards")
    return {
        "name": f"model:{key}",
        "ok": not missing,
        "detail": (
            str(root)
            if not missing
            else f"Missing {', '.join(missing)} in {root}. Run modalenough download-models."
        ),
    }


def diagnose(settings: Settings) -> dict[str, Any]:
    """Check executables, installed libraries, device and selected model resources."""
    checks: list[dict[str, Any]] = []
    for name in ("ffmpeg", "ffprobe"):
        binary = shutil.which(name)
        ok = False
        detail = "Install FFmpeg and put its bin directory on PATH."
        if binary:
            try:
                result = subprocess.run(
                    [binary, "-version"], capture_output=True, timeout=10, check=True
                )
                detail = result.stdout.decode("utf-8", errors="replace").splitlines()[0]
                ok = True
            except (OSError, subprocess.SubprocessError, IndexError) as exc:
                detail = f"Cannot execute {binary}: {exc}"
        checks.append({"name": name, "ok": ok, "detail": detail})
    enabled = [m for m in settings.modalities if COSTS[m] <= settings.budget + 1e-9]
    packages = ["torch", "numpy", "soundfile", "transformers", "faster-whisper"]
    native = "audio" in enabled or ("vision" in enabled and settings.backend == "local")
    if native:
        packages += ["accelerate", "bitsandbytes", "librosa", "num2words"]
    if settings.backend == "claude":
        packages += ["anthropic"]
        checks.append(
            {
                "name": "claude-key",
                "ok": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
                "detail": (
                    "ANTHROPIC_API_KEY present"
                    if os.environ.get("ANTHROPIC_API_KEY", "").strip()
                    else "Set ANTHROPIC_API_KEY in the process environment."
                ),
            }
        )
    for name in packages:
        try:
            version = importlib.metadata.version(name)
            checks.append({"name": name, "ok": True, "detail": version})
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "detail": 'Missing. Install modalenough video dependencies: pip install "selective-multimodal[video]" (or pip install -e ".[video]" from source).',
                }
            )
    try:
        import torch

        cuda = torch.cuda.is_available()
        requires_cuda = settings.device == "cuda" or native
        ok = not requires_cuda or (cuda and settings.device != "cpu")
        detail = f"torch {torch.__version__}; CUDA runtime {torch.version.cuda}; "
        detail += torch.cuda.get_device_name(0) if cuda else "no usable CUDA GPU"
        if not ok:
            detail += ". Native audio/vision needs CUDA PyTorch and an NVIDIA GPU; use --modalities text --device cpu for transcript-only processing."
        checks.append({"name": "device", "ok": ok, "detail": detail})
    except (ImportError, OSError, RuntimeError) as exc:
        checks.append(
            {
                "name": "device",
                "ok": False,
                "detail": f"PyTorch cannot initialise: {exc}",
            }
        )
    if "text" in enabled:
        checks.append(model_check(settings.model_root, "speech"))
        if settings.backend == "local":
            checks.append(model_check(settings.model_root, "text"))
    if native:
        checks.append(model_check(settings.model_root, "multimodal"))
    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "settings": settings.as_dict(),
        "note": "Checks verify local setup and required files, not model accuracy or paid-provider connectivity.",
    }
