"""Persistent CLI settings with explicit, predictable precedence."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .pipeline import VideoOptions


def data_home(environ: Mapping[str, str] | None = None) -> Path:
    """Return the platform data directory without creating it."""
    env = os.environ if environ is None else environ
    if env.get("MODALENOUGH_HOME"):
        return Path(env["MODALENOUGH_HOME"]).expanduser().resolve()
    if os.name == "nt":
        return (
            Path(env.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
            / "ModalEnough"
        )
    return (
        Path(env.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        / "modalenough"
    )


@dataclass(frozen=True)
class Settings:
    """Validated inference and storage configuration; never contains credentials."""

    model_root: Path
    output_root: Path
    backend: str = "local"
    device: str = "auto"
    budget: float = 1.0
    confidence: float = 0.85
    strategy: str = "adaptive"
    modalities: tuple[str, ...] = ("text", "audio", "vision")
    claude_model: str = "claude-sonnet-4-6"

    def __post_init__(self) -> None:
        for name in ("model_root", "output_root"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a nonempty path.")
            object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.backend not in ("local", "claude"):
            raise ValueError("backend must be local or claude.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda.")
        for name in ("budget", "confidence"):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), (int, float)
            ):
                raise ValueError(f"{name} must be a number.")  # noqa: TRY004
        if (
            not isinstance(self.modalities, (list, tuple))
            or not self.modalities
            or any(not isinstance(v, str) for v in self.modalities)
        ):
            raise ValueError("modalities must contain text, audio or vision.")
        object.__setattr__(self, "modalities", tuple(self.modalities))
        if not isinstance(self.claude_model, str) or not self.claude_model.strip():
            raise ValueError("claude_model must be a nonempty model identifier.")
        self.options()

    def options(self) -> VideoOptions:
        """Construct the inference controller's validated options."""
        return VideoOptions(
            self.budget, self.confidence, self.strategy, self.modalities
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe effective settings."""
        result = asdict(self)
        result["model_root"] = str(self.model_root)
        result["output_root"] = str(self.output_root)
        result["modalities"] = list(self.modalities)
        return result


def load_settings(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Resolve CLI > environment > TOML > platform defaults."""
    env = os.environ if environ is None else environ
    home = data_home(env)
    path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else home / "config.toml"
    )
    # Reuse existing editable-install downloads, independently of the current directory.
    legacy = Path(__file__).resolve().parents[3] / "runs" / "models"
    values: dict[str, Any] = {
        "model_root": legacy if legacy.is_dir() else home / "models",
        "output_root": home / "results",
    }
    if path.is_file():
        try:
            with path.open("rb") as stream:
                saved = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
        unknown = set(saved) - {f.name for f in fields(Settings)}
        if unknown:
            raise ValueError(
                "Unknown configuration keys: " + ", ".join(sorted(unknown))
            )
        for key in ("model_root", "output_root"):
            if key in saved and isinstance(saved[key], str):
                saved[key] = path.parent / Path(saved[key]).expanduser()
        values.update(saved)
    elif config_path:
        raise ValueError(f"Configuration file does not exist: {path}")
    for field in fields(Settings):
        key = "MODALENOUGH_" + field.name.upper()
        if key not in env:
            continue
        value: Any = env[key]
        if field.name in ("budget", "confidence"):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be a number.") from exc
        elif field.name == "modalities":
            value = tuple(v.strip() for v in value.split(","))
        values[field.name] = value
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return Settings(**values)


def write_config(path: Path, settings: Settings) -> None:
    """Create a TOML configuration without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ModalEnough settings. API keys belong in the process environment."]
    for key, value in settings.as_dict().items():
        lines.append(f"{key} = {json.dumps(value, ensure_ascii=True)}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
