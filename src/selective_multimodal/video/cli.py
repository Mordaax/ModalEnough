"""ModalEnough's local video-analysis command line."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import Settings, data_home, load_settings, write_config
from .diagnostics import diagnose
from .jobs import discover_inputs, read_manifest, run_batch
from .storage import atomic_json


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS, help="TOML settings file"
    )
    parser.add_argument("--model-root", type=Path, help="Downloaded model directory")
    parser.add_argument("--output-root", type=Path, help="Root for new saved runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--backend", choices=["local", "claude"])
    parser.add_argument("--budget", type=float, help="Effort budget, 0 to 1")
    parser.add_argument(
        "--confidence", type=float, help="Uncalibrated score threshold, 0.5 to 0.99"
    )
    parser.add_argument(
        "--claude-model",
        help="Claude model identifier (only used with --backend claude)",
    )
    parser.add_argument(
        "--modalities",
        type=lambda value: tuple(v.strip() for v in value.split(",")),
        help="Enabled inputs: text,audio,vision",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only transcribe and classify words; no native audio/frame analysis",
    )
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument(
        "--all",
        dest="strategy",
        action="store_const",
        const="all",
        help="Process every enabled affordable input",
    )
    strategy.add_argument("--strategy", choices=["adaptive", "all"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modalenough",
        description="Video-review sentiment with budgeted inputs and saved evidence.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="ModalEnough " + version("selective-multimodal"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML settings file (also accepted after the command)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Check local setup without loading model weights"
    )
    _options(doctor)
    doctor.add_argument("--json", action="store_true")
    download = commands.add_parser(
        "download-models", help="Download pinned models explicitly"
    )
    _options(download)
    configuration = commands.add_parser(
        "config", help="Create or inspect persistent settings"
    )
    configuration.add_argument(
        "action", choices=["init", "show"], default="show", nargs="?"
    )
    configuration.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    for name, help_text in [
        ("predict", "Analyse one video and save its evidence"),
        ("batch", "Analyse files or folders, or resume a saved batch"),
    ]:
        command = commands.add_parser(name, help=help_text)
        _options(command)
        command.add_argument(
            "--json",
            action="store_true",
            help="Machine-readable stdout; progress remains on stderr",
        )
        command.add_argument(
            "--quiet", action="store_true", help="Hide progress messages"
        )
        command.add_argument(
            "--output-dir", type=Path, help="Directory for this run (new or empty)"
        )
        if name == "predict":
            command.add_argument("video", type=Path)
            command.add_argument(
                "--output", type=Path, help="Also export result JSON to this file"
            )
        else:
            command.add_argument("paths", nargs="*", type=Path)
            command.add_argument("--recursive", action="store_true")
            command.add_argument(
                "--resume", type=Path, help="Resume a saved run at video boundaries"
            )
            command.add_argument(
                "--retry-failed",
                action="store_true",
                help="Also retry failed/degraded videos when resuming",
            )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {f.name: getattr(args, f.name, None) for f in fields(Settings)}
    if getattr(args, "text_only", False):
        if getattr(args, "modalities", None) and args.modalities != ("text",):
            raise ValueError("--text-only conflicts with --modalities.")
        values["modalities"] = ("text",)
    return {key: value for key, value in values.items() if value is not None}


def _progress(event: dict[str, Any]) -> None:
    if event["type"] == "started":
        # JSON escaping prevents control characters in a filename becoming terminal commands.
        name = json.dumps(Path(event["source"]).name, ensure_ascii=True)
        print(
            f"[{event['index']}/{event['total']}] {name}", file=sys.stderr, flush=True
        )
    elif event["type"] == "acquiring":
        print("  " + event["message"] + "...", file=sys.stderr, flush=True)
    elif event["type"] == "failed":
        print(
            "  Failed: " + json.dumps(event["error"], ensure_ascii=True),
            file=sys.stderr,
            flush=True,
        )
    elif event["type"] == "skipped":
        print(
            f"[{event['index']}/{event['total']}] Reusing completed result.",
            file=sys.stderr,
            flush=True,
        )


def _execute(args: argparse.Namespace) -> int:
    overrides = _overrides(args)
    config_path = getattr(args, "config", None)
    if args.command == "config":
        target = config_path or data_home() / "config.toml"
        if args.action == "init":
            settings = load_settings(config_path if target.is_file() else None)
            write_config(target, settings)
            print(f"Configuration created: {target.resolve()}")
        else:
            print(json.dumps(load_settings(config_path).as_dict(), indent=2))
        return 0
    resume = getattr(args, "resume", None)
    if resume and not config_path and not overrides:
        settings = Settings(
            **read_manifest(resume.resolve())["configuration"]["settings"]
        )
    else:
        settings = load_settings(config_path, overrides)
    if args.command == "doctor":
        result = diagnose(settings)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for check in result["checks"]:
                print(
                    f"{'OK' if check['ok'] else 'FAIL':4} {check['name']}: {check['detail']}"
                )
            print(result["note"])
        return 0 if result["ok"] else 2
    if args.command == "download-models":
        from .backends import prefetch

        with redirect_stdout(sys.stderr):
            prefetch(
                settings.model_root,
                include_multimodal=not getattr(args, "text_only", False)
                and any(m != "text" for m in settings.modalities),
            )
        print(f"Models saved: {settings.model_root}")
        return 0
    if getattr(args, "retry_failed", False) and not resume:
        raise ValueError("--retry-failed requires --resume DIRECTORY.")
    if resume and args.output_dir:
        raise ValueError("--output-dir cannot be combined with --resume.")
    paths = [args.video] if args.command == "predict" else args.paths
    inputs = (
        discover_inputs(paths, recursive=getattr(args, "recursive", False))
        if paths
        else []
    )
    if not inputs and not resume:
        raise ValueError(
            "Supply video files or a directory, or use --resume DIRECTORY."
        )
    export = getattr(args, "output", None)
    if export and export.expanduser().resolve() in inputs:
        raise ValueError("--output must not overwrite an input video.")
    if not resume:
        with redirect_stdout(sys.stderr):
            health = diagnose(settings)
        if not health["ok"]:
            issues = "\n".join(
                f"  {c['name']}: {c['detail']}" for c in health["checks"] if not c["ok"]
            )
            raise ValueError("Setup is incomplete. Run modalenough doctor.\n" + issues)
    destination = (
        (resume or args.output_dir or settings.output_root / uuid.uuid4().hex)
        .expanduser()
        .resolve()
    )
    if export and export.expanduser().resolve().is_relative_to(destination):
        raise ValueError(
            "--output must be outside the managed run directory; result.json is already saved inside it."
        )
    # Libraries occasionally print model-loading messages; keep stdout parseable.
    try:
        with redirect_stdout(sys.stderr):
            manifest = run_batch(
                inputs,
                destination,
                settings,
                resume=bool(resume),
                retry_failed=getattr(args, "retry_failed", False),
                on_event=None if args.quiet else _progress,
            )
    except KeyboardInterrupt:
        print(
            f'Interrupted. Resume with: modalenough batch --resume "{destination}"',
            file=sys.stderr,
        )
        return 130
    failures = sum(
        manifest["counts"].get(s, 0)
        for s in ("failed", "degraded", "cancelled", "pending", "running")
    )
    if args.command == "predict":
        job = manifest["jobs"][0]
        result_file = destination / job["id"] / "result.json"
        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if export:
                atomic_json(export.expanduser().resolve(), result)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
            else:
                label = result["prediction"] or "abstain"
                print(
                    f"{label.upper()} | {job['status']} | used {', '.join(result['used_inputs']) or 'none'} | effort {result['cost']:g}/{result['budget']:g}"
                )
                print(f"Report: {result_file.parent / 'report.md'}")
                print(f"JSON:   {result_file}")
        elif args.json:
            print(
                json.dumps(
                    {
                        "status": job["status"],
                        "error": job.get("error"),
                        "run_directory": str(destination),
                    }
                )
            )
        else:
            print(
                "Analysis failed: "
                + json.dumps(job.get("error", "Unknown error"), ensure_ascii=True),
                file=sys.stderr,
            )
    else:
        if args.json:
            print(
                json.dumps(
                    {**manifest, "run_directory": str(destination)},
                    indent=2,
                    ensure_ascii=True,
                )
            )
        else:
            print(
                ", ".join(
                    f"{count} {status}"
                    for status, count in sorted(manifest["counts"].items())
                )
            )
            print(f"Reports: {destination / 'summary.md'}")
            print(f"CSV:     {destination / 'summary.csv'}")
    if not args.quiet:
        print(f"Saved run: {destination}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning documented process exit codes."""
    try:
        args = _parser().parse_args(argv)
        return _execute(args)
    except (ValueError, FileExistsError) as exc:
        print("modalenough: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ImportError) as exc:
        print("modalenough: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
