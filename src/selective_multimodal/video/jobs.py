"""Durable, serial video jobs with content-aware resume and incremental audit logs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .pipeline import VideoPipeline
from .reports import render_failure, render_report, write_summaries
from .storage import atomic_json, atomic_text, directory_lock

VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}
SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return an unambiguous ISO timestamp for an actual processing event."""
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    """Hash content in bounded memory; path metadata alone is not an identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_inputs(paths: Sequence[Path], recursive: bool = False) -> list[Path]:
    """Expand local files/directories deterministically, deduplicating resolved paths."""
    found: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            candidates = sorted(path.rglob("*") if recursive else path.iterdir())
            candidates = [
                p
                for p in candidates
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
            ]
        elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            candidates = [path]
        else:
            raise ValueError(f"Expected a local video file or directory: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            found.setdefault(os.path.normcase(str(resolved)), resolved)
    if not found:
        raise ValueError("No supported videos found (MP4, MOV, MKV, WebM, AVI or M4V).")
    return list(found.values())


def make_backend(settings: Settings) -> Any:
    """Construct one lazy backend; weights load only when an input is acquired."""
    from .backends import ClaudeBackend, LocalBackend

    kwargs: dict[str, Any] = {
        "model_root": settings.model_root,
        "device": settings.device,
    }
    if settings.backend == "claude":
        return ClaudeBackend(**kwargs, model=settings.claude_model)
    return LocalBackend(**kwargs)


def _signature(settings: Settings) -> dict[str, Any]:
    from importlib.metadata import version

    from .backends import MODELS

    return {
        "settings": settings.as_dict(),
        "models": MODELS,
        "tool_version": version("selective-multimodal"),
        "schema_version": SCHEMA_VERSION,
    }


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read and validate the manifest's structural and path-safety boundaries."""
    try:
        value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            value["schema_version"] != SCHEMA_VERSION
            or not isinstance(value["jobs"], list)
            or not value["jobs"]
        ):
            raise ValueError("Unsupported manifest schema.")
        if not isinstance(value["id"], str) or not isinstance(
            value["configuration"], dict
        ):
            raise ValueError("Invalid batch identity or configuration.")  # noqa: TRY004
        Settings(**value["configuration"]["settings"])
        identities = set()
        sources = set()
        for job in value["jobs"]:
            identity = job["id"]
            if (
                not isinstance(identity, str)
                or not identity
                or any(c not in "0123456789abcdef-" for c in identity)
            ):
                raise ValueError("Invalid job directory in manifest.")
            if identity in identities or job["source"] in sources:
                raise ValueError("Duplicate job identity or source.")
            identities.add(identity)
            sources.add(job["source"])
            if job["status"] not in (
                "pending",
                "running",
                "complete",
                "failed",
                "degraded",
                "cancelled",
            ):
                raise ValueError("Unknown job status.")
            if not Path(job["source"]).is_absolute():
                raise ValueError("Manifest sources must be absolute paths.")
        return dict(value)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot resume {directory}: invalid or missing manifest ({exc})."
        ) from exc


def run_batch(
    inputs: Sequence[Path],
    destination: Path,
    settings: Settings,
    *,
    resume: bool = False,
    retry_failed: bool = False,
    backend_factory: Callable[[Settings], Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Process a batch serially, persisting each transition and preserving completed work."""
    destination = destination.expanduser().resolve()
    with directory_lock(destination):
        signature = _signature(settings)
        if resume:
            manifest = read_manifest(destination)
            if manifest["configuration"] != signature:
                raise ValueError(
                    "Resume configuration differs; use the original settings or start a new batch."
                )
            if inputs and [str(Path(p).resolve()) for p in inputs] != [
                j["source"] for j in manifest["jobs"]
            ]:
                raise ValueError(
                    "Resume inputs differ from the saved batch. Omit inputs to use its saved list."
                )
        else:
            if (destination / "manifest.json").exists():
                raise ValueError(
                    "This directory already contains a batch; use --resume or choose another directory."
                )
            if any(p.name != ".modalenough.lock" for p in destination.iterdir()):
                raise ValueError(
                    "A new run needs an empty output directory; choose a new directory or --resume."
                )
            if not inputs:
                raise ValueError("At least one video is required.")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "id": uuid.uuid4().hex,
                "created_at": utc_now(),
                "configuration": signature,
                "jobs": [
                    {
                        "id": uuid.uuid4().hex,
                        "source": str(Path(p).resolve()),
                        "status": "pending",
                    }
                    for p in inputs
                ],
            }
        backend = None
        active: dict[str, Any] | None = None
        interrupted = False

        def save() -> None:
            manifest["updated_at"] = utc_now()
            manifest["counts"] = dict(Counter(j["status"] for j in manifest["jobs"]))
            atomic_json(destination / "manifest.json", manifest)
            write_summaries(destination, manifest)

        def emit(event: dict[str, Any]) -> None:
            event = {
                "time": utc_now(),
                "job_id": active["id"] if active else None,
                **event,
            }
            with (destination / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            if on_event:
                on_event(event)

        save()
        try:
            for index, job in enumerate(manifest["jobs"], 1):
                active = job
                path = Path(job["source"])
                try:
                    digest = file_hash(path)
                    unchanged = digest == job.get("source_sha256")
                    result_path = destination / job["id"] / "result.json"
                    if (
                        unchanged
                        and job["status"] == "complete"
                        and result_path.is_file()
                        and file_hash(result_path) == job.get("result_sha256")
                    ):
                        saved = json.loads(result_path.read_text(encoding="utf-8"))
                        atomic_text(
                            result_path.parent / "report.md", render_report(saved)
                        )
                        emit(
                            {
                                "type": "skipped",
                                "source": str(path),
                                "index": index,
                                "total": len(manifest["jobs"]),
                            }
                        )
                        continue
                    if (
                        unchanged
                        and job["status"] in ("failed", "degraded")
                        and not retry_failed
                    ):
                        emit({"type": "skipped_failed", "source": str(path)})
                        continue
                    if not unchanged and job.get("source_sha256"):
                        job["id"] = uuid.uuid4().hex
                    job.update(
                        source_sha256=digest,
                        status="running",
                        started_at=utc_now(),
                        attempt=job.get("attempt", 0) + 1,
                    )
                    job.pop("error", None)
                    job.pop("result_sha256", None)
                    save()
                    emit(
                        {
                            "type": "started",
                            "source": str(path),
                            "index": index,
                            "total": len(manifest["jobs"]),
                        }
                    )
                    # Validate the clip before constructing a potentially paid backend.
                    from .media import probe

                    info = probe(path)
                    if backend is None:
                        backend = (backend_factory or make_backend)(settings)
                    result = VideoPipeline(backend).run(
                        path,
                        destination / job["id"],
                        settings.options(),
                        on_event=emit,
                        media_info=info,
                    )
                    if file_hash(path) != digest:
                        raise RuntimeError(
                            "Source file changed during inference; run again with a stable input."
                        )
                    result.update(
                        schema_version=SCHEMA_VERSION,
                        job_id=job["id"],
                        source={"path": str(path), "sha256": digest},
                        started_at=job["started_at"],
                        finished_at=utc_now(),
                        configuration=signature,
                    )
                    result_path = destination / job["id"] / "result.json"
                    atomic_json(result_path, result)
                    atomic_text(result_path.parent / "report.md", render_report(result))
                    job["result_sha256"] = file_hash(result_path)
                    job["status"] = (
                        ("degraded" if result["successful_acquisitions"] else "failed")
                        if result["errors"]
                        else "complete"
                    )
                    if result["errors"]:
                        job["error"] = "; ".join(
                            e["modality"] + ": " + e["message"]
                            for e in result["errors"]
                        )
                except Exception as exc:  # noqa: BLE001
                    job.update(status="failed", error=str(exc)[:1000])
                    job.pop("result_sha256", None)
                    failure_dir = destination / job["id"]
                    atomic_json(
                        failure_dir / "error.json",
                        {
                            "status": "failed",
                            "source": str(path),
                            "error": job["error"],
                            "time": utc_now(),
                        },
                    )
                    atomic_text(
                        failure_dir / "report.md",
                        render_failure(str(path), job["error"]),
                    )
                    emit({"type": "failed", "source": str(path), "error": job["error"]})
                job["finished_at"] = utc_now()
                save()
        except KeyboardInterrupt:
            interrupted = True
            if active and active["status"] == "running":
                active.update(
                    status="cancelled",
                    finished_at=utc_now(),
                    error="Interrupted by user; resume to retry this video.",
                )
            emit({"type": "cancelled"})
            raise
        finally:
            manifest["status"] = (
                "cancelled"
                if interrupted
                else (
                    "complete_with_errors"
                    if any(
                        j["status"] in ("failed", "degraded") for j in manifest["jobs"]
                    )
                    else "complete"
                )
            )
            save()
        return manifest
