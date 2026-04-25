"""Human-readable review reports and spreadsheet-safe batch summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from .storage import atomic_text


def _cell(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
        .replace("\r", "")
    )


def _quote(text: str) -> str:
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


def render_report(result: dict[str, Any]) -> str:
    """Render the actual acquired evidence, decisions, failures and provenance."""
    lines = [
        "# Video review analysis",
        "",
        f"Decision: **{result['prediction'] or 'abstain'}**",
        "",
        f"Source: {_cell(result['source']['path'])}",
        "",
        f"SHA-256: `{result['source']['sha256']}`",
        "",
        f"Inputs attempted: {', '.join(result['acquired']) or 'none'}. Inputs used by final scorer: {', '.join(result['used_inputs']) or 'none'}.",
        "",
        f"Effort: {result['cost']:g} / {result['budget']:g}. Elapsed: {result['elapsed_seconds']:g} seconds.",
        "",
        f"Stopped: `{result['stopping_reason']}`. Controller: `{result['policy']}`.",
        "",
        "## Model scores",
        "",
        "| Label | Relative score |",
        "|---|---:|",
    ]
    lines += [
        f"| {_cell(label)} | {score:.2%} |" for label, score in result["scores"].items()
    ]
    lines += ["", result["score_note"], "", "## Acquisition trail", ""]
    for step in result["steps"]:
        lines += [
            f"### {_cell(step['modality'])}: {_cell(step['status'])}",
            "",
            f"Effort +{step['incremental_cost']:g}; elapsed {step['elapsed_seconds']:g} seconds.",
            "",
        ]
        if step["status"] == "failed":
            lines += [_quote(step["error"]), ""]
            continue
        lines += [f"Model: {_cell(step['model'])}", "", _cell(step["details"]), ""]
        evidence = step.get("evidence", {})
        if evidence.get("transcript"):
            lines += ["Transcript:", "", _quote(evidence["transcript"]), ""]
        if evidence.get("segments"):
            lines += ["| Start (s) | End (s) | Speech |", "|---:|---:|---|"]
            lines += [
                f"| {s['start']:g} | {s['end']:g} | {_cell(s['text'])} |"
                for s in evidence["segments"]
            ]
            lines += [""]
        if evidence.get("frames"):
            lines += ["Sampled frames:", ""]
            lines += [
                f"- {f['timestamp']:g}s: `{_cell(f['file'])}` (in the job's frames directory)"
                for f in evidence["frames"]
            ]
            lines += [""]
        if evidence.get("windows"):
            lines += [
                "Audio windows: "
                + ", ".join(
                    f"{w['start']:g}-{w['end']:g}s" for w in evidence["windows"]
                ),
                "",
            ]
    lines += [
        "## Runtime and configuration",
        "",
        _quote(
            json.dumps(
                {
                    "runtime": result["runtime"],
                    "configuration": result["configuration"],
                },
                indent=2,
            )
        ),
        "",
        result["cost_note"],
        "",
        "Sentiment describes the expressed review opinion. Scores are uncalibrated; this report does not establish benchmark accuracy.",
        "",
    ]
    return "\n".join(lines)


def render_failure(source: str, error: str) -> str:
    """Keep an actionable report even when media fails before any model runs."""
    return (
        "# Video analysis failed\n\nSource: "
        + _cell(source)
        + "\n\n"
        + _quote(error)
        + "\n\nNo successful result is claimed. Resume with --retry-failed after correcting the problem.\n"
    )


def _csv_text(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_summaries(directory: Path, manifest: dict[str, Any]) -> None:
    """Replace CSV and Markdown indexes after each completed or interrupted job."""
    stream = io.StringIO(newline="")
    columns = [
        "file",
        "source",
        "status",
        "prediction",
        "used_inputs",
        "cost",
        "elapsed_seconds",
        "report",
        "error",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    lines = [
        "# ModalEnough batch",
        "",
        f"Run: `{manifest['id']}`",
        "",
        "| File | Status | Decision | Used inputs |",
        "|---|---|---|---|",
    ]
    for job in manifest["jobs"]:
        result: dict[str, Any] = {}
        if job.get("result_sha256"):
            try:
                payload = (directory / job["id"] / "result.json").read_bytes()
                if hashlib.sha256(payload).hexdigest() == job["result_sha256"]:
                    candidate = json.loads(payload)
                    if (
                        isinstance(candidate, dict)
                        and isinstance(candidate.get("used_inputs"), list)
                        and all(isinstance(v, str) for v in candidate["used_inputs"])
                    ):
                        result = candidate
            except (OSError, ValueError):
                pass
        row = {
            "file": Path(job["source"]).name,
            "source": job["source"],
            "status": job["status"],
            "prediction": result.get("prediction") or ("abstain" if result else ""),
            "used_inputs": ", ".join(result.get("used_inputs", [])),
            "cost": result.get("cost", ""),
            "elapsed_seconds": result.get("elapsed_seconds", ""),
            "report": (
                job["id"] + "/report.md"
                if (directory / job["id"] / "report.md").is_file()
                else ""
            ),
            "error": job.get("error", ""),
        }
        writer.writerow({key: _csv_text(value) for key, value in row.items()})
        lines.append(
            f"| {_cell(row['file'])} | {row['status']} | {_cell(row['prediction'])} | {_cell(row['used_inputs'])} |"
        )
    lines += [
        "",
        "Per-video directories contain full JSON results, Markdown reports, and acquired evidence. events.jsonl records progress. Failed or degraded jobs are not clean completions.",
        "",
    ]
    atomic_text(directory / "summary.csv", stream.getvalue())
    atomic_text(directory / "summary.md", "\n".join(lines))
