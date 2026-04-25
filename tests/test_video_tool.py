"""User-facing workflow tests; only the expensive model boundary is replaced."""

import json
from dataclasses import replace

import pytest

from selective_multimodal.video import config, jobs
from selective_multimodal.video.storage import directory_lock


@pytest.fixture
def settings(tmp_path):
    return config.Settings(
        model_root=tmp_path / "models",
        output_root=tmp_path / "results",
        device="cpu",
        modalities=("text",),
    )


@pytest.fixture
def clip(tmp_path):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required for workflow integration")
    target = tmp_path / "review.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=0.3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.3",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(target),
        ],
        check=True,
    )
    return target


class ReviewBackend:
    def acquire(self, modality, media, info, directory, observed):
        return {
            "scores": {"positive": 0.96, "negative": 0.02, "neutral": 0.02},
            "model": "test-boundary",
            "scoring_inputs": ["text"],
            "linguistic_evidence": True,
            "evidence": {"transcript": "I love this camera.\n```\n# untrusted review"},
            "details": "Review assessment",
        }

    def runtime(self):
        return {"backend": "test-boundary"}


def factory(settings):
    return ReviewBackend()


def test_settings_resolve_file_env_and_explicit_overrides(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model_root = "weights"\nbudget = 0.55\ndevice = "cpu"\nmodalities = ["text"]\n',
        encoding="utf-8",
    )
    value = config.load_settings(path, {"budget": 0.2}, {"MODALENOUGH_DEVICE": "cuda"})
    assert value.model_root == tmp_path / "weights"
    assert value.budget == 0.2
    assert value.device == "cuda"
    assert value.modalities == ("text",)


@pytest.mark.parametrize(
    "text",
    [
        "unknown = 1",
        "budget = -1",
        'device = "magic"',
        'budget = "0.2"',
        "modalities = []",
        'backend = "remote"',
        "confidence = nan",
    ],
)
def test_invalid_configuration_is_actionable(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_settings(path, environ={})


def test_init_never_overwrites_existing_config(settings, tmp_path):
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    assert config.load_settings(path, environ={}).model_root == settings.model_root
    with pytest.raises(FileExistsError):
        config.write_config(path, replace(settings, budget=0.2))


def test_batch_writes_readable_and_machine_reports(settings, clip, tmp_path):
    destination = tmp_path / "batch"
    result = jobs.run_batch([clip], destination, settings, backend_factory=factory)
    assert result["jobs"][0]["status"] == "complete"
    job = destination / result["jobs"][0]["id"]
    value = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert value["prediction"] == "positive"
    assert value["source"]["sha256"] == jobs.file_hash(clip)
    assert value["used_inputs"] == ["text"]
    report = (job / "report.md").read_text(encoding="utf-8")
    assert "positive" in report and "test-boundary" in report
    assert "````" in report  # embedded fences cannot escape the quoted transcript
    assert (destination / "summary.csv").is_file()
    assert "finished" in (destination / "events.jsonl").read_text(encoding="utf-8")


def test_batch_continues_after_invalid_media(settings, clip, tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not video")
    result = jobs.run_batch(
        [broken, clip], tmp_path / "batch", settings, backend_factory=factory
    )
    assert [v["status"] for v in result["jobs"]] == ["failed", "complete"]
    assert result["counts"]["failed"] == 1


def test_resume_skips_complete_but_detects_changed_source(settings, clip, tmp_path):
    destination = tmp_path / "batch"
    first = jobs.run_batch([clip], destination, settings, backend_factory=factory)

    def unavailable(settings):
        raise AssertionError("completed results must not load a model")

    second = jobs.run_batch(
        [clip], destination, settings, resume=True, backend_factory=unavailable
    )
    assert second["jobs"][0]["id"] == first["jobs"][0]["id"]
    clip.write_bytes(clip.read_bytes() + b"changed")
    third = jobs.run_batch(
        [clip], destination, settings, resume=True, backend_factory=factory
    )
    assert third["jobs"][0]["id"] != first["jobs"][0]["id"]
    assert third["jobs"][0]["status"] == "complete"


@pytest.mark.parametrize("damaged", ["{}", "[]", "null", "42", '{"used_inputs":1}'])
def test_resume_rebuilds_damaged_result(settings, clip, tmp_path, damaged):
    dest = tmp_path / "batch"
    first = jobs.run_batch([clip], dest, settings, backend_factory=factory)
    (dest / first["jobs"][0]["id"] / "result.json").write_text(
        damaged, encoding="utf-8"
    )
    result = jobs.run_batch(
        [clip], dest, settings, resume=True, backend_factory=factory
    )
    value = json.loads(
        (dest / result["jobs"][0]["id"] / "result.json").read_text(encoding="utf-8")
    )
    assert value["prediction"] == "positive"


def test_resume_rejects_different_settings(settings, clip, tmp_path):
    dest = tmp_path / "batch"
    jobs.run_batch([clip], dest, settings, backend_factory=factory)
    with pytest.raises(ValueError, match="configuration"):
        jobs.run_batch(
            [clip],
            dest,
            replace(settings, confidence=0.9),
            resume=True,
            backend_factory=factory,
        )


def test_cancellation_preserves_completed_jobs(settings, clip, tmp_path):
    other = tmp_path / "other.mp4"
    other.write_bytes(clip.read_bytes())

    class Interrupted(ReviewBackend):
        def acquire(self, modality, media, *args):
            if media.name == "other.mp4":
                raise KeyboardInterrupt
            return super().acquire(modality, media, *args)

    dest = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        jobs.run_batch(
            [clip, other], dest, settings, backend_factory=lambda s: Interrupted()
        )
    saved = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert [j["status"] for j in saved["jobs"]] == ["complete", "cancelled"]
    resumed = jobs.run_batch(
        [clip, other], dest, settings, resume=True, backend_factory=factory
    )
    assert [j["status"] for j in resumed["jobs"]] == ["complete", "complete"]


def test_directory_lock_prevents_two_writers(tmp_path):
    with (
        directory_lock(tmp_path),
        pytest.raises(RuntimeError, match="already"),
        directory_lock(tmp_path),
    ):
        pass
    with directory_lock(tmp_path):
        pass


def test_discovery_deduplicates_and_handles_duplicate_names(clip, tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    other = sub / "review.mp4"
    other.write_bytes(clip.read_bytes())
    assert len(jobs.discover_inputs([tmp_path, clip], recursive=True)) == 2
    assert jobs.discover_inputs([tmp_path], recursive=False) == [clip]


def test_failed_models_are_not_reported_as_success(settings, clip, tmp_path):
    class Failed(ReviewBackend):
        def acquire(self, *args):
            raise RuntimeError("weights missing")

    result = jobs.run_batch(
        [clip], tmp_path / "batch", settings, backend_factory=lambda s: Failed()
    )
    assert result["jobs"][0]["status"] == "failed"
    assert "weights missing" in result["jobs"][0]["error"]


def test_csv_prevents_filename_formula_execution(settings, clip, tmp_path):
    bad = tmp_path / "=formula.mp4"
    bad.write_bytes(clip.read_bytes())
    dest = tmp_path / "batch"
    jobs.run_batch([bad], dest, settings, backend_factory=factory)
    import csv

    with (dest / "summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["file"] == "'=formula.mp4"


def test_doctor_rejects_missing_model_shards(settings):
    from selective_multimodal.video.diagnostics import model_check

    root = settings.model_root / "sentiment"
    root.mkdir(parents=True)
    for name in ("config.json", "vocab.json", "merges.txt"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"missing.safetensors"}}', encoding="utf-8"
    )
    result = model_check(settings.model_root, "text")
    assert not result["ok"]
    assert "missing" in result["detail"].lower()
    (root / "missing.safetensors").write_bytes(b"weight-boundary")
    assert model_check(settings.model_root, "text")["ok"]


def test_resume_does_not_retry_failed_jobs_without_request(settings, clip, tmp_path):
    class Failed(ReviewBackend):
        def acquire(self, *args):
            raise RuntimeError("fixture unavailable")

    dest = tmp_path / "batch"
    jobs.run_batch([clip], dest, settings, backend_factory=lambda s: Failed())

    def unavailable(settings):
        raise AssertionError("failed jobs need an explicit retry")

    saved = jobs.run_batch(
        [clip], dest, settings, resume=True, backend_factory=unavailable
    )
    assert saved["jobs"][0]["attempt"] == 1
    retried = jobs.run_batch(
        [clip], dest, settings, resume=True, retry_failed=True, backend_factory=factory
    )
    assert retried["jobs"][0]["status"] == "complete"
    assert retried["jobs"][0]["attempt"] == 2


def test_partial_acquisition_failure_is_degraded(settings, clip, tmp_path):
    class Partial(ReviewBackend):
        def acquire(self, modality, *args):
            if modality != "text":
                raise RuntimeError("native model unavailable")
            return super().acquire(modality, *args)

    settings = replace(settings, modalities=("text", "audio"), strategy="all")
    result = jobs.run_batch(
        [clip], tmp_path / "batch", settings, backend_factory=lambda s: Partial()
    )
    assert result["jobs"][0]["status"] == "degraded"
    assert result["counts"]["degraded"] == 1


def test_cli_prediction_json_stdout_and_additional_export(
    settings, clip, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    monkeypatch.setattr(cli, "diagnose", lambda settings: {"ok": True, "checks": []})
    monkeypatch.setattr(jobs, "make_backend", factory)
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    export = tmp_path / "export.json"
    assert (
        cli.main(
            [
                "--config",
                str(path),
                "predict",
                str(clip),
                "--json",
                "--output",
                str(export),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    value = json.loads(captured.out)
    assert value["prediction"] == "positive"
    assert json.loads(export.read_text(encoding="utf-8")) == value
    assert "Transcribing" in captured.err
    assert len(list(settings.output_root.glob("*/manifest.json"))) == 1


def test_cli_reports_bad_input_without_traceback(settings, tmp_path, capsys):
    from selective_multimodal.video import cli

    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    assert (
        cli.main(["--config", str(path), "predict", str(tmp_path / "missing.mp4")]) == 2
    )
    captured = capsys.readouterr()
    assert not captured.out
    assert "local video" in captured.err
    assert "Traceback" not in captured.err


def test_cli_config_and_doctor_json(settings, tmp_path, monkeypatch, capsys):
    from selective_multimodal.video import cli

    path = tmp_path / "settings.toml"
    monkeypatch.setenv("MODALENOUGH_MODEL_ROOT", str(settings.model_root))
    assert cli.main(["--config", str(path), "config", "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--config", str(path), "config", "show"]) == 0
    assert json.loads(capsys.readouterr().out)["model_root"] == str(settings.model_root)
    monkeypatch.setattr(
        cli,
        "diagnose",
        lambda settings: {
            "ok": False,
            "checks": [{"name": "fixture", "ok": False, "detail": "missing weights"}],
        },
    )
    assert cli.main(["--config", str(path), "doctor", "--json"]) == 2
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_cli_refuses_retry_flag_without_resume(settings, tmp_path, capsys):
    from selective_multimodal.video import cli

    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    assert (
        cli.main(["--config", str(path), "batch", str(tmp_path), "--retry-failed"]) == 2
    )
    assert "--resume" in capsys.readouterr().err


def test_failed_media_has_its_own_readable_failure_report(settings, tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"broken")
    dest = tmp_path / "batch"
    manifest = jobs.run_batch([path], dest, settings, backend_factory=factory)
    directory = dest / manifest["jobs"][0]["id"]
    assert (directory / "error.json").is_file()
    assert "failed" in (directory / "report.md").read_text(encoding="utf-8").lower()
    assert "report.md" in (dest / "summary.csv").read_text(encoding="utf-8")


def test_new_run_never_overwrites_an_unrelated_directory(settings, clip, tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    summary = dest / "summary.csv"
    summary.write_text("keep my data", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        jobs.run_batch([clip], dest, settings, backend_factory=factory)
    assert summary.read_text(encoding="utf-8") == "keep my data"


@pytest.mark.parametrize("weight_map", [[], None, {"x": "../outside.safetensors"}])
def test_doctor_handles_invalid_weight_indexes(settings, weight_map):
    from selective_multimodal.video.diagnostics import model_check

    root = settings.model_root / "sentiment"
    root.mkdir(parents=True)
    for name in ("config.json", "vocab.json", "merges.txt"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    assert not model_check(settings.model_root, "text")["ok"]


@pytest.mark.parametrize(
    "manifest",
    [
        {"schema_version": 1, "jobs": []},
        {
            "schema_version": 1,
            "jobs": [{"id": "../escape", "source": "/tmp/input.mp4"}],
        },
        {
            "schema_version": 1,
            "id": "abc",
            "configuration": {},
            "jobs": [{"id": "abc", "source": "relative.mp4", "status": "complete"}],
        },
    ],
)
def test_invalid_manifest_gives_a_resumable_error(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        jobs.read_manifest(tmp_path)


def test_deleted_input_does_not_keep_a_stale_success_in_summary(
    settings, clip, tmp_path
):
    dest = tmp_path / "batch"
    jobs.run_batch([clip], dest, settings, backend_factory=factory)
    clip.unlink()
    saved = jobs.run_batch([], dest, settings, resume=True, backend_factory=factory)
    assert saved["jobs"][0]["status"] == "failed"
    import csv

    with (dest / "summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["prediction"] == ""


def test_cli_batch_json_and_resume_restore_saved_settings(
    settings, clip, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    monkeypatch.setattr(cli, "diagnose", lambda settings: {"ok": True, "checks": []})
    monkeypatch.setattr(jobs, "make_backend", factory)
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    dest = tmp_path / "batch"
    assert (
        cli.main(
            [
                "--config",
                str(path),
                "batch",
                str(clip),
                "--output-dir",
                str(dest),
                "--json",
                "--quiet",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["counts"] == {"complete": 1}
    assert cli.main(["batch", "--resume", str(dest), "--json", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"][0]["attempt"] == 1


def test_cli_cancellation_returns_130_and_can_resume(
    settings, clip, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    class Interrupted(ReviewBackend):
        def acquire(self, *args):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "diagnose", lambda settings: {"ok": True, "checks": []})
    monkeypatch.setattr(jobs, "make_backend", lambda s: Interrupted())
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    dest = tmp_path / "batch"
    assert (
        cli.main(["--config", str(path), "batch", str(clip), "--output-dir", str(dest)])
        == 130
    )
    captured = capsys.readouterr()
    assert "--resume" in captured.err
    assert jobs.read_manifest(dest)["jobs"][0]["status"] == "cancelled"


def test_cli_setup_failure_does_not_create_a_run(
    settings, clip, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    monkeypatch.setattr(
        cli,
        "diagnose",
        lambda s: {
            "ok": False,
            "checks": [{"name": "weights", "ok": False, "detail": "download-models"}],
        },
    )
    assert cli.main(["--config", str(path), "predict", str(clip)]) == 2
    assert "download-models" in capsys.readouterr().err
    assert not settings.output_root.exists()


def test_cli_failed_decode_returns_one_and_json_error(
    settings, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"broken")
    monkeypatch.setattr(cli, "diagnose", lambda s: {"ok": True, "checks": []})
    assert cli.main(["--config", str(path), "predict", str(bad), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_doctor_reports_missing_decoders_key_gpu_and_weights(settings, monkeypatch):
    import torch

    from selective_multimodal.video import diagnostics

    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    value = diagnostics.diagnose(
        replace(
            settings,
            backend="claude",
            device="cuda",
            modalities=("text", "audio", "vision"),
        )
    )
    failures = {item["name"] for item in value["checks"] if not item["ok"]}
    assert {
        "ffmpeg",
        "ffprobe",
        "claude-key",
        "device",
        "model:speech",
        "model:multimodal",
    } <= failures
    assert not value["ok"]


def test_config_home_and_environment_values_are_validated(tmp_path):
    settings = config.load_settings(
        environ={
            "MODALENOUGH_HOME": str(tmp_path),
            "MODALENOUGH_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "MODALENOUGH_MODALITIES": "text,vision",
            "MODALENOUGH_BUDGET": "0.65",
        }
    )
    assert settings.output_root == tmp_path / "outputs"
    assert settings.modalities == ("text", "vision")
    assert settings.budget == 0.65
    with pytest.raises(ValueError, match="number"):
        config.load_settings(environ={"MODALENOUGH_BUDGET": "invalid"})
    with pytest.raises(ValueError, match="does not exist"):
        config.load_settings(tmp_path / "missing.toml", environ={})
    path = tmp_path / "invalid.toml"
    path.write_text("a = [", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read"):
        config.load_settings(path, environ={})


def test_process_exit_releases_the_directory_lock(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    source_root = str(Path(config.__file__).resolve().parents[2])
    code = (
        f"import sys; sys.path.insert(0, {source_root!r}); "
        + "from pathlib import Path; from selective_multimodal.video.storage import directory_lock; import sys,time; lock=directory_lock(Path(sys.argv[1])); lock.__enter__(); print('ready',flush=True); time.sleep(30)"
    )
    child = subprocess.Popen(
        [getattr(sys, "_base_executable", sys.executable), "-c", code, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="already"), directory_lock(tmp_path):
            pass
    finally:
        child.kill()
        child.wait(timeout=10)
        child.stdout.close()
    with directory_lock(tmp_path):
        pass


def test_export_cannot_overwrite_run_state(
    settings, clip, tmp_path, monkeypatch, capsys
):
    from selective_multimodal.video import cli

    monkeypatch.setattr(cli, "diagnose", lambda settings: {"ok": True, "checks": []})
    monkeypatch.setattr(jobs, "make_backend", factory)
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    destination = tmp_path / "batch"
    assert (
        cli.main(
            [
                "--config",
                str(path),
                "predict",
                str(clip),
                "--output-dir",
                str(destination),
                "--output",
                str(destination / "manifest.json"),
            ]
        )
        == 2
    )
    assert not (destination / "manifest.json").exists()
    assert "outside" in capsys.readouterr().err


def test_cli_human_reports_and_doctor(settings, clip, tmp_path, monkeypatch, capsys):
    from selective_multimodal.video import cli

    monkeypatch.setattr(
        cli,
        "diagnose",
        lambda settings: {
            "ok": True,
            "checks": [{"name": "test", "ok": True, "detail": "ready"}],
            "note": "No API calls.",
        },
    )
    monkeypatch.setattr(jobs, "make_backend", factory)
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    assert cli.main(["--config", str(path), "predict", str(clip)]) == 0
    captured = capsys.readouterr()
    assert "POSITIVE" in captured.out and "report.md" in captured.out
    assert cli.main(["--config", str(path), "batch", str(clip)]) == 0
    assert "summary.csv" in capsys.readouterr().out
    assert cli.main(["--config", str(path), "doctor"]) == 0
    assert "ready" in capsys.readouterr().out


def test_download_command_respects_model_root_without_network(
    settings, tmp_path, monkeypatch, capsys
):
    import huggingface_hub

    from selective_multimodal.video import cli

    def download(repo_id, revision, local_dir, **kwargs):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "test-download").write_text(repo_id, encoding="utf-8")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    path = tmp_path / "config.toml"
    config.write_config(path, settings)
    assert cli.main(["--config", str(path), "download-models", "--text-only"]) == 0
    assert (settings.model_root / "sentiment" / "test-download").is_file()
    assert not (settings.model_root / "qwen-omni").exists()
    assert "Models saved" in capsys.readouterr().out


def test_report_includes_segments_audio_windows_and_frames(settings, clip, tmp_path):
    from selective_multimodal.video.reports import render_report

    dest = tmp_path / "batch"
    saved = jobs.run_batch([clip], dest, settings, backend_factory=factory)
    result = json.loads(
        (dest / saved["jobs"][0]["id"] / "result.json").read_text(encoding="utf-8")
    )
    evidence = result["steps"][0]["evidence"]
    evidence["segments"] = [{"start": 0, "end": 1, "text": "| <script>bad</script>"}]
    evidence["windows"] = [{"start": 0, "end": 10}]
    evidence["frames"] = [{"file": "frame_00.jpg", "timestamp": 1.25}]
    text = render_report(result)
    assert "frame_00.jpg" in text and "0-10s" in text
    assert "&lt;script&gt;" in text and "<script>" not in text


def test_doctor_handles_missing_packages_and_broken_decoders(settings, monkeypatch):
    import importlib.metadata
    import subprocess

    from selective_multimodal.video import diagnostics

    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "broken-decoder")

    def fail_process(*args, **kwargs):
        raise subprocess.TimeoutExpired("broken-decoder", 10)

    monkeypatch.setattr(diagnostics.subprocess, "run", fail_process)

    def missing_package(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(diagnostics.importlib.metadata, "version", missing_package)
    value = diagnostics.diagnose(settings)
    assert not value["ok"]
    assert any(c["name"] == "transformers" and not c["ok"] for c in value["checks"])
    assert all(
        not c["ok"] for c in value["checks"] if c["name"] in ("ffmpeg", "ffprobe")
    )


def test_quantized_cache_can_run_without_the_original_large_checkpoint(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    import torch

    from selective_multimodal.video.backends import MODELS, LocalBackend

    cached = tmp_path / ("qwen-omni-nf4-" + MODELS["multimodal"]["revision"][:8])
    cached.mkdir()
    (cached / "modalenough-ready.json").write_text(
        json.dumps(MODELS["multimodal"]), encoding="utf-8"
    )

    class Loaded:
        def __init__(self, path):
            self.loaded_from = path

        def eval(self):
            return self

    fake = SimpleNamespace(
        BitsAndBytesConfig=lambda **kwargs: object(),
        Qwen2_5OmniForConditionalGeneration=SimpleNamespace(
            from_pretrained=lambda path, **kwargs: Loaded(path)
        ),
        Qwen2_5OmniProcessor=SimpleNamespace(
            from_pretrained=lambda path, **kwargs: Loaded(path)
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    backend = LocalBackend(tmp_path, device="cuda")
    backend._load_omni()
    assert backend._omni.loaded_from == cached
    assert backend._processor.loaded_from == cached
    assert not (tmp_path / MODELS["multimodal"]["directory"]).exists()
