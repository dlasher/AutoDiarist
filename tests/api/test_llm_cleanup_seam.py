"""Service-level tests for the opt-in LLM cleanup seam (card t_48c2efae).

AC4: the cleanup client is caller-injected; the service itself never reads
model output and no env toggle hardwires a network call. AC10: with a fake
injected through the seam, the job completes offline and the report shows
only permitted differences.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("MOCK_PIPELINE", "1")

import pytest

from api import main
from pipeline.llm_cleanup import CleanupSettings, authoritative_lines


def echo(prompt: str) -> str:
    return "\n".join(f"{i}|{line}" for i, line in enumerate(authoritative_lines(prompt), 1))


def evil(prompt: str) -> str:
    return "\n".join(
        f"{i}|{line.replace('welcome', 'greeting')}"
        for i, line in enumerate(authoritative_lines(prompt), 1)
    )


@pytest.fixture()
def job_root(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    job_dir = root / "tok1"
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(
        json.dumps({"token": "tok1", "status": "queued", "progress": 0.0,
                    "message": "", "error": ""})
    )
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    return root


def _run(job_root, **kwargs):
    main.process_job_background(
        "tok1", str(job_root / "tok1" / "audio_source.wav"), "demo.wav", **kwargs
    )
    return json.loads((job_root / "tok1" / "status.json").read_text())


def test_cleanup_is_opt_in_and_default_off(job_root):
    """No injected callable -> no cleanup, no report file, job still done."""
    state = _run(job_root)
    assert state["status"] == "done"
    assert not (job_root / "tok1" / "llm_cleanup_report.md").exists()
    md = (job_root / "tok1" / "markdown_report.md").read_text()
    assert "LLM transcript cleanup" not in md


def test_injected_llm_runs_pass_and_writes_report(job_root):
    state = _run(job_root, llm=echo,
                 cleanup_settings=CleanupSettings(chunk_char_budget=500, max_retries=0))
    assert state["status"] == "done"
    report = (job_root / "tok1" / "llm_cleanup_report.md").read_text()
    assert "accepted" in report.lower()
    md = (job_root / "tok1" / "markdown_report.md").read_text()
    assert "## LLM transcript cleanup" in md
    # transcript content is preserved by the echo pass
    result = json.loads((job_root / "tok1" / "output_result.json").read_text())
    assert "welcome" in result["segments"][0]["text"]


def test_wording_altering_llm_is_rejected_by_service(job_root):
    """The guard runs inside the service path: altered wording keeps originals."""
    state = _run(job_root, llm=evil,
                 cleanup_settings=CleanupSettings(chunk_char_budget=500, max_retries=0))
    assert state["status"] == "done"
    report = (job_root / "tok1" / "llm_cleanup_report.md").read_text()
    assert "rejected" in report.lower()
    result = json.loads((job_root / "tok1" / "output_result.json").read_text())
    assert "welcome" in result["segments"][0]["text"]
    assert "greeting" not in result["segments"][0]["text"]


def test_service_source_hardwires_no_network_call():
    source = Path(inspect.getsourcefile(main)).read_text(encoding="utf-8")
    for forbidden in ("urllib.request", "httpx", "requests.", "openai",
                      "LLM_CLEANUP_ENABLE", "LLM_CLEANUP_URL"):
        assert forbidden not in source, f"service must not hardwire {forbidden!r}"
