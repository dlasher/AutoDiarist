"""Tests for scripts/smoke-test.sh (card t_66115e7d).

The script is exercised as a real subprocess against tests/scripts/stub_server.py
(a stdlib stub of the AutoDiarist API on an ephemeral port) — the suite never
touches a container, docker, the GPU, or the real internet.

AC map (amended fixture lifecycle per planner update, 2026-10-22):
  AC1  no_activation_path_reference, no_audio_committed
  AC2  --endpoint ephemeral port (every stub run passes it)
  AC3  fixture lifecycle: downloaded at runtime into a temp dir, used as
       published, removed on exit incl. killed runs (cleanup_after_run,
       cleanup_after_kill), fallback tone when download fails (offline_falls_back_to_tone),
       no committed audio (no_audio_committed)
  AC4  default endpoint is 127.0.0.1:8000 (default_endpoint_is_local_8000)
  AC5  every failure mode exits non-zero naming the failed check:
       docs_404, post_fail, job_failed, timeout
  AC6  exact curl commands are printed (curl_commands_are_printed)
  AC7  disclaimer "ALIVENESS AND PLUMBING ONLY" (disclaimer_printed)
  AC8  missing ffmpeg -> clear actionable error, never a silent pass
       (missing_ffmpeg_clear_error)
  AC9  README documents the script as OPTIONAL (readme_documents_optional)
  AC10 these tests themselves are the stub-server coverage
  INFO informational speaker count, non-gating (speaker_count_informational)
  RR1  review-round-1 blocker regression: a fixture URL that answers 200
       with a NON-WAV body must fall back to a REAL generated tone, must
       state the truth (WAV sanity check failed, not "download failed"),
       must upload genuine RIFF/WAVE bytes, and must exit 2 without ever
       uploading when ffmpeg is absent
       (junk_200_download_falls_back_to_real_tone,
        junk_200_missing_ffmpeg_exits_2_never_uploads)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[1]
SCRIPT = REPO / "scripts" / "smoke-test.sh"
STUB = TESTS_DIR / "stub_server.py"

TIMEOUT = 120


def _env(tmp_path: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # Pin mktemp -d target so we can assert the fixture lifecycle.
    work = tmp_path / "tmpwork"
    work.mkdir(exist_ok=True)
    env["TMPDIR"] = str(work)
    # The script must work without proxy env (card behaviour spec).
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env["NO_PROXY"] = "*"
    env["SMOKE_TEST_POLL_INTERVAL"] = "0.25"
    if extra:
        env.update(extra)
    return env


def _leftovers(tmp_path: Path) -> list[str]:
    work = tmp_path / "tmpwork"
    if not work.exists():
        return []
    return [str(p) for p in work.rglob("*")]


def _run(tmp_path: Path, port: int, *args: str,
         extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = _env(tmp_path, extra_env)
    env.setdefault("SMOKE_TEST_FIXTURE_URL",
                   f"http://127.0.0.1:{port}/fixture/marklex1min.wav")
    return subprocess.run(
        ["bash", str(SCRIPT),
         "--endpoint", f"http://127.0.0.1:{port}", *args],
        capture_output=True, text=True, timeout=TIMEOUT, env=env,
    )


def _start_stub(state: str = "pass") -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(
        [sys.executable, str(STUB), "--state", state],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    assert line.startswith("READY "), f"stub failed to start: {line!r}"
    return proc, int(line.split()[1])


# Pinned TMPDIR contents after any script invocation: our tmpwork dir must hold
# no mktemp dirs. The script's fixture must be gone (AC3 lifecycle).
def test_cleanup_after_run(tmp_path: Path) -> None:
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        assert r.returncode == 0, r.stdout + r.stderr
        assert _leftovers(tmp_path) == [], "smoke-test left files in TMPDIR"
    finally:
        proc.terminate()


def test_all_pass_exit_zero(tmp_path: Path) -> None:
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        # Checks (a)-(f) all PASS, exactly the six gates (t_fef54150 added
        # the cancel-while-processing gate; the stub now answers DELETE
        # mid-processing with 202 and lands the job terminal 'cancelled').
        for tag in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"):
            assert f"PASS" in out and tag in out, f"no PASS line for {tag}:\n{out}"
        # Amended AC3: primary fixture is the downloaded one, and the script
        # says so explicitly.
        assert "downloaded" in out.lower() and "fixture" in out.lower(), out
        # Card: the script states it validates aliveness/plumbing only.
        assert "ALIVENESS AND PLUMBING ONLY" in out
    finally:
        proc.terminate()


def test_disclaimer_printed(tmp_path: Path) -> None:
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        assert "transcription quality is not measured" in out.lower(), out
    finally:
        proc.terminate()


def test_curl_commands_are_printed(tmp_path: Path) -> None:
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        for needle in ("curl", "/docs", "/v1/jobs", "/download"):
            assert needle in out, f"missing {needle!r} in script output"
    finally:
        proc.terminate()


def test_speaker_count_informational(tmp_path: Path) -> None:
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        # Stub returns two distinct speakers -> an INFO line mentioning "2".
        assert "INFO" in out and "speak" in out.lower(), out
        import re
        m = re.search(r"INFO.*?speaker.*?\b2\b", out, re.IGNORECASE | re.DOTALL)
        assert m, f"no informational speaker-count line:\n{out}"
    finally:
        proc.terminate()


def test_default_endpoint_is_local_8000() -> None:
    text = SCRIPT.read_text()
    assert "http://127.0.0.1:8000" in text, "AC4: default endpoint must be 127.0.0.1:8000"


def test_check_a_docs_404_fails(tmp_path: Path) -> None:
    proc, port = _start_stub("docs_404")
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        assert r.returncode != 0, out
        assert "FAIL" in out and "(a)" in out, out
        assert _leftovers(tmp_path) == [], "cleanup must still run on failure"
    finally:
        proc.terminate()


def test_check_b_post_fail_fails(tmp_path: Path) -> None:
    proc, port = _start_stub("post_fail")
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        assert r.returncode != 0, out
        assert "FAIL" in out and "(b)" in out, out
    finally:
        proc.terminate()


def test_check_c_job_failed_fails(tmp_path: Path) -> None:
    proc, port = _start_stub("job_failed")
    try:
        r = _run(tmp_path, port, "--timeout", "30")
        out = r.stdout + r.stderr
        assert r.returncode != 0, out
        assert "FAIL" in out and "(c)" in out, out
    finally:
        proc.terminate()


def test_check_c_timeout_reports_last_status(tmp_path: Path) -> None:
    proc, port = _start_stub("never_done")
    try:
        r = _run(tmp_path, port, "--timeout", "3")
        out = r.stdout + r.stderr
        assert r.returncode != 0, out
        assert "FAIL" in out and "(c)" in out, out
        assert "processing" in out, "timed-out run must report last observed status"
    finally:
        proc.terminate()


# AC3 (amended): a killed run must not leave the downloaded fixture behind.
def test_cleanup_after_kill(tmp_path: Path) -> None:
    proc, port = _start_stub("never_done")
    try:
        env = _env(tmp_path)
        env["SMOKE_TEST_FIXTURE_URL"] = f"http://127.0.0.1:{port}/fixture/marklex1min.wav"
        child = subprocess.Popen(
            ["bash", str(SCRIPT), "--endpoint", f"http://127.0.0.1:{port}",
             "--timeout", "60"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=env, start_new_session=True,
        )
        work = tmp_path / "tmpwork"
        deadline = time.time() + 30
        staged = False
        while time.time() < deadline:
            if work.exists():
                for _root, _dirs, files in os.walk(work):
                    if any(f.endswith(".wav") for f in files):
                        staged = True
                        break
            if staged:
                break
            if child.poll() is not None:
                pytest.fail("script exited before fixture was staged:\n"
                            + child.stdout.read())
            time.sleep(0.2)
        assert staged, "fixture never appeared under TMPDIR"
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        child.wait(timeout=15)
        assert _leftovers(tmp_path) == [], "killed run left fixture files behind"
    finally:
        proc.terminate()


# AC3 (amended): download unavailable -> fall back to the generated tone, and
# say clearly which fixture was used. Works fully offline.
def test_offline_falls_back_to_tone(tmp_path: Path) -> None:
    if not any(os.access(os.path.join(p, "ffmpeg"), os.X_OK) for p in os.environ["PATH"].split(os.pathsep)):
        pytest.skip("ffmpeg not installed")
    proc, port = _start_stub()
    try:
        # Unreachable fixture URL (port 1) -> download fails -> tone fallback.
        r = _run(tmp_path, port, "--timeout", "30",
                 extra_env={"SMOKE_TEST_FIXTURE_URL": "http://127.0.0.1:1/marklex1min.wav"})
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        assert "tone" in out.lower() and "fallback" in out.lower(), \
            f"must state fallback fixture:\n{out}"
    finally:
        proc.terminate()


# AC8 (amended): ffmpeg missing AND download failed -> clear actionable error,
# never a silent pass.
def test_missing_ffmpeg_clear_error(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX PATH sandboxing")
    sandbox = tmp_path / "sandbox-bin"
    sandbox.mkdir()
    for name in ("bash", "sh", "curl", "python3", "mktemp", "rm", "mkdir",
                 "sleep", "grep", "sed", "date", "wc", "head", "cat", "env",
                 "uname", "sort", "tr", "cut"):
        src = subprocess.run(["which", name], capture_output=True, text=True).stdout.strip()
        if src:
            os.symlink(src, sandbox / name)
    assert not (sandbox / "ffmpeg").exists()
    proc, port = _start_stub()
    try:
        env = _env(tmp_path, {"PATH": str(sandbox),
                              "SMOKE_TEST_FIXTURE_URL": "http://127.0.0.1:1/marklex1min.wav"})
        r = subprocess.run(
            [str(sandbox / "bash"), str(SCRIPT),
             "--endpoint", f"http://127.0.0.1:{port}", "--timeout", "30"],
            capture_output=True, text=True, timeout=TIMEOUT, env=env,
        )
        out = r.stdout + r.stderr
        assert r.returncode != 0, out
        assert "ffmpeg" in out, f"error must name ffmpeg:\n{out}"
        # Card AC8: no silent pass/skip even on the error path.
        assert "ALIVENESS AND PLUMBING ONLY" in out, out
    finally:
        proc.terminate()


# ---- RR1 blocker regressions (review round 1, comment 135) ------------------
# Repro of the blocker: the fixture URL answers 200 with a NON-WAV body (stub
# route /fixture/notreal.wav). Pre-fix, those junk bytes stayed staged,
# ffmpeg (without -y) refused to overwrite the existing file yet EXITED 0,
# so `|| die` never fired and the script uploaded the HTML error page under a
# "generated tone" label — wrong provenance and plumbing that could still
# PASS end to end.

def _has_ffmpeg() -> bool:
    return any(os.access(os.path.join(p, "ffmpeg"), os.X_OK)
               for p in os.environ["PATH"].split(os.pathsep))


def _bin_sandbox(tmp_path: Path) -> Path:
    """Minimal PATH without ffmpeg (POSIX). dd/tail are needed by the
    sanity-check and error-hint paths the junk-URL case exercises."""
    sandbox = tmp_path / "sandbox-bin"
    sandbox.mkdir()
    for name in ("bash", "sh", "curl", "python3", "mktemp", "rm", "mkdir",
                 "sleep", "grep", "sed", "date", "wc", "head", "cat", "env",
                 "uname", "sort", "tr", "cut", "dd", "tail"):
        src = subprocess.run(["which", name], capture_output=True, text=True).stdout.strip()
        if src:
            os.symlink(src, sandbox / name)
    assert not (sandbox / "ffmpeg").exists()
    return sandbox


def _last_upload(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/last-upload",
                                timeout=10) as resp:
        return json.loads(resp.read())


def test_junk_200_download_falls_back_to_real_tone(tmp_path: Path) -> None:
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not installed")
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30",
                 extra_env={"SMOKE_TEST_FIXTURE_URL":
                            f"http://127.0.0.1:{port}/fixture/notreal.wav"})
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        # Truthful provenance: curl SUCCEEDED with 200; the cause is the WAV
        # sanity check failing, never a phantom "download failed".
        assert "WAV sanity check" in out, \
            f"must state the real fallback cause:\n{out}"
        assert "download failed" not in out, \
            f"must not claim a download failure when 200 was served:\n{out}"
        assert "generated" in out and "tone" in out and "fallback" in out, out
        # What was ACTUALLY uploaded must be a genuine RIFF/WAVE tone, never
        # the junk bytes the origin served.
        up = _last_upload(port)
        assert up.get("magic") == "RIFF", f"non-WAV bytes reached the API: {up}"
        assert up.get("size", 0) > 44, up
    finally:
        proc.terminate()


def test_junk_200_missing_ffmpeg_exits_2_never_uploads(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX PATH sandboxing")
    sandbox = _bin_sandbox(tmp_path)
    proc, port = _start_stub()
    try:
        env = _env(tmp_path, {"PATH": str(sandbox),
                              "SMOKE_TEST_FIXTURE_URL":
                              f"http://127.0.0.1:{port}/fixture/notreal.wav"})
        r = subprocess.run(
            [str(sandbox / "bash"), str(SCRIPT),
             "--endpoint", f"http://127.0.0.1:{port}", "--timeout", "30"],
            capture_output=True, text=True, timeout=TIMEOUT, env=env,
        )
        out = r.stdout + r.stderr
        assert r.returncode == 2, \
            f"untrusted fixture + no ffmpeg must die (exit 2), got {r.returncode}:\n{out}"
        assert "ffmpeg" in out, f"error must name ffmpeg:\n{out}"
        assert "WAV sanity check" in out, \
            f"die path must state the truth (200 served, not a download failure):\n{out}"
        # The junk file must NEVER have been uploaded.
        up = _last_upload(port)
        assert up == {"uploaded": False}, f"junk upload reached the API: {up}"
        # AC8: no silent pass/skip even on the error path.
        assert "ALIVENESS AND PLUMBING ONLY" in out, out
    finally:
        proc.terminate()


# AC1: nothing in the container-activation path may reference the smoke test.
def test_no_activation_path_reference() -> None:
    banned = [p for p in REPO.iterdir()
              if p.name.startswith(("Dockerfile", "docker-compose"))
              or p.name in ("entrypoint.sh",)]
    for p in banned:
        if p.is_file():
            text = p.read_text(errors="ignore")
            assert "smoke" not in text.lower(), f"{p.name} references the smoke test"
    compose_dir = REPO / "deploy"
    for p in (compose_dir.glob("**/*") if compose_dir.exists() else []):
        if p.is_file() and p.suffix in (".yml", ".yaml", ".sh", ""):
            text = p.read_text(errors="ignore") if p.stat().st_size < 1_000_000 else ""
            assert "smoke-test" not in text, f"{p} references the smoke test"
    for p in REPO.glob("*.yml"):
        assert "smoke-test" not in p.read_text(errors="ignore"), f"{p.name} references the smoke test"


# AC1: the repo stays audio-free.
def test_no_audio_committed() -> None:
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.wav", "*.mp3", "*.m4a",
                        "*.aac", "*.ogg", "*.flac"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"committed audio found:\n{r.stdout}"


# Card: optional --audio FILE points the script at a user file; the generated
# tone / download lifecycle is then skipped entirely.
def test_audio_override_skips_fixture_lifecycle(tmp_path: Path) -> None:
    user_wav = tmp_path / "myvoice.wav"
    user_wav.write_bytes(b"RIFF" + b"\x00" * 100)  # stub never parses it
    proc, port = _start_stub()
    try:
        r = _run(tmp_path, port, "--timeout", "30", "--audio", str(user_wav))
        out = r.stdout + r.stderr
        assert r.returncode == 0, out
        assert "user-supplied" in out, out
        assert "staging fixture: downloading" not in out, \
            "fixture download lifecycle must be skipped with --audio"
    finally:
        proc.terminate()


# AC9: the README documents the script, marked OPTIONAL, separate from activation.
def test_readme_documents_optional() -> None:
    text = (REPO / "README.md").read_text()
    assert "scripts/smoke-test.sh" in text
    assert "OPTIONAL" in text
    section = text[text.index("scripts/smoke-test.sh") - 2000:
                  text.index("scripts/smoke-test.sh") + 6000]
    lowered = section.lower()
    assert "activation" in lowered, "section must state it is not part of activation"
    assert "not" in lowered and ("quality" in lowered or "accuracy" in lowered), \
        "section must state it is not a transcription-quality test"
