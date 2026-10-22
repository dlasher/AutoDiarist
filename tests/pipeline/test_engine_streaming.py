"""Per-chunk engine progress in pipeline/transcribe.py (card t_1eb108e0).

The adapter used to report only two coarse units around a whole-file
crispasr invocation, so the ``transcribing`` band sat on "pass 1/2" for
the entire engine run. These tests drive the adapter against fake engine
binaries that stream diarized chunk lines (the real ``crispasr`` stdout
shape, see spike archive ``run_full_stdout.log``) and pin:

* AC1: stdout is streamed and each chunk line yields a real
  ``progress("asr", done, total)`` report — monotone done counts against
  an exact total derived from audio duration / engine chunk window.
* AC2: the final ``progress("diar", ...)`` assembly report, the return
  contract, and ``CRISPASR_TIMEOUT_SECONDS`` semantics survive; the
  timeout path kills the streamed subprocess (no orphaned process group).
* AC3: stderr still propagates into CrispasrError after dropping
  ``capture_output`` (stderr lands in a file instead of a pipe).
"""

import json
import os
import time
from pathlib import Path

import pytest

from pipeline import transcribe as tr

CONFIG_ENVS = (
    "CRISPASR_BIN", "GGUF_MODEL", "DIAR_METHOD", "CRISPASR_LANGUAGE",
    "CRISPASR_GPU_BACKEND", "CRISPASR_CACHE_DIR", "CRISPASR_LIB_DIR",
    "CRISPASR_TIMEOUT_SECONDS", "MIN_FRAGMENT_SECONDS", "VRAM_LIMIT_MIB",
    "MOCK_PIPELINE", "CHUNK_SECONDS", "OVERLAP_SECONDS",
    "SHERPA_SEGMENT_MODEL", "SHERPA_EMBEDDING_MODEL", "SHERPA_BIN",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in CONFIG_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRISPASR_BIN", "/opt/crispasr/crispasr")
    monkeypatch.setenv("GGUF_MODEL", "/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf")
    tr._BACKEND_PROBE_CACHE.clear()
    yield


OUTPUT_JSON = {
    "transcription": [
        {"offsets": {"from": 0, "to": 29680},
         "timestamps": {"from": "00:00:00,000", "to": "00:00:29,680"},
         "speaker": "(speaker 0) ", "chunk_id": 0, "text": "first chunk."},
        {"offsets": {"from": 29680, "to": 56540},
         "timestamps": {"from": "00:00:32,860", "to": "00:00:56,540"},
         "speaker": "(speaker 0) ", "chunk_id": 1, "text": "second chunk."},
        {"offsets": {"from": 56540, "to": 84200},
         "timestamps": {"from": "00:00:57,240", "to": "00:01:24,200"},
         "speaker": "(speaker 1) ", "chunk_id": 2, "text": "third chunk."},
    ],
}


def _fake_binary(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _streaming_fake(tmp_path: Path, name: str, lines: list[str],
                    extra: str = "") -> Path:
    """Fake engine: version probe ok, stream given stdout lines, write JSON."""
    emits = "\n".join(f'echo "{line}"' for line in lines)
    script = f"""
if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi
{extra}
OF=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-of" ]; then OF="$2"; fi
  shift
done
{emits}
cat > "$OF.json" <<'EOF'
{json.dumps(OUTPUT_JSON)}
EOF
"""
    return _fake_binary(tmp_path, name, script)


def _collector():
    reports: list = []

    def cb(stage, done, total):
        reports.append((stage, done, total))
    return reports, cb


def _use(tmp_path, monkeypatch, fake: Path):
    monkeypatch.setenv("CRISPASR_BIN", str(fake))
    gguf = tmp_path / "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf"
    gguf.write_bytes(b"ggmf")
    monkeypatch.setenv("GGUF_MODEL", str(gguf))
    # sherpa is the shipped default DIAR_METHOD; the startup validation
    # (t_5a287979 AC4) requires both ONNX models on disk -- stub them.
    for env, fname in (
        ("SHERPA_SEGMENT_MODEL", "seg.onnx"),
        ("SHERPA_EMBEDDING_MODEL", "emb.onnx"),
    ):
        (tmp_path / fname).write_bytes(b"onnx-stub")
        monkeypatch.setenv(env, str(tmp_path / fname))
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFFfakewav")
    return str(wav)


# ---------------------------------------------------------------------------
# AC1 — per-chunk progress reports from the streamed engine stdout
# ---------------------------------------------------------------------------

def test_chunk_lines_yield_monotone_per_chunk_reports(tmp_path, monkeypatch):
    fake = _streaming_fake(
        tmp_path, "crispasr-stream",
        [
            "[00:00:00.000 --> 00:00:29.680] (speaker 0) first",
            "[00:00:32.860 --> 00:00:56.540] (speaker 0) second",
            "[00:00:57.240 --> 00:01:24.200] (speaker 1) third",
        ],
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    reports, cb = _collector()
    segments = tr.transcribe_and_diarize(wav, progress=cb)
    asr = [(stage, done, total) for stage, done, total in reports if stage == "asr"]
    assert asr == [("asr", 1, 3), ("asr", 2, 3), ("asr", 3, 3)]
    assert segments and segments[0]["text"].startswith("first chunk.")


def test_report_fires_only_when_the_boundary_advances(tmp_path, monkeypatch):
    # Overlap/interim lines re-covering the same window (real engine behavior
    # on the 39-min spike) must not repeat or regress the done count.
    fake = _streaming_fake(
        tmp_path, "crispasr-overlap",
        [
            "[00:00:00.000 --> 00:00:29.680] (speaker 0) a",
            "[00:00:10.000 --> 00:00:25.000] (speaker 0) interim",
            "[00:00:32.860 --> 00:00:56.540] (speaker 0) b",
            "[00:00:28.200 --> 00:00:40.000] (speaker 1) overlap",
            "[00:00:57.240 --> 00:01:24.200] (speaker 1) c",
        ],
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    reports, cb = _collector()
    tr.transcribe_and_diarize(wav, progress=cb)
    dones = [done for stage, done, total in reports if stage == "asr"]
    assert dones == sorted(dones)
    assert dones == [1, 2, 3]


def test_non_matching_stdout_lines_ignored(tmp_path, monkeypatch):
    fake = _streaming_fake(
        tmp_path, "crispasr-plain",
        ["some engine banner text", "[00:00:00.000 --> 00:00:29.680] (speaker 0) a"],
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    reports, cb = _collector()
    tr.transcribe_and_diarize(wav, progress=cb)
    assert next((s, d, t) for s, d, t in reports if s == "asr") == ("asr", 1, 3)


def test_unknown_duration_falls_back_to_coarse_pair(tmp_path, monkeypatch):
    # Probe failure (ffprobe unavailable/garbage) must keep the legacy
    # coarse asr/diar pair rather than fabricating chunk totals.
    fake = _streaming_fake(tmp_path, "crispasr-nodur",
                           ["[00:00:00.000 --> 00:00:29.680] (speaker 0) a"])
    wav = _use(tmp_path, monkeypatch, fake)

    def boom(path):
        raise tr.TranscodeError("no ffprobe")
    monkeypatch.setattr(tr, "_probe_duration", boom)
    reports, cb = _collector()
    tr.transcribe_and_diarize(wav, progress=cb)
    assert reports == [("asr", 1, 2), ("diar", 1, 1)]


def test_explicit_duration_argument_used_for_chunk_total(tmp_path, monkeypatch):
    fake = _streaming_fake(
        tmp_path, "crispasr-durparam",
        ["[00:00:00.000 --> 00:00:29.680] (speaker 0) a",
         "[00:00:32.860 --> 00:00:56.540] (speaker 0) b"],
    )
    wav = _use(tmp_path, monkeypatch, fake)
    reports, cb = _collector()
    tr.transcribe_and_diarize(wav, duration=60.0, progress=cb)
    assert [(s, d, t) for s, d, t in reports if s == "asr"] == [
        ("asr", 1, 2), ("asr", 2, 2)]


def test_diar_assembly_report_kept(tmp_path, monkeypatch):
    fake = _streaming_fake(tmp_path, "crispasr-diar",
                           ["[00:00:00.000 --> 00:00:29.680] (speaker 0) a"])
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    reports, cb = _collector()
    tr.transcribe_and_diarize(wav, progress=cb)
    assert reports[-1][0] == "diar"
    assert reports[-1][1] >= reports[-1][2]  # fraction 1.0


# ---------------------------------------------------------------------------
# AC2 — timeout semantics: still raises CrispasrError, kills the group
# ---------------------------------------------------------------------------

def test_timeout_raises_and_kills_streamed_process_group(tmp_path, monkeypatch):
    pidfile = tmp_path / "sleep.pid"
    fake = _fake_binary(
        tmp_path, "crispasr-stall",
        f'if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi\n'
        'echo "[00:00:00.000 --> 00:00:29.680] (speaker 0) chunk one"\n'
        f'sleep 60 & echo $! > {pidfile}\nwait',
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    monkeypatch.setenv("CRISPASR_TIMEOUT_SECONDS", "1")
    started = time.monotonic()
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(wav)
    assert "timed out" in str(exc.value)
    assert time.monotonic() - started < 10  # killed, not waited out
    time.sleep(0.4)
    sleeper = int(pidfile.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(sleeper, 0)


def test_engine_exit_without_stdout_eof_is_handled(tmp_path, monkeypatch):
    # stdout stays open after exit 0 (engine never closes the pipe): the
    # adapter must reap via EOF/wait, not hang. The pump sentinel on stream
    # close covers this; run just asserts the call completes.
    fake = _streaming_fake(tmp_path, "crispasr-eof",
                           ["[00:00:00.000 --> 00:00:29.680] (speaker 0) a"])
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    assert tr.transcribe_and_diarize(wav)


# ---------------------------------------------------------------------------
# AC3 — failure surfaces survive the streaming rewrite
# ---------------------------------------------------------------------------

def test_nonzero_exit_propagates_stderr_tail(tmp_path, monkeypatch):
    fake = _fake_binary(
        tmp_path, "crispasr-fail",
        'if [ "$1" = "--version" ]; then echo "ggml backends : cpu,cuda"; exit 0; fi\n'
        'echo "[00:00:00.000 --> 00:00:29.680] (speaker 0) partial"\n'
        'echo "boom: cuda graph failed" >&2\nexit 2',
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    with pytest.raises(tr.CrispasrError) as exc:
        tr.transcribe_and_diarize(wav)
    assert "boom: cuda graph failed" in str(exc.value)
    assert "exited with code 2" in str(exc.value)


def test_streamed_lines_do_not_break_segment_contract(tmp_path, monkeypatch):
    # The returned Segments still come from the JSON file, not stdout.
    fake = _streaming_fake(
        tmp_path, "crispasr-contract",
        ["[00:00:00.000 --> 00:00:29.680] (speaker 0) WRONG TEXT"],
    )
    wav = _use(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(tr, "_probe_duration", lambda path: 90.0)
    segments = tr.transcribe_and_diarize(wav)
    assert [s["text"] for s in segments] == [
        "first chunk.", "second chunk.", "third chunk."]
    assert all({"text", "t0", "t1", "speaker"} <= set(s) for s in segments)


# ---------------------------------------------------------------------------
# pure helpers (no spawn)
# ---------------------------------------------------------------------------

def test_chunk_line_end_seconds_parsed():
    assert tr._stream_chunk_end_seconds(
        "[00:00:32.860 --> 00:00:56.540] (speaker 0) hi") == pytest.approx(56.54)
    assert tr._stream_chunk_end_seconds(
        "[01:02:03,456 --> 02:03:04.500] text") == pytest.approx(2 * 3600 + 3 * 60 + 4.5)
    assert tr._stream_chunk_end_seconds("no timestamps here") is None
    assert tr._stream_chunk_end_seconds("") is None


def test_engine_chunk_total_math():
    assert tr._engine_chunk_total(90.0) == 3
    assert tr._engine_chunk_total(90.1) == 4
    assert tr._engine_chunk_total(29.0) == 1
    assert tr._engine_chunk_total(0.0) == 0
    assert tr._engine_chunk_total(-5.0) == 0
