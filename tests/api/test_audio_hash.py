"""Audio content-hash acceptance tests (card t_971ed589, AC A1-A6).

Every job admitted through POST /v1/jobs gets `audio_sha256`: the SHA-256 of
the file's DECODED 16 kHz mono s16 PCM (not the container bytes), streamed
off the event loop at admission exactly like the duration probe, truncated to
32 hex chars — the house fingerprint style of `_hash_ident` (api/main.py).
The value is stored on the persisted job state and surfaced on the 202 body,
the poll response and the session listing. Presentation + bookkeeping ONLY:
nothing reads this hash yet (the later history/cache cards key on it).

Honest container-invariance scope (AC A4): only a LOSSLESS container pair
(.wav vs .flac of the same PCM) hashes identically. A lossy re-encode
(mp3/m4a/ogg) physically CHANGES the decoded samples, so the same recording
transcoded lossy gets a different fingerprint by design — the card's
"same meeting recording re-encoded" guarantee therefore means re-uploads of
the SAME file (always hit), not lossy transcoding. Tests below pin both
halves of that truth.

Harness note (same as cards t_4b40acfd/t_f97d12a8): MOCK_PIPELINE +
QUEUE_WORKER_ENABLED=0 — admission is pure bookkeeping, jobs stay 'queued'
unless a test rewrites status.json. The hash helper shells out to ffmpeg,
so the whole module skips if ffmpeg is absent (the pipeline depends on it
anyway; the A3 degradation path proves its absence never rejects uploads).
"""
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="audio content hash shells out to ffmpeg",
)

# The junk upload fixture: not a decodable file. Admission must still 202
# with audio_sha256 null (AC A3) — same bytes the listing tests post.
JUNK_AUDIO = b"RIFF....WAVEfmt ....data...."

RATE = 16000  # the pipeline's decode rate: the hash domain is 16k mono s16.


def _setup(tmp_path, monkeypatch, depth=50):
    root = tmp_path / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "JOBS_DIR", root)
    monkeypatch.setenv("MOCK_PIPELINE", "1")
    monkeypatch.setenv("QUEUE_WORKER_ENABLED", "0")
    monkeypatch.setenv("QUEUE_POLL_SECONDS", "0.02")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("MAX_QUEUE_DEPTH", str(depth))
    monkeypatch.delenv("MAX_JOBS_PER_SESSION", raising=False)
    monkeypatch.delenv("RETRY_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("MAX_PROCESSING_PER_SESSION", raising=False)
    return root


def _post(client, session, name="a.wav", data=None):
    r = client.post(
        "/v1/jobs",
        headers={"X-Session": session},
        files={"file": (name, data if data is not None else JUNK_AUDIO, "audio/wav")},
    )
    assert r.status_code == 202, r.text
    return r.json()


def _state(root, token):
    return json.loads((root / token / "status.json").read_text())


def _set_state(root, token, **updates):
    st = _state(root, token)
    st.update(updates)
    (root / token / "status.json").write_text(json.dumps(st))
    return st


def _strip_state_key(root, token, key):
    """Simulate a pre-feature state.json: the key is absent, not null."""
    st = _state(root, token)
    st.pop(key, None)
    (root / token / "status.json").write_text(json.dumps(st))
    return st


def _tone_pcm(freq=440.0, secs=0.25):
    """Deterministic 16 kHz mono s16 PCM — the hash domain itself."""
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / RATE)))
        for i in range(int(secs * RATE))
    )


def _wav_bytes(pcm, rate=RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _remux(tmp_path, wav, suffix):
    """Container-pair fixture: ffmpeg-encode the wav into `suffix`."""
    src = tmp_path / f"src{suffix}.in.wav"
    src.write_bytes(wav)
    dst = src.with_suffix(suffix)
    res = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(dst)],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    return dst.read_bytes()


def _golden(pcm):
    """Expected fingerprint for our 16k mono s16 fixture: the decode is an
    identity resample, so the golden is pure-Python sha256 over the frames —
    no ffmpeg in the EXPECTED value (the helper under test is the only thing
    holding a pipe). Truncated to 32 hex, house style (AC A2)."""
    return hashlib.sha256(pcm).hexdigest()[:32]


HEX32 = 32


# --- AC A1/A2: admission stores the truncated decoded-PCM hash -------------

def test_a1_a2_admission_stores_32hex_hash_of_decoded_pcm(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    pcm = _tone_pcm()
    wav = _wav_bytes(pcm)
    with TestClient(main.app) as client:
        body = _post(client, "user-a", "tone.wav", data=wav)
        assert len(body["audio_sha256"]) == HEX32
        assert all(c in "0123456789abcdef" for c in body["audio_sha256"])
        # State is durable: the value lives in status.json (AC A5), and the
        # golden check proves it is the hash of the DECODED PCM, container
        # never touched, truncated to 32 hex (A1+A2).
        assert _state(root, body["token"])["audio_sha256"] == _golden(pcm)
        assert body["audio_sha256"] == _golden(pcm)


# --- AC A4 (honest version): lossless container pair -> same hash ----------

def test_a4_same_decoded_pcm_in_wav_and_flac_containers_same_hash(tmp_path, monkeypatch):
    """Container invariance for LOSSLESS re-muxes: same decoded PCM in .wav
    and .flac must map to ONE fingerprint (the cache key later is content,
    not bytes). Lossy transcoding is covered by the sibling test below."""
    root = _setup(tmp_path, monkeypatch)
    pcm = _tone_pcm()
    wav = _wav_bytes(pcm)
    flac = _remux(tmp_path, wav, ".flac")
    with TestClient(main.app) as client:
        a = _post(client, "user-a", "tone.wav", data=wav)
        b = _post(client, "user-a", "tone.flac", data=flac)
        assert a["token"] != b["token"]
        assert a["audio_sha256"] == b["audio_sha256"] == _golden(pcm)


def test_a4_lossy_reencode_changes_decoded_samples_so_hash_differs(tmp_path, monkeypatch):
    """Documents the physical limit the card pins: an .mp3 re-encode is
    lossy, its decoded samples differ from the original wav's, so its
    fingerprint differs. Re-uploads of the SAME file always hit; that is
    what the later cache needs. A lossy upload still gets a valid hash."""
    root = _setup(tmp_path, monkeypatch)
    pcm = _tone_pcm()
    wav = _wav_bytes(pcm)
    mp3 = _remux(tmp_path, wav, ".mp3")
    with TestClient(main.app) as client:
        a = _post(client, "user-a", "tone.wav", data=wav)
        b = _post(client, "user-a", "tone.mp3", data=mp3)
        assert a["audio_sha256"] == _golden(pcm)
        assert len(b["audio_sha256"]) == HEX32  # lossy file still fingerprints
        assert b["audio_sha256"] != a["audio_sha256"]


# --- AC A1/A2 determinism: different content -> different hash -------------

def test_different_decoded_content_different_hash(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        a = _post(client, "user-a", "440.wav", data=_wav_bytes(_tone_pcm(440.0)))
        b = _post(client, "user-a", "882.wav", data=_wav_bytes(_tone_pcm(882.0)))
        assert a["audio_sha256"] != b["audio_sha256"]
        assert a["audio_sha256"] == _golden(_tone_pcm(440.0))
        assert b["audio_sha256"] == _golden(_tone_pcm(882.0))


def test_a1_resubmission_of_same_file_is_deterministic(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    wav = _wav_bytes(_tone_pcm())
    with TestClient(main.app) as client:
        a = _post(client, "user-a", "tone.wav", data=wav)
        b = _post(client, "user-a", "tone.wav", data=wav)
        assert a["token"] != b["token"]
        assert a["audio_sha256"] == b["audio_sha256"] == _golden(_tone_pcm())
        # The two hashes were computed in two separate thread-pool runs;
        # the helper is pure, so it cannot drift between submissions.
        assert _state(root, b["token"])["audio_sha256"] == a["audio_sha256"]


# --- AC A3: decode failure degrades to null, never rejects -----------------

def test_a3_corrupt_upload_admits_with_null_hash(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        body = _post(client, "user-a", "junk.wav", data=JUNK_AUDIO)
        assert body["status"] == "queued"
        assert body["audio_sha256"] is None
        st = _state(root, body["token"])
        assert st["audio_sha256"] is None
        assert client.get(f"/v1/jobs/{body['token']}").json()["audio_sha256"] is None
        listed = client.get("/v1/jobs", headers={"X-Session": "user-a"}).json()["jobs"][0]
        assert listed["audio_sha256"] is None


def test_a3_helper_returns_none_on_undecodable_file(tmp_path):
    # Direct helper unit: undecodable bytes -> None, never raises.
    p = tmp_path / "junk.wav"
    p.write_bytes(JUNK_AUDIO)
    assert main._hash_decoded_audio(str(p)) is None
    # Missing file -> None as well (same degradation posture).
    assert main._hash_decoded_audio(str(tmp_path / "nope.wav")) is None


def test_a3_helper_returns_none_when_ffmpeg_is_absent(monkeypatch, tmp_path):
    # "ffmpeg is missing" branch: the child cannot even spawn (OSError) ->
    # None; admission must not reject because hashing failed (AC A3).
    p = tmp_path / "tone.wav"
    p.write_bytes(_wav_bytes(_tone_pcm()))
    real_popen = subprocess.Popen

    def boom(*args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], (list, tuple)) and args[0] and args[0][0] == "ffmpeg":
            raise FileNotFoundError(2, "No such file or directory: 'ffmpeg'")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(main.subprocess, "Popen", boom)
    assert main._hash_decoded_audio(str(p)) is None


def test_a3_helper_kills_a_stalled_child(monkeypatch, tmp_path):
    """Watchdog zero-byte-hang branch: a child that never produces a byte
    is killed at the stall cap and the job degrades to a null hash
    (AC A3) — admission is never blocked indefinitely by a hung ffmpeg."""
    class _SilentProc:
        def __init__(self):
            self.out_r, self.out_w = os.pipe()
            self.err_r, self.err_w = os.pipe()
            self.stdout = os.fdopen(self.out_r, "rb")
            self.stderr = os.fdopen(self.err_r, "rb")
            self.returncode = None
            self.killed = False

        def poll(self):
            return self.returncode

        def kill(self):
            if not self.killed:
                self.killed = True
                self.returncode = -9
                os.close(self.out_w)
                os.close(self.err_w)

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(main, "_HASH_DECODE_STALL_SECONDS", 0.2)
    proc = _SilentProc()
    monkeypatch.setattr(main.subprocess, "Popen", lambda *a, **k: proc)
    p = tmp_path / "tone.wav"
    p.write_bytes(_wav_bytes(_tone_pcm()))
    assert main._hash_decoded_audio(str(p)) is None
    assert proc.killed  # the watchdog really killed the hung child


def test_a3_stderr_flood_cannot_deadlock_the_hash_stream(tmp_path):
    """A stderr-chunk flood must never starve the stdout reader: the
    selector drains BOTH child pipes. A big mp3-error flood on a failing
    decode still returns promptly with None (never a hang, never a raise).
    """
    # A file whose decode emits many stderr lines then fails (rc != 0).
    # ffmpeg on a corrupt-but-long file is exactly this shape.
    p = tmp_path / "noisy.wav"
    p.write_bytes(JUNK_AUDIO * 4096)  # ~74 KiB of undecodable junk
    assert main._hash_decoded_audio(str(p)) is None  # returns fast, no hang


# --- AC A5: survives worker state updates (read-modify-write) --------------

def test_a5_hash_survives_worker_state_updates(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    pcm = _tone_pcm()
    with TestClient(main.app) as client:
        body = _post(client, "user-a", "tone.wav", data=_wav_bytes(pcm))
        token = body["token"]
        # Simulate the worker's terminal write (same shape the listing
        # tests use): update-in-place must not drop the admission key.
        _set_state(root, token, status="done", progress=1.0, stage="done")
        assert _state(root, token)["audio_sha256"] == _golden(pcm)
        listed = client.get(
            "/v1/jobs", headers={"X-Session": "user-a"}
        ).json()["jobs"][0]
        assert listed["audio_sha256"] == _golden(pcm)


# --- AC A6: surfaced on the 202 body, the poll and the listing -------------

def test_a6_hash_surfaces_on_202_poll_and_listing(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    pcm = _tone_pcm()
    wav = _wav_bytes(pcm)
    with TestClient(main.app) as client:
        body = _post(client, "user-a", "tone.wav", data=wav)
        token = body["token"]
        assert body["audio_sha256"] == _golden(pcm)

        poll = client.get(f"/v1/jobs/{token}").json()
        assert poll["audio_sha256"] == _golden(pcm)

        r = client.get("/v1/jobs", headers={"X-Session": "user-a"})
        assert r.status_code == 200
        entries = {j["token"]: j for j in r.json()["jobs"]}
        assert entries[token]["audio_sha256"] == _golden(pcm)


def test_a6_pre_feature_state_renders_null_on_poll_and_listing(tmp_path, monkeypatch):
    """A job persisted before this card exists has NO audio_sha256 key in
    status.json. Responses must carry the key as null, not 500 or omit it —
    the shape stays stable for clients (AC A6 'pre-feature state renders
    null')."""
    root = _setup(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        body = _post(client, "user-a", "tone.wav", data=_wav_bytes(_tone_pcm()))
        token = body["token"]
        _strip_state_key(root, token, "audio_sha256")
        _set_state(root, token, status="done", progress=1.0, stage="done")

        poll = client.get(f"/v1/jobs/{token}").json()
        assert poll["audio_sha256"] is None

        listed = client.get(
            "/v1/jobs", headers={"X-Session": "user-a"}
        ).json()["jobs"][0]
        assert listed["audio_sha256"] is None
