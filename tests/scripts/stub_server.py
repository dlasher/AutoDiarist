#!/usr/bin/env python3
"""Throwaway stub of the AutoDiarist HTTP API for scripts/smoke-test.sh tests.

Implements exactly the endpoints the smoke test touches, shaped like the real
api/main.py responses (POST /v1/jobs -> 202 JobResponse with `token`;
GET /v1/jobs/{token} -> JobStatus; GET /v1/jobs/{token}/download?format=md|json),
plus GET /docs (aliveness) and GET /fixture/<name> (a small but valid RIFF/WAVE
payload) so the script's download -> use -> delete fixture lifecycle can be
exercised without reaching the real internet.

Prints `READY <port>` on stdout once listening (port 0 = ephemeral).

--state selects which check the tests want to trip:
  pass        docs 200; POST 202; polls queued, processing, then done;
              downloads valid.  All five checks (a)-(e) should PASS.
  docs_404    GET /docs -> 404.  Check (a) must FAIL.
  post_fail   POST /v1/jobs -> 500.  Check (b) must FAIL.
  job_failed  polls reach terminal 'failed'.  Check (c) must FAIL.
  never_done  polls stay 'processing' forever.  Check (c) must FAIL on the
              script's bounded timeout; also used for kill/cleanup tests.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import sys
import threading
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ~0.5 s of 440 Hz sine, 16 kHz mono s16 — small, but a REAL RIFF/WAVE so the
# smoke test's magic-byte sanity check sees a genuinely downloadable fixture.
def _tiny_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 16000)))
            for i in range(8000)
        ))
    return buf.getvalue()

WAV_BYTES = _tiny_wav_bytes()

MD_REPORT = (
    "# AutoDiarist transcript\n\n"
    "| Field | Value |\n"
    "| --- | --- |\n"
    "| Duration | 0:00:05 |\n"
    "| Speakers | SPEAKER_00, SPEAKER_01 |\n\n"
    "## Transcript\n\n"
    "**[0:00:00 - 0:00:02] SPEAKER_00:** hello\n"
    "**[0:00:02 - 0:00:05] SPEAKER_01:** hi there\n"
)

# Per-poll status sequences for the GET /v1/jobs/{token} endpoint.
STATE_POLLS = {
    "pass": ["queued", "processing", "done"],
    "job_failed": ["queued", "failed"],
}


class Handler(BaseHTTPRequestHandler):
    state = "pass"
    poll_count = 0
    lock = threading.Lock()

    def log_message(self, format, *args):  # noqa: A002 - match base signature
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _next_status(self) -> str:
        with Handler.lock:
            n = Handler.poll_count
            Handler.poll_count += 1
        if Handler.state == "never_done":
            return "processing"
        seq = STATE_POLLS.get(Handler.state, STATE_POLLS["pass"])
        return seq[min(n, len(seq) - 1)]

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        if path == "/docs":
            if Handler.state == "docs_404":
                self._send(404, b"not found", "text/plain")
            else:
                self._send(200, b"<html><body>stub docs</body></html>", "text/html")
            return
        if path.startswith("/fixture/"):
            self._send(200, WAV_BYTES, "audio/wav")
            return
        if path.endswith("/download") and path.startswith("/v1/jobs/"):
            token = path.split("/")[3]
            fmt = parse_qs(url.query).get("format", ["json"])[0]
            if fmt == "md":
                self._send(200, MD_REPORT.encode(), "text/markdown")
            else:
                result = {
                    "token": token,
                    "status": "done",
                    "filename": "fixture.wav",
                    "duration_seconds": 79.35,
                    "segments": [
                        {"speaker": "SPEAKER_00", "t0": 0.0, "t1": 2.0, "text": "hello"},
                        {"speaker": "SPEAKER_01", "t0": 2.0, "t1": 5.0, "text": "hi there"},
                    ],
                }
                self._send(200, json.dumps(result).encode(), "application/json")
            return
        if path.startswith("/v1/jobs/"):
            token = path.split("/")[3]
            status = self._next_status()
            body = json.dumps({"token": token, "status": status, "stage": status,
                               "progress": 0.5}).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b"no route", "text/plain")

    def do_POST(self):
        if self.path.startswith("/v1/jobs"):
            # Drain the multipart body without parsing it — the real API
            # validates the .wav extension and streams to disk; the smoke
            # test only cares about the response shape.
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if Handler.state == "post_fail":
                self._send(500, b'{"detail":"stub refuses"}', "application/json")
                return
            body = json.dumps({
                "token": str(uuid.uuid4()),
                "status": "queued",
                "session": "stub-session",
                "queue_position": 1,
                "estimated_wait_seconds": None,
                "estimated_completion_at": None,
            }).encode()
            self._send(202, body, "application/json")
            return
        self._send(404, b"no route", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--state", default="pass",
                    choices=["pass", "docs_404", "post_fail", "job_failed", "never_done"])
    args = ap.parse_args()
    Handler.state = args.state
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"READY {server.server_address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
