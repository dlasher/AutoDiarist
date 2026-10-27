# Troubleshooting

## CrispASR backend assertion (the important one)

**Symptom:** every real job fails fast with a status.json error like

```text
crispasr build at '/opt/crispasr/crispasr' registers backends ['cpu'] but GPU backend 'cuda' is required; install libcudart.so.13 and libcublas.so.13 (apt: cuda-cudart-13-0 libcublas-13-0) or include their directory in CRISPASR_LIB_DIR. Continuing on CPU would be an ~11x silent slowdown.
```

**Cause:** the adapter pins `--gpu-backend cuda` and asserts, via
`crispasr --version` (`ggml backends :` line), that the pinned backend is
registered by the binary **before** running a job. The crispasr CLI itself
does not fail on a missing CUDA runtime — it silently drops to CPU (~1.7x
wall on short clips, ~11.6x on production-length files; card t_18ee2e43, exit
0). The assertion converts that silent trap into a typed,
fast, per-job failure. This is working as intended.

**Fix:** ensure `libcudart.so.13` and `libcublas.so.13` are present and
discoverable. In the shipped image they come from apt
`cuda-cudart-13-0` + `libcublas-13-0` and `CRISPASR_LIB_DIR` points at
`/usr/local/cuda-13.0/targets/x86_64-linux/lib` (the adapter prepends it to
the subprocess `LD_LIBRARY_PATH`). Verify manually:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  /opt/crispasr/crispasr --version   # must list cuda in the backends line
```

Debugging escape hatch: `CRISPASR_GPU_BACKEND=cpu` skips the assertion
deliberately — slow but useful for offline testing. Do not ship it.

## Typed configuration errors

| Error (status.json `error` field) | Check |
| --- | --- |
| `CRISPASR_BIN is not set` / `CRISPASR_BIN='…' is not an executable file` | `CRISPASR_BIN` must be an executable path (image default `/opt/crispasr/crispasr`). |
| `DIAR_METHOD='…' not supported` | Accepted values: `sherpa` (default), `ecapa`, `foxnose`, `sortformer`, `none`. |
| `DIAR_METHOD='sherpa' requires the sherpa-onnx models; missing: …` | Startup refusal (card t_5a287979 AC4): the image lacks the ONNX files baked at `/opt/models` (see README "Diarization models"), or the env points somewhere else. The service refuses to start instead of silently emitting one speaker. |
| `crispasr produced no speaker labels for DIAR_METHOD='sherpa'` | Runtime refusal of silent degradation: the engine ran but every segment came back without a speaker label — usually the sherpa subprocess failed (missing `SHERPA_BIN`/libs). Check `--sherpa-bin` wiring and `/opt/sherpa/lib`. |
| `CRISPASR_GPU_BACKEND='…' not supported` | Accepted: `cpu`, `cuda`, `metal`, `vulkan`. |
| `GGUF_MODEL='…' not found` | The baked GGUF path moved, or you set `GGUF_MODEL` to a non-existent file. Set it **empty** to use engine `-m auto --auto-download` into `CRISPASR_CACHE_DIR` (first download needs network). |
| `crispasr exited with code N: …` | Engine-side failure; the tail of stderr/stdout is embedded. Common on the VM: GPU busy, unsupported audio, corrupt WAV. |
| `crispasr exited 0 but wrote no … crispasr_out.json; refusing silent fallback` | Engine produced no artifact; treat as engine bug/version drift. The adapter never substitutes an empty transcript. |
| `crispasr produced no usable segments from N rows` | All rows lacked times or became empty after `<en-US>` tag stripping; inspect with `DIAR_METHOD=none` or another model. |
| `crispasr timed out after 1800s` | Raise `CRISPASR_TIMEOUT_SECONDS` for long files, or check for GPU contention (at `MAX_PROCESSING_GLOBAL>1` the card is deliberately shared; the timeout budget is PER JOB, so concurrent jobs each get the full window). |

## TranscriptionOOM

`peak VRAM NNNN MiB exceeded fleet cap MMMM MiB (per-job LLLL MiB x N
concurrent workers…)` (raised as `TranscriptionOOM` after the run) means the
`nvidia-smi` sampler caught the CARD TOTAL crossing the fleet cap —
`VRAM_LIMIT_MIB` scaled by `MAX_PROCESSING_GLOBAL` (card t_dbad2235: the
sampler reads the whole card, so at N>1 the per-job share is multiplied by
N; at the serial default the math is exactly the old `exceeded cap LLLL
MiB`). The job is failed rather than thrashing the card — the matching
`VRAM stage=crispasr_diarized_asr … status=EXCEEDED` INFO line carries the
numbers. On a 4GB Turing card the Q4_K model normally peaks ~1085 MiB
(39-minute fixture), so exceeding the FLEET cap usually means **something
else is using the GPU** — check `nvidia-smi` for co-tenants.
`VRAM_LIMIT_MIB=0` disables the guard (revokes the OOM protection; not
recommended).

## Logs and VRAM instrumentation

`allocated=` / `peak=` stage lines are INFO-level from the `pipeline.*`
loggers. They reach `docker logs` only because `api/main.py` configures the
root handler before pipeline imports. If you embed the app in another
process and see no such lines, configure logging for the `pipeline` loggers.
On hosts without `nvidia-smi` the sampler logs one INFO line and skips
sampling (no OOM protection).

## Queue admission failures

| HTTP | Meaning | Action |
| --- | --- | --- |
| `503` + `Retry-After` on POST | Queue at/above `MAX_QUEUE_DEPTH` (default 50) queued jobs — the **only** admission rejection since V1.5 (card t_82bfbc15): the detail names both the depth and the Retry-After value. | Retry after the header delay; raise `MAX_QUEUE_DEPTH` if the load is real. A second job from the same session is no longer rejected (the old `429` session quota is gone). |
| Startup log: `MAX_JOBS_PER_SESSION=... is DEPRECATED and IGNORED` | The variable is still accepted so old `.env` files load, but it has no effect (card t_82bfbc15). | Delete the line from your env. Admission is queue-depth only. |
| Many clients sharing one identity bucket | Session identity falls back to client IP, then to a shared `anonymous` bucket, when neither the `SESSION_HEADER` (default `X-Session`) nor the server-issued `autodiarist_session` cookie is present. Bucketing no longer gates admission (V1.5) but still groups jobs per person for listing and scheduling. | Let the issued cookie flow (browser default; `curl -b jar`), send a per-client `X-Session` value, or set `SESSION_HEADER` to a header you already control (e.g. `Authorization`). |
| Jobs accepted but never start | `QUEUE_WORKER_ENABLED=0` ops valve, the global concurrency cap already saturated (`processing_count` on `GET /v1/jobs` equals `MAX_PROCESSING_GLOBAL` — with the default 1 one straggler job holds the only slot), or the worker died (`worker ... crashed processing` in logs). | Flip the valve back to `1`; watch `processing_count` against the startup log line `concurrency: MAX_PROCESSING_GLOBAL=N`; check job-specific errors. Raising `MAX_PROCESSING_GLOBAL` (compose-interpolated, host-overridable) adds real parallel workers on the next boot. |

## Cancelling and deleting jobs

`DELETE /v1/jobs/{token}` cancels a **queued** job (audio removed
immediately, `status.json` kept so the `cancelled` state stays observable —
and so a straggler worker can never resurrect it). A **`processing`** job is
cancellable too (card t_fef54150): the answer is `202 {"status": "cancelling"}`
— `cancel_requested: true` lands in the job state under the queue lock, the
live engine's process group is TERMed then KILLed (the same tested machinery
the timeout path uses), and the worker finalises the terminal `cancelled`
state within seconds. `GET /v1/jobs/{token}` shows `cancel_requested` while
the kill is in flight; the job's `status_journal.jsonl` records
`cancel_request -> cancelled`. A `processing` state whose worker is gone
(e.g. a crash in the kill window) is finalised `cancelled` directly by the
handler, and the boot reclaim finalises — never requeues — any
cancel-requested job. Finished jobs — `cancelled`/`failed`/`done` — are
expired automatically by the retention sweep (`RETENTION_DAYS`, default 30;
runs at boot and hourly, removals logged with the token). The live pair
`queued`/`processing` is never age-swept; under disk pressure
(`DISK_MIN_FREE_MIB`) queued dirs can still be evicted oldest-first, but a
`processing` job never is (CONFIG.md).

## Restart-recovery behavior

The queue is fully on disk (`JOBS_DIR/<token>/status.json` + monotonic
`seq`), so:

- Jobs that were `queued` stay `queued` across restarts; the first job of a
  fresh process sees the same per-session FIFO order, and the round-robin
  pointer reloads from `JOBS_DIR/_scheduler.json`, so fair-share scheduling
  resumes in the same round — a session that was just served does not get
  the next claim ahead of one that has never been served.
- Jobs caught mid-flight in `processing` at crash/restart time are
  **requeued on boot** with their `seq` intact (message: `Requeued after
  service restart`) and are reprocessed from the top — transcription is not
  checkpointed.
- Admission depth (`MAX_QUEUE_DEPTH`) is revalidated from persisted state on
  every POST, so a restart cannot smuggle extra concurrency.
  `MAX_JOBS_PER_SESSION` is deprecated and ignored (V1.5, card t_82bfbc15).
- A restart mid-upload never admits a job: `status.json` is written only
  after the full body is received, and the queue scan ignores directories
  without one (a partially written `audio_source.*` may linger as inert
  garbage in the job dir).
- The backend probe cache is per-process: the first job after a restart
  re-runs `crispasr --version`.

## Failed job post-mortems ("quarantine" directory)

There is no separate quarantine path: the per-job directory under
`JOBS_DIR` (`/data/jobs/<token>/` in the compose deployment) *is* the
post-mortem record. A failed job keeps its `status.json` with the typed
error text, and the worker's stage reached is visible from `progress` +
`message`. Directories age out via the retention sweep
(`RETENTION_DAYS`, default 30) or disk-pressure eviction
(`DISK_MIN_FREE_MIB`), so copy the job dir aside promptly if you want a
post-mortem to survive. Rejected uploads (bad extension,
oversize, queue depth) leave no directory on disk at all.
Successful jobs additionally hold `output_result.json` and
`markdown_report.md`; real-mode engine scratch (`crispasr_out.*`) lives only
in an ephemeral temp dir and is gone after the run.

## Docker specifics

- `could not select device driver "nvidia"` — the NVIDIA container toolkit
  is missing on the host, or the compose `deploy.resources` GPU reservation
  is skipped by a non-Swarm runtime; the image itself builds fine.
- First run downloads nothing for the model (GGUF is baked into the image);
  it **does** fetch the --diarize-embedder auto GGUF embedders on first diarized run into
  `CRISPASR_CACHE_DIR` (`/var/cache/crispasr`, persisted by the
  `crispasr-cache` volume). No network at that moment fails the job with an
  engine-side error — pre-warm once or switch `DIAR_METHOD` deliberately.
- `MOCK_PIPELINE=1` on the compose stack runs the API+queue end-to-end with
  no GPU inference at all — use it to separate pipeline faults from GPU
  faults.

## Transcripts and automation/content filters

Rendered transcripts contain whatever people said: profanity, health data,
credentials read aloud, or other provider-flagged content is possible. When
debugging or triaging jobs **through an LLM agent or any provider-backed
automation, do not paste transcript bodies (or raw `output_result.json`)
into the agent context** — that is how provider content-filter errors appear
mid-debug and how confidential audio leaks into third-party prompts. Pass
only structured failure facts: token, `status.json` fields, error text,
segment counts, timings. Inspect actual transcript text with local tools
(`read_file`, `grep`) in the operator's own session, not in agent context.

## Endpoint-level gotchas

- Tokens are UUID4; a malformed token answers `422`, an unknown one `404`.
- `GET /v1/jobs/{token}/download` before `done` answers `409`
  (`Cannot download results when status is …`). `format=md` gives
  `markdown_report.md`; the default (no `format`) gives
  `output_result.json`; any other value answers `422`.
- `queue_position` is only present in status while `queued` (1-based rank in
  the **fair-scheduled execution order**, card t_4b40acfd — sessions served
  round-robin, FIFO within a session; it is NOT raw admission order and a
  job's reported rank can change as the pointer advances and new sessions
  arrive).
- Progress `0.05`/`0.1` claims/decode values are monotonic bands, not time
  estimates; a stuck `processing` job with `progress` 0.4 is most likely
  inside the engine until `CRISPASR_TIMEOUT_SECONDS` fires.
- Uploaded file type is validated by **extension** (`wav mp3 m4a aac ogg
  flac`), not sniffing; a mislabeled file fails later in ffmpeg decode with
  a `failed` job instead of a `415`.
