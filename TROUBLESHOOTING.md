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
| `DIAR_METHOD='…' not supported` | Accepted values: `foxnose` (default), `ecapa`, `sherpa`, `sortformer`, `none`. |
| `CRISPASR_GPU_BACKEND='…' not supported` | Accepted: `cpu`, `cuda`, `metal`, `vulkan`. |
| `GGUF_MODEL='…' not found` | The baked GGUF path moved, or you set `GGUF_MODEL` to a non-existent file. Set it **empty** to use engine `-m auto --auto-download` into `CRISPASR_CACHE_DIR` (first download needs network). |
| `crispasr exited with code N: …` | Engine-side failure; the tail of stderr/stdout is embedded. Common on the VM: GPU busy, unsupported audio, corrupt WAV. |
| `crispasr exited 0 but wrote no … crispasr_out.json; refusing silent fallback` | Engine produced no artifact; treat as engine bug/version drift. The adapter never substitutes an empty transcript. |
| `crispasr produced no usable segments from N rows` | All rows lacked times or became empty after `<en-US>` tag stripping; inspect with `DIAR_METHOD=none` or another model. |
| `crispasr timed out after 1800s` | Raise `CRISPASR_TIMEOUT_SECONDS` for long files, or check for GPU contention (single worker assumes exclusive card). |

## TranscriptionOOM

`peak VRAM NNNN MiB exceeded cap 3584 MiB` (raised as `TranscriptionOOM`
after the run) means the `nvidia-smi` sampler caught the engine crossing
`VRAM_LIMIT_MIB`; the job is failed rather than thrashing the card — the
matching `VRAM stage=crispasr_diarized_asr … status=EXCEEDED` INFO line
carries the numbers. On a 4GB Turing card the Q4_K model normally peaks
~1085 MiB (39-minute fixture), so hitting the limit usually means
**something else is using the GPU** — check `nvidia-smi` for co-tenants.
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
| `503` + `Retry-After` on POST | Queue at/above `MAX_QUEUE_DEPTH` (default 50) queued jobs. | Retry after the header delay. |
| `429` + `Retry-After` on POST | This **session** already has `MAX_JOBS_PER_SESSION` (default 2) active jobs. | Wait or `DELETE` a queued job. |
| Many clients sharing one quota bucket | Session identity falls back to client IP, then to a shared `anonymous` bucket, when the `SESSION_HEADER` (default `Authorization`) is absent. | Send a per-client bearer token or set `SESSION_HEADER`. |
| Jobs accepted but never start | `QUEUE_WORKER_ENABLED=0` ops valve, or the worker died (`worker crashed processing` in logs). | Flip the valve back to `1`; check job-specific errors. |

## Cancelling and deleting jobs

`DELETE /v1/jobs/{token}` cancels a **queued** job (audio removed
immediately, `status.json` kept so the `cancelled` state stays observable —
and so a straggler worker can never resurrect it). Cancelling a `processing`
job answers `409`: GPU concurrency is 1 and mid-chunk graceful cancel is out
of v1 scope. Finished jobs — `cancelled`/`failed`/`done` — are expired
automatically by the retention sweep (`RETENTION_DAYS`, default 30; runs at
boot and hourly, removals logged with the token). The live pair
`queued`/`processing` is never age-swept; under disk pressure
(`DISK_MIN_FREE_MIB`) queued dirs can still be evicted oldest-first, but a
`processing` job never is (CONFIG.md).

## Restart-recovery behavior

The queue is fully on disk (`JOBS_DIR/<token>/status.json` + monotonic
`seq`), so:

- Jobs that were `queued` stay `queued` across restarts; the first job of a
  fresh process sees the same FIFO order.
- Jobs caught mid-flight in `processing` at crash/restart time are
  **requeued on boot** with their `seq` intact (message: `Requeued after
  service restart`) and are reprocessed from the top — transcription is not
  checkpointed.
- Admission limits (`MAX_QUEUE_DEPTH`, `MAX_JOBS_PER_SESSION`) are
  revalidated from persisted state, so a restart cannot smuggle extra
  concurrency.
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
oversize, depth, session quota) leave no directory on disk at all.
Successful jobs additionally hold `output_result.json` and
`markdown_report.md`; real-mode engine scratch (`crispasr_out.*`) lives only
in an ephemeral temp dir and is gone after the run.

## Docker specifics

- `could not select device driver "nvidia"` — the NVIDIA container toolkit
  is missing on the host, or the compose `deploy.resources` GPU reservation
  is skipped by a non-Swarm runtime; the image itself builds fine.
- First run downloads nothing for the model (GGUF is baked into the image);
  it **does** fetch the foxnose embedder on first diarized run into
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
- `queue_position` is only present in status while `queued` (1-based FIFO
  rank — exactly the worker pickup order, both keyed on `seq`).
- Progress `0.05`/`0.1` claims/decode values are monotonic bands, not time
  estimates; a stuck `processing` job with `progress` 0.4 is most likely
  inside the engine until `CRISPASR_TIMEOUT_SECONDS` fires.
- Uploaded file type is validated by **extension** (`wav mp3 m4a aac ogg
  flac`), not sniffing; a mislabeled file fails later in ffmpeg decode with
  a `failed` job instead of a `415`.
