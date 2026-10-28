# AutoDiarist

Asynchronous HTTP job API that takes WAV/MP3/M4A/AAC/OGG/FLAC uploads and
returns a diarized Markdown transcript. Backend per the spike GO verdict
(card t_7e17589e): the `crispasr` CUDA CLI v0.8.43 running the Nemotron 3.5
ASR streaming 0.6B model as a Q4_K GGUF, with sherpa-onnx global-speaker
diarization by default (card t_5a287979; see "Diarization models"). The retired in-process NeMo/torch path is gone; no
`nemo` or `torch` import remains on the real path.

## Architecture

```text
client
  | POST /v1/jobs (multipart audio; extension whitelist wav/mp3/m4a/aac/ogg/flac)
  v
FastAPI + persistent fair-share job queue on disk
  JOBS_DIR/<token>/status.json   atomic writes, monotonic `seq` = arrival order
  JOBS_DIR/_scheduler.json       round-robin pointer (survives restart)
  JOBS_DIR/<token>/status_journal.jsonl   append-only publish history (stage audit)
  JOBS_DIR/<token>/audio_source.<ext>
  |
  | in-process worker POOL: N = MAX_PROCESSING_GLOBAL (default 1 = the old
  | serial behaviour); atomic claim under a lock, cap enforced at claim time;
  | QUEUE_WORKER_ENABLED=1
  v
worker stages per job (api/main.py process_job_background):
  decode (ffmpeg -> 16 kHz mono WAV; skipped in mock mode)
  transcribe + diarize (pipeline.transcribe -> crispasr subprocess, whole file)
  spell check          (pipeline.spellcheck, ON by default; SPELL_CHECK=0/off
                        to disable; /app/vocab.txt domain vocabulary)
  llm cleanup          (pipeline.llm_cleanup, opt-in: only when an embedding
                        caller injects llm= — never from the HTTP/queue path)
  speaker self-ID      (pipeline.self_identification, weighted evidence)
  render               (pipeline.renderer -> markdown_report.md)
  v
GET    /v1/jobs                  -> list MY jobs (session-scoped; newest first; ?limit=1..200, default 50; + ETA for queued; + processing_count = service-wide in-flight)
GET    /v1/jobs/{token}          -> status (+ audio_sha256 content fingerprint; + queue_position and queue-aware ETA while queued; stage + monotonic progress)
DELETE /v1/jobs/{token}          -> cancel a queued job (409 while processing)

GET    /v1/jobs                  -> list MY jobs (session-scoped; newest first; ?limit=1..200, default 50; + ETA for queued + running-phase ETA for processing; + processing_count = service-wide in-flight)
GET    /v1/jobs/{token}          -> status (+ queue_position and queue-aware ETA while queued; running-phase ETA while processing; stage + monotonic progress; progress_percent)
DELETE /v1/jobs/{token}          -> cancel a queued OR running job (running: 202 + cancelling, then engine kill and a terminal cancelled state)
GET    /v1/jobs/{token}/download -> output_result.json (default) or ?format=md
GET    /v1/history               -> MY transcript history (session-scoped derived view; newest first; ?limit=1..200, default 30; ?format=json|csv export)
```

`MOCK_PIPELINE=1` short-circuits the ASR stage with two deterministic
segments end-to-end; everything else (queueing, limits, cancellation, files)
behaves identically, which is what the API integration tests run against.

## HTTP API

Seven endpoints: six under `/v1/jobs` plus the session history view
`/v1/history`. Base URL in the examples below is the
quickstart default (`http://127.0.0.1:8000`). All error responses use the
FastAPI envelope `{"detail": "..."}` unless noted. Every status code here is
implemented in `api/main.py`.

### POST /v1/jobs — submit a job

Multipart form upload; the file field name is **`file`**. The server accepts
the extensions `wav`, `mp3`, `m4a`, `aac`, `ogg`, `flac` (case-insensitive,
from the uploaded filename). On success:

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
# -> 202 {"token": "6a7f…uuid4…", "status": "queued", "session": "9f2c…uuid4hex…", "queue_position": 1, "estimated_wait_seconds": 1273.5, "estimated_completion_at": "2026-10-21T09:12:33Z", "audio_sha256": "134eb1a10f415c05d55cea2fe9a90b39"}
```

Admission is checked in this order, each with its own failure mode:

| Code | When | Detail |
|---|---|---|
| 415 | extension not whitelisted (checked before the upload touches disk) | `Unsupported file type. Supported: wav, mp3, m4a, aac, ogg, flac` |
| 413 | upload exceeds `MAX_UPLOAD_BYTES` (enforced while streaming, not after) | `File exceeds maximum size of <bytes> bytes` |
| 503 | queue at `MAX_QUEUE_DEPTH` (count of `queued` jobs; default 50) — **the only rejection path** (V1.5): a second job from the same session is queued, never turned away | `Queue is full (n/50 jobs queued). Retry after 30s.` + `Retry-After: <RETRY_AFTER_SECONDS>` header (default 30); the detail names both the depth and the Retry-After value |

A request that passes these checks is always accepted with `202` and
queued: the body carries the `token`, the `session` id, the
`queue_position` (1-based rank in the **fair-scheduled execution order** at
admission — see *Fair-share scheduling* below) and the queue-aware wait
estimate (`estimated_wait_seconds` + `estimated_completion_at`, recomputed
on every poll — see *Wait estimates* below), plus `audio_sha256` — the
audio content fingerprint of the upload (see *Audio content fingerprint*
below; `null` when the file could not be decoded, never a rejection). `MAX_JOBS_PER_SESSION` is
deprecated — if set, the service warns at boot and ignores it; the live
per-session knob is `MAX_PROCESSING_PER_SESSION` (default **1**), which caps
concurrently `processing` jobs per session.

Rejected uploads leave no trace on disk: the partially streamed job directory
is deleted before the error is returned.

### GET /v1/jobs — list my jobs

Session-scoped view of what *you* submitted (V1.5): find your own tokens,
watch progress, and decide what to cancel. Filtered by the same hashed
session identity admission stamps on every job (see *Session identity*) —
other sessions' jobs can never appear, and there is deliberately no
cross-session/admin mode or pagination cursor at this scale.

```bash
curl -sS -H 'X-Session: user-a' http://127.0.0.1:8000/v1/jobs
curl -sS -H 'X-Session: user-a' 'http://127.0.0.1:8000/v1/jobs?limit=10'
```

Answers `200` for every session, including a brand-new one (empty list):

```json
{"jobs": [{"token": "6a7f…uuid4…", "status": "queued", "progress": 0.0,
           "progress_percent": 0,
           "stage": "queued", "source": "meeting.wav",
           "created_at": 1760000000.1, "audio_sha256": "134eb1a10f415c05d55cea2fe9a90b39",
           "queue_position": 1,
           "estimated_wait_seconds": 1273.5,
           "estimated_completion_at": "2026-10-21T09:12:33Z"}],
 "count": 1, "total": 1, "limit": 50}
```

- Every entry carries `token`, `status`, `progress`, `stage`, `source` (the
  uploaded filename), `created_at` and `audio_sha256` (the audio content
  fingerprint, *null* for undecodable uploads and for jobs persisted
  before the field existed — see *Audio content fingerprint*).
  `queued` entries add `queue_position`

- Every entry carries `token`, `status`, `progress`, `progress_percent`
  (integer 0..100 view of the same banded float), `stage`, `source` (the
  uploaded filename) and `created_at`. `queued` entries add `queue_position`
  (the same fair-scheduled rank the poll endpoint reports) plus the
  queue-aware wait estimate `estimated_wait_seconds` /
  `estimated_completion_at` — the SAME live values `GET /v1/jobs/{token}`
  reports (see *Wait estimates*; `null` when no honest estimate exists,
  absent when ETA emission is disabled); `processing` entries add the
  running-phase ETA trio (`estimated_remaining_seconds` +
  `estimated_completion_at`, with `estimated_wait_seconds` aliasing the
  remaining seconds) — the SAME live ratchet values the poll reports (see
  *Running-phase ETA*); `failed` entries
  add `error`. `cancelled` jobs stay listed — terminal states are still part
  of "what did I submit".
- Order: newest-first (`created_at` desc, stable `seq` desc tiebreak).
- `?limit=` defaults to 50 and clamps to `[1, 200]` — out-of-range values
  clamp instead of erroring; a non-integer `limit` is `422`. `total` reports
  the session's job count before truncation so a client can see it was cut.
- Like `POST`, the response echoes the session identity (`Set-Cookie`, plus
  the `X-Session` header when the client doesn't manage its own), so an
  IP-only client can re-list by replaying the minted id.

### GET /v1/jobs/{token} — poll a job

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
```

The response is the job's persisted state. Common to every status: `token`,
`status`, `stage` (see *Progress stages* below), `seq` (monotonic admission
ticket — arrival order), `queued_at`, `progress`, `message`, `filename`,
`duration_seconds` (the audio length probed at admission; `null` when the
probe could not read the file), `audio_sha256` (the audio content
fingerprint — see *Audio content fingerprint* below; `null` for undecodable
uploads and for jobs persisted before the field existed), plus the
bookkeeping fields `session` (hashed session identity), `session_ident`
(hashed session id, see *Session identity*) and `audio_path` (server-local
path) — treat those as opaque.
`status` is one of `queued`, `processing`, `done`, `failed`, `cancelled`:

- `queued` — adds `queue_position`: the job's **1-based rank in the
  fair-scheduled execution order** the worker actually follows (see
  *Fair-share scheduling*), recomputed from the live queue + round-robin
  pointer on every poll; `queue_position: 1` means next. **Semantic change
  in V1.5** (card t_4b40acfd): under strict FIFO this field was the raw
  admission rank, and it never rose while a job stayed queued. With
  fair-share two things change: a job can sit *behind* a later arrival from
  a session that has been waiting longer, and a brand-new arrival from a
  never-served session can jump ahead of it (raising its reported position).
  What is preserved: jobs of the **same session** never overtake each other
  (within-session FIFO by `seq`). It also adds the queue-aware wait
  estimate (see *Wait estimates*): `estimated_wait_seconds` — seconds from
  *now* until this job's result should exist (the running job's remaining
  estimate + everything queued ahead in the fair order + this job's own
  estimate) — and `estimated_completion_at`, the matching UTC wall-clock
  time (`YYYY-MM-DDTHH:MM:SSZ`). Both are recomputed on every poll, so they
  track queue movement instantly; both are `null` when no honest estimate
  exists (this job's duration probe failed at admission), and the fields
  are omitted entirely when `PROCESSING_SECONDS_PER_AUDIO_SECOND=0`
  disables ETA emission.
- `processing` — adds `started_at` and `attempt_started_at` (both stamped
  on every claim, reclaims included — they always mark THIS attempt and
  are equal; `attempt_started_at` is the alias the running-phase estimator
  reads), `stage` moves through
  the pipeline in order while `progress` advances monotonically and
  `message` carries the stage text (e.g. `Claimed by worker`,
  `Transcribing pass 2/4`). It also carries the running-phase ETA (see
  *Running-phase ETA* below): `estimated_remaining_seconds` +
  `estimated_completion_at` derived from the job's own observed progress,
  with `estimated_wait_seconds` aliasing the remaining seconds. All three
  are present but `null` until an honest estimate exists (fresh job, no
  start stamp, or progress frozen past the freshness allowance) — null,
  never stale.
- `done` — `stage: "done"`, `progress: 1.0`, `message: "Transcription complete."`.
  The result itself is fetched from the download endpoint at
  `/v1/jobs/{token}/download` (the poll payload does not carry a URL field).
- `failed` — `stage` stays at the boundary where the work broke, `progress`
  holds its monotonic value **below 1.0** (a failure never looks done), plus
  `error` (the exception message from the worker).
- `cancelled` — `message: "Cancelled by client while queued"` plus
  `cancelled_at`; `stage`/`progress` freeze at their last published values.

#### Wait estimates (ESTIMATES, never guarantees)

A queued client gets an honest number to decide *wait or cancel* from.
Reported on the `POST /v1/jobs` 202 body, on every `GET /v1/jobs/{token}`
poll, and on `queued` entries of the session listing `GET /v1/jobs` — all
three from the same live queue snapshot, recomputed on each request. The
model is deliberately simple and fully deterministic:

```
estimate(job) = duration_seconds(job) x PROCESSING_SECONDS_PER_AUDIO_SECOND
wait(job@rank k) = remaining(currently processing) + sum(estimates of ranks 1..k-1) + estimate(job)
```

- `duration_seconds` comes from a cheap ffprobe container-header read at
  **admission** — never a full decode, and never re-probed when you poll.
  The job's own run refreshes it once with the decoded-wav truth.
- The rank walk uses the **fair-scheduled execution order** (round-robin
  across sessions), i.e. the same order `queue_position` reports, so
  successive positions get distinct, non-overlapping estimates and the
  completion times chain: each position finishes exactly its own estimate
  after the one ahead.
- The currently-processing job contributes only its **remaining** estimate
  (linear estimate minus wall time since it was claimed, clamped at 0), and
  that value only shrinks as the job runs — the ETA never moves because of
  a timer alone, only when queue members genuinely leave.
- Each poll recomputes everything from live state, so admissions,
  cancellations and completions are reflected with no client hint.

What it ignores: diarization variance, chunk overhead, VRAM contention, and
the fixed per-run startup cost — short files land worst (the measured T400
long-file asymptote is 0.185 s/audio-s; a 1-min file adds ~55 s of fixed
cost the linear model does not see). Observed wall time can and will
differ: treat the numbers as a planning estimate, **not a service-level
guarantee**. A job whose own duration probe failed shows `null` — the API
never fabricates a duration for the job you are asking about.

#### Audio content fingerprint

Every job state carries `audio_sha256`: a SHA-256 **content fingerprint**
of the audio, computed at admission and surfaced on the `202` body, every
`GET /v1/jobs/{token}` poll and every session-listing entry (card
t_971ed589 — presentation and bookkeeping only: no cache reads it yet).
The hashed domain is the **decoded audio**, not the uploaded bytes: the
upload is streamed through
`ffmpeg -i file -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -` (the
pipeline's own decode domain — 16 kHz mono s16) into hashlib, so the same
decoded content re-muxed into a different **lossless** container
(`.wav` vs `.flac`) fingerprints identically, while a **lossy** re-encode
(`.mp3`/`.m4a`/`.ogg`) genuinely changes the decoded samples and therefore
hashes differently — re-uploads of the *same file* always match; lossy
transcodes of one recording do not, and cannot. The value is truncated to
**32 hex chars**, the house fingerprint style of the hashed session
identity. The hash runs off the event loop in the same thread pool the
duration probe uses, with a stall/timeout watchdog on the ffmpeg child; a
missing ffmpeg, an undecodable upload or a hung decode yields `null` and
the job is admitted exactly as before — hashing failure never rejects.
`null` also appears for jobs persisted before this field existed. A client
wanting an MD5-style short id should truncate this value client-side; the
server deliberately stores no second digest.

#### Running-phase ETA (V1.5.x, card t_d64bdf29)

For a `processing` job the waiting happens in the run itself, so the poll
(now and the session listing's processing entries) additionally answers
*when will this finish* — derived from the job's **own observed rate**,
never from the fixed linear model above:

```
fraction  = published monotonic `progress` (real work boundaries only)
rate      = fraction / wall seconds since `attempt_started_at`
remaining = (1 - fraction) / rate        -> estimated_remaining_seconds
                                       (+ estimated_completion_at, UTC)
```

`estimated_wait_seconds` aliases `estimated_remaining_seconds` on a
running payload — one question ("when is my result ready?") keeps one
field name across both phases.

Honesty rules, because the first observation of a job is its worst
observation:

* **Emission gates.** Null until the job genuinely crossed
  `RUNNING_ETA_MIN_PROGRESS` (default `0.30` — the transcribing band's
  lower edge, so decode + per-run startup cost is fully behind the
  elapsed-time window instead of masquerading as steady-state rate), and
  — when the duration is known — until it has lived longer than the
  linear model's startup floor for that duration. A fresh job is null,
  never a fabricated value.
* **Never inflates.** The published value ratchets down: a real progress
  advance may tighten the estimate (ratchet min), but a poll can never
  see the number grow, even when the job slows down or the service
  restarted mid-run. On a restart there is no started_at-preservation
  trick: the claim re-stamps both clocks (they are always equal), drops
  the dead attempt's estimate memory, and keeps the preserved `progress`
  value only as a no-regression floor. The honesty comes from the gates:
  while progress sits frozen at that preserved value the freeze guard is
  already engaged (the boot reclaim leaves `last_progress_at` in the
  past) and the startup floor holds emission null — the optimistic
  restored-fraction-over-new-clock reading is never published. An
  estimate resumes only once the new attempt genuinely advances past the
  preserved fraction (pinned by the reclaim regression test).
* **Fresh or null.** Progress freezing is normal (spellcheck /
  llm_cleanup publish at band edges only). A held value survives only
  until `RUNNING_ETA_REANCHOR_SECONDS` (default `120`) without a real
  advance, or until it passes its own promised completion time — then it
  goes null until progress genuinely moves again. A value computed long
  ago that reality has not reconfirmed is worse than no value.
* **Same kill switch.** `PROCESSING_SECONDS_PER_AUDIO_SECOND=0` disables
  running-phase ETA emission too.

The value is still an ESTIMATE: progress bands are ordinal markers of
work, not time (the transcribing band is where real per-chunk boundaries
land, so mid-run estimates sharpen exactly when the engine streams its
`pass n/m` reports); early emissions on long files tend optimistic and
converge as chunks complete. A `cancelling` job (in-flight cancel) or a
terminal one reports no completion estimate — `stage`/`progress` remain,
the ETA fields are simply absent/null. The queued-side model (above) and
every queued job's `estimated_wait_seconds` are untouched by this feature.

Alongside the banded float `progress`, every status payload (poll and
listing, all states) now also carries `progress_percent`: an integer
`0..100` derived from the same published number (`round(progress × 100)`),
so a client can render a bar without inventing its own rounding. It is
band-consistent by construction and monotonic because `progress` is.

#### Progress stages

`stage` is the machine-readable pipeline position — a stable enum, in order
of first appearance:

| Stage | Meaning | Progress band |
|-------|---------|---------------|
| `queued` | waiting in the fair-share queue (briefly also when a worker claims the job before decode starts) | 0.00 |
| `decoding` | audio decode / resample to 16 kHz | 0.05 |
| `transcribing` | ASR passes; engine-reported pass/chunk counts refine within the band (e.g. `Transcribing pass 2/4`) | 0.30–0.60 |
| `diarizing` | diarized assembly, likewise refined per reported boundary | 0.60–0.75 |
| `spellcheck` | ON by default; opt out `SPELL_CHECK=0/off` (domain vocab `/app/vocab.txt`) | 0.75–0.80 |
| `llm_cleanup` | optional; only published when an embedding caller injects the LLM cleanup seam (`llm=`) — the HTTP/queue path never emits it | 0.80–0.85 |
| `self_identification` | speaker-name suggestions | 0.85–0.92 |
| `rendering` | Markdown report + result write-out | 0.92–1.00 |
| `done` | terminal success, `progress == 1.0` | 1.00 |

`progress` semantics: float in `[0, 1]`, **monotonic non-decreasing** across
the whole job, published only at real work boundaries (stage entries and the
engine's per-pass/per-chunk reports) — never from timers. It reaches exactly
**1.0 when and only when `status: "done"`**. Values are ordinal markers of
pipeline position, not percentage-accurate time estimates: a flat stretch
means one long boundary (e.g. a GPU pass), not a stuck job.

A fast stage tail (e.g. spellcheck→done in under a poll interval) can be
missed by a polling client — watch
`JOBS_DIR/<token>/status_journal.jsonl` instead: an append-only record of
every publish (`at`, `status`, `stage`, `progress`, `message`), so the full
ordered stage sequence is always auditable after the run.

Errors: `404` `Job not found` (unknown token), `422` `Invalid token format`
(token must be a canonical UUID4).

### GET /v1/jobs/{token}/download — fetch results

Only for `done` jobs. Default is JSON (`output_result.json` — token, status,
filename, duration, diarized segments); `?format=md` returns the rendered
report (`markdown_report.md`) as `text/markdown`:

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN/download -o result.json
# Content-Type: application/json; attachment; filename="<token>_result.json"
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md" -o report.md
# Content-Type: text/markdown; attachment; filename="<token>_report.md"
```

Errors: `409` `Cannot download results when status is <status>` for any job
that is not `done`; `422` `Unknown format requested. Use format=json or
format=md` for any other `format` value; `404` for unknown tokens (and, as a
consistency guard, `Result file missing` / `Markdown report missing` if a
finished job's artifact is gone from disk).

### GET /v1/history — history & exports (V1.7, card t_a8a71898)

```bash
curl -sS -H 'X-Session: user-a' 'http://127.0.0.1:8000/v1/history'
curl -sS -H 'X-Session: user-a' 'http://127.0.0.1:8000/v1/history?format=csv' -o history.csv
curl -sS -H 'X-Session: user-a' 'http://127.0.0.1:8000/v1/history?limit=100'
```

A session-scoped **derived view** of what *you* submitted — the same
on-disk `status.json` files and the same hashed `session_ident` filter that
back `GET /v1/jobs`. There is deliberately **no separate history store**:
rows disappear exactly when the retention sweep or disk-pressure eviction
removes the job, so history can never become an unbounded disk consumer.
Scoping is a security requirement, not a preference: one tenant's history is
never visible to another, and there is no cross-session/admin mode.

Window and ordering: newest-first (identical clock and tiebreak to the
listing — `created_at` desc, `seq` desc), default `HISTORY_DEFAULT_LIMIT` =
**30** rows; `?limit=` takes precedence and **clamps** into
`[1, MAX_LIST_LIMIT]` (200) — out-of-range values clamp, they never 4xx
(FastAPI `422` only for a non-integer `limit`). The 30 is a code default,
not an env knob.

`?format=json` (default) returns `{"history": [...], "count", "total",
"limit", "columns"}` with `Content-Disposition: attachment;
filename="_session_history.json"`. `?format=csv` returns the **identical
row set** rendered by the stdlib `csv` writer (RFC-4180 escaping — filenames
with commas, quotes or newlines round-trip intact) as `text/csv` with
`Content-Disposition: attachment; filename="_session_history.csv"`. Export
filenames carry no tenant-identifying info; `422` for any other `format`
value.

Pinned column set (identical in CSV and JSON; the header row is always
present, even for an empty history):

| column | meaning |
|---|---|
| `token` | job token — links `GET /v1/jobs/{token}` and the downloads |
| `filename` | the upload's original file name |
| `audio_sha256` | content fingerprint (card t_971ed589); `null` for legacy jobs persisted before the field existed |
| `status` | `queued` / `processing` / `done` / `failed` / `cancelled` |
| `created_at` | job-creation stamp, ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`); legacy fallback `queued_at` |
| `completed_at` | ISO-8601 UTC from the terminal `status.json` file mtime — the **same clock the retention sweep uses**; `null` while the job is not terminal |
| `duration_seconds` | audio length probed at admission; `null` when unknown |
| `download_json` | relative URL `/v1/jobs/<token>/download?format=json` |
| `download_md` | relative URL `/v1/jobs/<token>/download?format=md` — the per-job `markdown_report.md` already answers the .md export ask; history builds no md tables |
| `cache_hit` | reserved for the cache card: always present, `false` until that card stamps it, so the column set never changes |

### DELETE /v1/jobs/{token} — cancel a queued OR running job

```bash
curl -sS -X DELETE http://127.0.0.1:8000/v1/jobs/$TOKEN
# queued    -> 202 {"token": "...", "status": "cancelled"}
# running   -> 202 {"token": "...", "status": "cancelling"}  (then GET shows cancelled)
```

Cancelling a `queued` job frees both queue depth and the session slot; the
source audio is deleted immediately and the state file is kept so `cancelled`
remains observable via GET. Repeat cancels are idempotent (`202`).

A `processing` job is cancellable too (card t_fef54150): the handler marks
`cancel_requested: true` + `cancel_requested_at` in the job state under the
same lock the worker uses for every state write, then signals the live
attempt through a process-local registry. The engine subprocess — already
spawned in its own session, with the tested TERM-then-KILL
`_kill_process_group` on the other end — is terminated immediately, so a
long job stops within seconds instead of finishing its remaining chunks.
The WORKER finalises the terminal `cancelled` state (the handler's `cancelling`
answer is honest: the kill is in flight, one writer per state) and the job's
`status_journal.jsonl` records the `cancel_request -> cancelled` ordering.
`GET /v1/jobs/{token}` and `GET /v1/jobs` expose `cancel_requested` while the
kill is in flight, and the boot reclaim can never requeue a cancel-requested
job onto the GPU — a restart in the kill window lands it `cancelled`.
`done` and `failed` jobs get `409` (`Job already finished with status '<x>'.`).

### Queueing and persistence

The queue lives entirely on disk under `JOBS_DIR`: one directory per token
holding `status.json` (written atomically via temp-file rename) and the
uploaded `audio_source.wav`, plus a single `JOBS_DIR/_scheduler.json` round-robin
pointer. Admission assigns `seq = max(existing seq) + 1`
under a lock, so `seq` is a monotonic arrival ticket — inside one session it
is exactly the execution order (within-session FIFO). Across sessions the
worker does **not** follow raw seq: it serves sessions round-robin
(*Fair-share scheduling* below), and `queue_position` reports ranks in that
fair order. `queued` and
`processing` jobs count against limits; each job's `session` id is persisted
alongside it.

**Restart behavior:** the queue is rebuilt by scanning `JOBS_DIR`, so a
restart (crash or redeploy) does not lose queued work. A job caught
mid-`processing` is requeued on boot with its `seq` intact (message:
`Requeued after service restart`) and re-runs from the start — per-session
FIFO order is preserved across restarts. The round-robin pointer is a file
under `JOBS_DIR` too, so a restarted service continues the same fairness
round instead of letting an established submitter re-jump ahead of a session
that has not been served yet.

#### Fair-share scheduling (V1.5, card t_4b40acfd)

The GPU is a shared resource, so one session submitting many jobs must not
starve another that submitted one. Selection at claim time:

- Each session gets a FIFO sub-queue (ordered by `seq`); the worker serves
  the session **whose turn is longest overdue** — tracked as a global
  `serve_counter` and a per-session `last_served` stamp in
  `_scheduler.json`. Never-served sessions count as overdue-0, so a fresh
  session always gets a turn before an established one is served twice.
  Ties break by earliest sub-queue head (`seq`), then session key — fully
  deterministic, which is what lets `queue_position` predict the worker's
  real pickup order.
- `MAX_PROCESSING_PER_SESSION` (default **1**) caps how many jobs of one
  session may be `processing` concurrently. A session at the cap is skipped
  at claim time even when its jobs sit at the head of the queue; its work
  simply stays queued. The cap is enforced by the **scheduler itself**, not
  by the single-worker setup, so fairness and the "never two in-flight jobs
  of one session" guarantee survive a future multi-worker scale-up. Values
  `< 1` or unparseable warn and clamp to 1 (a zero cap would freeze the
  queue).
- Deferred by the maintainer, deliberately out of scope: duration/size-aware
  priority and preemption of a running job. Fairness applies at selection
  only — a long job holds the GPU to its end, then the overdue session gets
  the next turn.

#### Global concurrency (`MAX_PROCESSING_GLOBAL`, V1.5.1, card t_dbad2235)

The fair-share claim above used to run behind ONE serial-by-construction
worker. `MAX_PROCESSING_GLOBAL=N` starts N workers over the same atomic
claim, and the claim itself enforces a service-wide in-flight cap so the
service never starts a job it cannot run:

- Default **1** — an existing deployment is byte-for-byte unchanged until it
  opts in; opt in per host from MEASUREMENT, not a guess.
- Selection stays the fair round-robin above and the per-session cap keeps
  working exactly as before at any N: 3 workers with 3 jobs of ONE session
  and cap 1 still run them one at a time.
- A hard engine abort under GPU oversubscription (measured on the T400 at
  N=4: `sched alloc encoder graph failed`) fails ONLY that job — the real
  stderr tail lands on its record — and every worker keeps serving.
- Measured NVIDIA T400 operating point: N=3 sustains 3/3 engines at
  3207 MiB of the 4096 MiB card. AMD has since been swept too (measured
  safe at N=4, 12/12 jobs complete) — the full service-level curves for
  BOTH hosts, with the per-host ceilings and causes, are published under
  "Concurrency — measured service-level curve" in "Other backends" below.
  Shipped defaults stay 1; opt in per host from that measurement.
- The VRAM sampler reads the CARD TOTAL, so at N>1 the effective ceiling is
  the FLEET cap `VRAM_LIMIT_MIB × N`; `VRAM_LIMIT_MIB` itself remains the
  documented per-job share (approach (a), planner note on the card).
- Override it from the host: `MAX_PROCESSING_GLOBAL=3 docker compose up -d`
  works because the knob ships interpolated inside both compose
  `environment:` blocks (Docker Compose does not pass arbitrary host env
  vars into containers). The live in-flight count is exposed on
  `GET /v1/jobs` as `processing_count`; the pool size is logged at boot.

Full table and cross-check procedure: `CONFIG.md`.

**Retention:** `RETENTION_DAYS` (default **30**) is live. Finished jobs
(`done`/`failed`/`cancelled`) whose `status.json` mtime is older than the
window are deleted at service boot and hourly after that; every removal logs
the job token and `reason=age-expiry`. The live pair `queued`/`processing` is
never age-swept. Independently, when free space on the filesystem backing
`JOBS_DIR` drops below `DISK_MIN_FREE_MIB` (default 8192), job dirs are
evicted oldest-first until the watermark recovers — a `processing` job is
never evicted — checked at boot and every `DISK_CHECK_INTERVAL_SECONDS`
(default 300). Full table in `CONFIG.md`.

### Session identity

Every client is tracked as a *session*. On first contact the server issues a
random high-entropy session id (uuid4 hex) and returns it **both** as the
`autodiarist_session` cookie (HttpOnly, SameSite=Lax) **and** as the `session`
field of the `POST /v1/jobs` body — so raw API clients can capture the id
without a cookie jar and send it back themselves. The server keeps no session
table: the id lives on the client, and identity is always derived from what
the client presents, in this order (then hashed, sha256 → first 32 hex chars,
so no raw id, token or IP is ever stored on disk):

1. the value of the session header — `X-Session` by default, override with
   `SESSION_HEADER` (e.g. `-H 'X-Session: alice'`; with
   `SESSION_HEADER=Authorization`, `-H 'Authorization: Bearer ***'`
   works as before and gives `alice` a stable identity across IP changes);
2. the server-issued `autodiarist_session` cookie (name overridable via
   `SESSION_COOKIE_NAME`; a browser sends it back automatically, a curl
   client can with `-c jar -b jar`);
3. the client IP when neither header nor cookie is present (`request.client.host`);
4. an anonymous bucket when neither is available.

Each accepted job persists two digests in its state: `session` (the identity
above; since V1.5 it no longer gates admission — `MAX_JOBS_PER_SESSION` is
deprecated and ignored) and the dedicated `session_ident`
key (the hash of the session id itself — including the one issued on the very
first contact — so the scheduler and job listing can group by person rather
than by IP).

A slot frees when the job reaches `done`, `failed`, or `cancelled`. Behind a
reverse proxy that replaces the peer address, all clients share one IP
bucket until a cookie or session header flows — set a real `SESSION_HEADER`
or trust the issued cookie in that deployment.

## Quickstart (GPU host, Docker)

One repo, one Dockerfile, two compose files. **You know your hardware —
pick the compose file that matches it. `docker compose up` is the entire
activation: no file edits, no `.env` creation, nothing else required**
(packaging contract, card t_f82ad22a).

NVIDIA CUDA host (reference GPU: T400 4GB, Turing CC 7.5, validated
`sm_75` kernels; prerequisites: Docker with the NVIDIA container toolkit
and a CUDA-13-era host driver):

```bash
git clone https://github.com/dlasher/AutoDiarist.git autodiarist
cd autodiarist
docker compose up --build
```

AMD/Vulkan host (reference GPU: RX 6600, gfx11; Mesa **RADV** hardware
Vulkan on amdgpu — no ROCm stack needed; prerequisites: loaded `amdgpu`
driver exposing `/dev/dri` + `/dev/kfd`, docker without any GPU runtime
plugin):

```bash
git clone https://github.com/dlasher/AutoDiarist.git autodiarist
cd autodiarist
sudo bash scripts/vulkan-host-setup.sh --persist
docker compose -f docker-compose.amd.yml up --build
```

`vulkan-host-setup.sh` is the ONE host-side step, and only for device
permission: it grants the image's `app` user (uid 10001) rw access to
`/dev/kfd` and `/dev/dri/*` via `setfacl` (primary), or node mode 0666
with a loud note if `setfacl` is missing (fallback), and with
`--persist` installs a udev rule so it survives reboots. Idempotent. The
compose file itself is static — host gids never enter any tracked file.
This replaces `~/CODE/HERMES-PATTERNS/amd-bringup.sh`: that script
GENERATED its own Dockerfile.amd + compose with `DIAR_METHOD=foxnose`
and no sherpa stack, silently regressing the diarization fix — do not
run it anymore (see "AMD/Vulkan bring-up gotchas" below).

In another terminal (see *Test fixture* below — `audio.wav` is not tracked
in this repo):

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
```

Save the token from the response, then poll and download:

```bash
TOKEN=<token-from-response>
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md" -o transcript.md
```

Both compose files pin the same app contract (`DIAR_METHOD=sherpa`, the
baked sherpa-onnx CLI + both ONNX models, `CRISPASR_TIMEOUT_SECONDS=1800`,
retention/disk defaults from `api/main.py`) and the same non-root `app`
user; they differ ONLY in hardware: backend literal (`cuda` | `vulkan`
pinned literals — never env defaults that could silently CPU-fall-back),
the GPU_BACKEND build arg (engine asset + runtime libs layer), device
access (nvidia `deploy.resources` reservation | `/dev/dri` + `/dev/kfd`
devices), and the VRAM ceiling (3584 MiB for the 4 GB T400, 7168 MiB for
the 8 GB RX 6600). State persists in named volumes (`crispasr-cache:/var/cache/crispasr`,
`jobs-data:/data/jobs`); `HOST_PORT` (default 8000) remaps the published port.

## Diarization models (card t_5a287979)

Default `DIAR_METHOD=sherpa`: the engine shells out to the external
**sherpa-onnx** CLI for segmentation + speaker embedding + clustering —
the engine's own `--help` names `sherpa`/`ecapa` as the reliable path.
This replaced the old `foxnose` default after live measurement on the NV
host: foxnose's global pass prints `windows=357 skipped=0 -> 1 speakers
(cosine_similarity_single_speaker)` on the 249 s NASA SEWP panel file —
its BIC counter's cosine single-speaker veto (fixed threshold 0.16 in
the engine's spectral diarizer) collapses 5 real speakers to one label
before any silhouette sweep runs. The embeddings are separable (pinning
`--diarize-num-speakers 5` yields 76 turns across speakers 0..4), so the
collapse is the count estimator, not the audio.

Three vendored artifacts are baked into the image (`Dockerfile`, all
from **k2-fsa/sherpa-onnx GitHub releases** — never guessed, each pinned
by sha256 of the exact asset bytes; GitHub publishes no digests, ours
were computed on the NV host 2026-10-22):

| Artifact | Release asset URL | sha256 |
|---|---|---|
| sherpa-onnx CLI + libs | `github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-linux-x64-shared.tar.bz2` -> `/opt/sherpa/{bin,lib}` | `c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4` |
| segmentation model | `.../download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2` -> `/opt/models/sherpa-onnx-reverb-diarization-v1.onnx` | `615761e980be1688da0ef81618c056134d63aa55ea0a5f1494c47393b9398eab` (tarball), inner `model.onnx` `8249e2e323f8fb0566a387d94b40a73d0b75d54ee60d02dcd729a56a5ba8ecea` |
| embedding model | `.../download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` -> `/opt/models/` (same name) | `1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b` |

The segmentation model is the REVERB-trained variant, not plain
pyannote-segmentation-3-0: NV-host A/B over the card fixtures shows it
roughly halves the single-speaker over-split while the panel file keeps
separating.

`SHERPA_BIN` points the engine at `scripts/sherpa-diarization-wrapper.sh`
(installed `/opt/sherpa/sherpa-diarization-wrapper`): the engine passes
`--clustering.num-clusters` + the model paths but has **no flag** for
`--clustering.cluster-threshold`, which is the knob that decides how
many speakers sherpa's FastClustering emits. Measured via the full
engine contract (reverb-seg + eres2net + threshold):

| threshold | panel file (249 s, 5 real speakers) | BestPractice (137 s, ~1-2) | marklex control (79 s, 2) |
|---|---|---|---|
| 0.35 | **5 speakers** (= reference count) | 3 | 4 |
| 0.38 | 3 | 3 | 4 |
| 0.40 (shipped default) | 3 | 2 | 4 |
| 0.45 | 2 (collapse) | 2 | not measured |

The shipped default 0.40 meets the card's `>= 3` bar on the panel while
keeping single-speaker output no worse than the retired foxnose default;
tune `SHERPA_CLUSTER_THRESHOLD` per deployment (lower -> more speakers).
Honest limits: 0.40 yields 3 speakers on a 5-speaker panel (the exact 5
needs 0.35 at the cost of 3 labels on single-speaker audio — engine-side
limitation, no service-side knob achieves both; filed upstream against
CrispASR), and marklex over-splits 2 -> 4 on the zh-trained embedding.
Separately from threshold tuning, `DIARIZATION_MAX_SPEAKERS` (default 8)
caps the label count on pathological audio — see
`autodiarist-speaker-count-ceiling.md`.

**Diarization threading (cards t_3cb8f97c + t_71a5768f).** The sherpa CLI
defaults to ONE thread for both the segmentation and the embedding
network, and diarization is ~80% of a job's wall time. Threaded per-box
(measured sweeps on both shipped hosts; absolute walls are NOT comparable
across hosts — different clips — so each row normalises to seconds of
diarization wall per second of audio):

AMD host — 12 cores, 249 s panel clip (PLATEAU above the optimum):

| threads | sherpa wall | vs 1 thread | s wall / s audio | speakers |
|---|---|---|---|---|
| 1 | 52.8 s | 1.00x | 0.212 | 5 |
| 2 | 30.8 s | 1.72x | 0.124 | 5 |
| **4** | **19.8 s** | **2.66x** | **0.080** | 5 | ← optimum
| 6 | 21.0 s | 2.52x | 0.084 | 5 |
| 8 | 21.2 s | 2.49x | 0.085 | 5 |
| 12 | 22.2 s | 2.38x | 0.089 | 5 |

NVIDIA T400 host — 4 cores, 137.5 s clip (CLIFF above the optimum):

| threads | sherpa wall | vs 1 thread | s wall / s audio | speakers |
|---|---|---|---|---|
| 1 | 58.3 s | 1.00x | 0.424 | 4 |
| 2 | 32.3 s | 1.81x | 0.235 | 4 |
| **4** | **20.4 s** | **2.85x** | **0.148** | 4 | ← optimum
| 6 | 53.4 s | 1.09x | 0.388 | 4 |
| 8 | 69.3 s | 0.84x | 0.504 | 4 |
| 12 | 110.4 s | 0.53x | 0.803 | 4 |

**4 threads is the measured optimum on BOTH hosts** — the speaker count
was constant at every thread count, so threading is a performance knob,
not a quality knob. The two curve SHAPES differ: on the 4-core NVIDIA box,
6 threads OVERSUBSCRIBE and diarization falls off a cliff (8 threads is
worse than 1); the 12-core AMD box has no oversubscription and stays on a
flat plateau above 4 — the work appears to be MEMORY-BANDWIDTH bound and 4
threads already saturate it. An earlier "optimum == core count" rule
(derived on the 4-core box alone, where it predicts the same 4 as a
constant) was corrected by the AMD sweep: the shipped default cap is now
`SHERPA_MAX_THREADS=4`, not 8.

The wrapper computes
`threads = min(max(1, cores // MAX_PROCESSING_GLOBAL), SHERPA_MAX_THREADS)`
and passes the value as both flags on every sherpa call, ceiling re-run
included. The `cores // workers` divisor stays as the SAFETY rule (it
bounds total demand at the core count: at N=4 on the 4-core box it yields
1 thread per job; a hardcoded 4 would put 16 threads on 4 cores), and the
cap then trims the N=1 default to the measured 4 on both boxes. The
default is never worse than single-thread behaviour. Override per host
with compose (`SHERPA_NUM_THREADS=2 docker compose up -d`) or without
restart (`docker compose exec -e SHERPA_NUM_THREADS=2 autodiarist …`) —
both knobs are in both compose files. On the AMD box: set nothing — the
shipped default is the measured optimum there. Mechanics + both measured
curves: CONFIG.md "Diarization threading".

`DIAR_METHOD=foxnose` remains selectable and is faster; `none` is the
documented escape (segments get one speaker by design). If `sherpa` or
`ecapa` is configured without its models, the service **refuses to
start** (`CrispasrConfigError` names the missing files) — the old silent
degradation to a fake one-speaker transcript is closed at both startup
and runtime.

## Quickstart (no GPU, mock pipeline)

Prerequisites: Python 3.10+ only. `ffmpeg` is required exclusively for real
transcription runs; mock mode never touches it.

```bash
git clone https://github.com/dlasher/AutoDiarist.git autodiarist
cd autodiarist
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
MOCK_PIPELINE=1 uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal (any `.wav` filename validates; mock mode does not read
the bytes):

```bash
curl -sS -F 'file=@audio.wav;type=audio/wav' http://127.0.0.1:8000/v1/jobs
TOKEN=<token-from-response>
curl -sS http://127.0.0.1:8000/v1/jobs/$TOKEN
curl -sS "http://127.0.0.1:8000/v1/jobs/$TOKEN/download?format=md"
```

## Test fixture

`audio.wav` (~74 MB, 39-minute meeting recording) is a **local untracked
test fixture** — `.gitignore` excludes `*.wav` so the repo stays under 5 MB.
Place any WAV file at `./audio.wav` in the repo root to follow the
curl examples above. Mock mode (`MOCK_PIPELINE=1`) never reads the bytes;
real runs need 16 kHz-capable content (any standard WAV is accepted —
`ffmpeg` normalizes it). To generate a stand-in tone locally:

```bash
ffmpeg -f lavfi -i sine=frequency=440:duration=5 -ac 1 -ar 16000 audio.wav
```

## Smoke test (OPTIONAL — not part of activation)

`scripts/smoke-test.sh` is an OPTIONAL, user-invoked probe that validates a
running AutoDiarist deployment is alive and its API round-trip works end to
end. It is NOT a container-activation step — activation stays exactly
`docker compose up` (+ the compose file for your hardware) — and no
Dockerfile, compose file, entrypoint, or healthcheck references it; a fresh
clone never runs it. It is also NOT a transcription-quality or WER test:
**ALIVENESS AND PLUMBING ONLY — transcription quality is not measured.**

```bash
./scripts/smoke-test.sh                              # default http://127.0.0.1:8000
./scripts/smoke-test.sh --endpoint http://<host>:8000 --timeout 1800
./scripts/smoke-test.sh --audio my.wav               # optional user-supplied file
```

Six PASS/FAIL checks; exit 0 only when all six pass, non-zero exits name
the failed check, and every curl command used is printed verbatim so a
reviewer can reproduce from a laptop with no host access:

| Check | Verifies |
| --- | --- |
| (a) | `GET /docs` returns 200 (service is up) |
| (b) | `POST /v1/jobs` fixture upload returns 202 + token |
| (c) | the job reaches terminal status `done` within `--timeout` (bounded polling; last observed status/stage/progress is reported on expiry; a `failed` job fails here) |
| (d) | `GET /v1/jobs/{token}/download?format=md` is non-empty markdown with a `Duration` metadata row and a `## Transcript` section |
| (e) | `GET /v1/jobs/{token}/download?format=json` is valid JSON with `status: done` and a non-empty `segments` array (observed speaker count is printed as INFO — informational, never a gate) |
| (f) | a SECOND upload is DELETEd while `processing`: `202 cancelling`, then terminal `cancelled` with `cancel_requested=true` within 60 s of grace — the kill must never surface as `failed` or `done` (card t_fef54150) |

Fixture lifecycle — no audio is committed and none is required from you:
the primary fixture is pyannote's published `marklex1min.wav`, a real 79 s
two-speaker conversation, downloaded at runtime into a `mktemp -d` temp dir
and uploaded **as published** (never pre-converted — if the pipeline cannot
decode 44.1 kHz stereo, the smoke test surfaces that as a genuine finding).
If the download fails (offline, blocked) — or the server answers 200 with a
body that is not a real WAV — the script discards any downloaded bytes and
falls back to the generated 5 s 16 kHz mono tone from *Test fixture* above,
stating truthfully which case occurred and which fixture was used. The temp
dir — and the wav inside it — is removed
on exit, including on failure, Ctrl-C, or SIGTERM; nothing is cached in any
user-visible location and nothing touches `/data/jobs` or any mounted
volume. `--audio FILE` skips the lifecycle and uses your file as-is.

Requirements: `curl` + `python3` only (ffmpeg is needed solely on the
tone-fallback path; if ffmpeg is missing **and** no usable download exists —
fetch failed or the 200 body was not a real WAV — the script stops with a
clear actionable error, never a silent pass or skip).
The script runs zero docker commands and creates exactly one short job per
invocation, so it is safe to run against the production deploy. Coverage:
`tests/scripts/test_smoke_test.py` exercises it against a stdlib HTTP stub —
no container, GPU, or internet needed.

## Language

`en-US` is fixed for v1 (SPEC 08, "Languages" section): the adapter always
passes `-l en-US` (overridable via `CRISPASR_LANGUAGE` for local per-locale
experimentation) and the rendered metadata table reports `en-US`. No
`language` request parameter is accepted.

### AMD/Vulkan bring-up gotchas (card t_f82ad22a)

The AMD host (10.4.10.55) is a live bring-up target, and the old
operator flow (`~/CODE/HERMES-PATTERNS/amd-bringup.sh`) hit two traps
that the in-repo pack closes structurally. Keep them in mind if you
bring up another AMD box:

1. **Two entrypoints.** The generated `Dockerfile.amd` carried its own
   `/usr/local/bin/entrypoint.sh` CMD while the repo Dockerfile launched
   uvicorn through an inline `CMD ["sh", "-c", ...]` — two competing
   startup definitions, only one of which did Vulkan ICD selection, and
   drift between them was guaranteed. The pack now has exactly ONE
   entrypoint (`scripts/docker-entrypoint.sh`, baked via `ENTRYPOINT`),
   shared by both hardware targets: it exports `VK_ICD_FILENAMES` to the
   single `*radeon*|*amdgpu*` ICD found in `/usr/share/vulkan/icd.d/`
   and execs uvicorn. Selecting ALL ICD jsons there made the engine die
   at init with `vulkan:11` (`VK_ERROR_INITIALIZATION_FAILED`) — the
   restricted glob is the fix, and it no-ops on the CUDA image.
2. **The `video` group is NOT the group you need.** On this host class
   the *card* node `/dev/dri/cardN` and the *render* node
   `/dev/dri/renderD*` are owned by DIFFERENT groups, and which is
   `video` vs `render` varies by driver stack and kernel — so any
   hardcoded gid/group (the old flow's `group_add: [video, render]` +
   `HOST_VIDEO_GID` build args) is wrong on at least one host. Device
   gids are host state, and compose cannot interpolate `group_add` from
   `.env` anyway. The pack's answer: grant **uid 10001 directly**
   (`setfacl -m u:10001:rw`, run by `scripts/vulkan-host-setup.sh`,
   `--persist` installs the udev rule), with mode-0666 as the loud
   fallback; the tracked compose file contains NO host-specific value
   of any kind.

`amd-bringup.sh` must not be used to rebuild this service anymore: its
generated image predates card t_5a287979 (no sherpa stack,
`DIAR_METHOD=foxnose`) and a rebuild through it silently regresses
multi-speaker diarization to one label. Its job is now: `git pull`, then
`docker compose -f docker-compose.amd.yml up --build`.

## Other backends

The image pins the `cuda13` Linux release asset on the CUDA target (the
Dockerfile's `GPU_BACKEND=cuda` default) and the `vulkan` asset on the
AMD target (`GPU_BACKEND=vulkan`, `docker-compose.amd.yml`), but the
engine itself carries a wider build matrix. The authoritative source for
per-asset detail is upstream: CrispASR
[docs/install.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/install.md)
("Prebuilt Linux tarballs") and
[docs/environment-variables.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/environment-variables.md).
**Aside from the CPU execution path (measured below), no further row
under `cuda13` is verified by this project.** The AMD/Vulkan target is
**packaged in-repo (card t_f82ad22a)** — base `ubuntu:24.04` +
`libvulkan1` + `mesa-vulkan-drivers` (Mesa RADV on amdgpu, no ROCm), the
sha-pinned vulkan engine asset, `/dev/dri` + `/dev/kfd` — built on the
GPU path proven live on the AMD host (10.4.10.55) by the old
`amd-bringup.sh` bring-up; run the Quickstart's AMD block on the host for
the zero-edit contract proof. HIP, WASM, and CANN remain unsupported: the
adapter accepts only `cpu`/`cuda`/`metal`/`vulkan` for
`CRISPASR_GPU_BACKEND` (CONFIG.md). **Diarization parity (card t_5a287979,
AC5):** the sherpa-onnx CLI, its shared libraries, and both diarization
ONNX models live in backend-independent Dockerfile layers (sherpa-onnx
clustering runs on CPU onnxruntime — see "Diarization models"), so the
CUDA and Vulkan images carry the SAME model files and the SAME default
`DIAR_METHOD=sherpa` — literally the same instructions, since t_f82ad22a
unified the pack into ONE Dockerfile (asserted by
tests/test_docker_packaging.py). Operator details for the remaining
unverified rows — per-row URLs, prerequisites, selection flags, a
container-asset-swap recipe, and a "convert a row to verified" checklist
— follow in the subsections below the measured-CPU material.

| Engine path (v0.8.43 asset) | Prerequisites | Selection flag | Differences that matter operationally | Verified here? |
| --- | --- | --- | --- | --- |
| `crispasr-linux-x86_64-cuda13.tar.gz` (pinned) | NVIDIA driver + CUDA 13 runtime (`libcudart.so.13`, `libcublas.so.13`; image apt-pins `cuda-cudart-13-0` + `libcublas-13-0`) | image contract: `CRISPASR_GPU_BACKEND=cuda` → `--gpu-backend cuda` | CUDA 13; falls back to CPU when CUDA libs are absent (fallback rule below) | yes — T400, `sm_75` (spike t_7e17589e); CPU execution of this same asset tested separately (row below, measurements after the table) |
| CPU execution of the pinned `cuda13` asset (`libggml-cuda.so` absent → `GGML_BACKEND_DL` fallback) | same asset; fires when the NVIDIA driver / CUDA runtime is missing | automatic once the CUDA module fails to load; `--gpu-backend cpu` alone is *not* CPU-only (fallback rule below) | v0.8.43 on the 60 s spike source; a pure-CPU host (no NVIDIA driver at all) is N/A on this GPU box | yes — ~1.7x slower wall than CUDA on the same 60 s input; results in kanban t_18ee2e43 (`REPORT_CPU.md`) |
| `crispasr-linux-x86_64-cuda.tar.gz` | NVIDIA driver + CUDA 12 runtime (`libcudart.so.12`, CUDA-12 `libcublas`) | `--gpu-backend cuda`; service-side pin unchanged | CUDA 12; same `GGML_BACKEND_DL` silent-CPU fallback as cuda13 | no — NOT verified on this project's hardware |
| `crispasr-linux-x86_64.tar.gz` (plain) | none (CPU-only build); AVX2 baseline — "it fails fast with an explanatory error on older CPUs since #380/#405" (upstream install.md) | plain binary; `--gpu-backend cpu` | CPU-only standalone asset; the asset GitHub-release installers default to | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-cpu-legacy.tar.gz` | none (CPU-only); generic x86-64/SSE2 baseline (upstream install.md table) | `--gpu-backend cpu` | CPU-only; targets pre-AVX2 hosts | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-avx512.tar.gz` | none (CPU-only); AVX-512-capable CPU (upstream install.md table) | `--gpu-backend cpu` | CPU-only; AVX-512 vectorized build | no — NOT verified on this project's hardware (as a separate asset) |
| `crispasr-linux-x86_64-vulkan.tar.gz` | Vulkan-capable GPU plus its vendor ICD/runtime — upstream: the tarball "require[s] their respective runtimes at link time (no CPU fallback)" | **packaged AMD target (t_f82ad22a):** built via `docker-compose.amd.yml` (`GPU_BACKEND=vulkan` + `CRISPASR_GPU_BACKEND: vulkan` pinned literal, `--gpu-backend vulkan` at the engine; `-dev N` picks the device) | statically linked: **no** CPU fallback — fails loudly without the runtime; entrypoint pins the RADV/amdgpu ICD | **yes — measured.** AMD target: Radeon 8060S iGPU (Ryzen AI Max+ 395, gfx1151, Mesa RADV) on host 10.4.10.55; NVIDIA reference: T400 4GB (Turing CC 7.5). Same image, same `DIAR_METHOD=sherpa`, same model bytes — only the engine build and the GPU differ. **AMD is ~2.2x FASTER** end-to-end (see the A/B table below); it also produced *identical* segment and speaker counts to the T400 on the same audio. Built from a fresh clone with `docker compose -f docker-compose.amd.yml up --build` — no hand-edited files, no `.env`. |
| `crispasr-linux-x86_64-hip.tar.gz` | AMD GPU + ROCm runtime (same upstream sentence as Vulkan) | the asset *is* the ROCm build; upstream documents no `--gpu-backend hip` example — and there is **no service-side path**: `CRISPASR_GPU_BACKEND` accepts only `cuda`/`cpu`/`metal`/`vulkan` (CONFIG.md) | statically linked: **no** CPU fallback | no — NOT verified on this project's hardware |
| Metal — `-DGGML_METAL=ON` source build on macOS/Apple Silicon (the v0.8.43 page also carries `crispasr-macos-arm64.tar.gz`; never opened here) | macOS on Apple Silicon (upstream marks `-DGGML_METAL=ON` as the Apple Silicon flag) | `--gpu-backend metal`; service-side `CRISPASR_GPU_BACKEND=metal` is accepted for the pip-install dev path, not the Linux container | there is no Metal release asset for Linux at all — this project's image is Linux | no — NOT verified on this project's hardware |
| WASM — `crispasr-wasm.tar.gz` (prebuilt) or `build-wasm.sh` source build | browser with threading (upstream README: "Multithreaded, runs entirely client-side with COOP/COEP headers"); emsdk toolchain for source builds | n/a for the CLI/service — browser client-side artifact; this service spawns the native CLI | browser-oriented path, not a server deployment | no — NOT verified on this project's hardware |
| CANN — `-DGGML_CANN=ON` source build via the manual workflow | Huawei Ascend NPU + CANN SDK; **no CANN release asset exists in v0.8.43** (checked against the release's asset list) | build with `-DGGML_CANN=ON`, then `--gpu-backend cann -dev N` | see the CANN caveat below: a green `cann-build.yml` run is compile proof only | no — NOT verified on this project's hardware |

### AMD vs NVIDIA — measured A/B (2026-10-22, kanban t_97bbd597)

Two deploys of the **same** image family, the **same** model bytes
(Nemotron 3.5 ASR streaming 0.6B Q4_K GGUF, sha256 `22480dd2…`), the **same**
`DIAR_METHOD=sherpa` and the **same** diarization models. Only the engine build
and the GPU differ. Measured as a pure consumer over the HTTP API — no host
access, no GPU tooling on either side.

| host | GPU | engine build | cores / RAM |
|---|---|---|---|
| NVIDIA (10.4.10.58) | T400 4GB, Turing CC 7.5 (320 GB/s HBM2) | `crispasr-linux-x86_64-cuda13.tar.gz` | 4 / i9-13900H |
| AMD (10.4.10.55) | Radeon 8060S iGPU (Ryzen AI Max+ 395, gfx1151, Mesa RADV) | `crispasr-linux-x86_64-vulkan.tar.gz` | many-core / 131 GB |

| input | NVIDIA wall | NVIDIA RTF | AMD wall | AMD RTF | AMD lead |
|---|---|---|---|---|---|
| `whatIsNASASEWP_2026Update-CC` (249.0 s, 5-speaker panel) | 222.7 s | 1.12x | **105.4 s** | **2.36x** | **2.1x** |
| `BestPractice_RFQ_CC` (137.5 s, single voice) | 78.4 s | 1.75x | **39.2 s** | **3.50x** | **2.0x** |

Both backends returned the same output SHAPE on both files — 18 segments /
3 speakers on the panel, 8 / 2 on the single-voice file — and that first
report said "identical". That word is now FALSIFIED at the text level by a
directly matched-build re-run (see the correction below): the segment and
speaker COUNTS agree everywhere measured, but the transcript TEXT is not
the same — roughly 6% of the words differ.

Measured on the **matched build** (`000bc2f` — speaker-count ceiling, domain
lexicon and long-job progress), with both hosts on the same app layer, on an
**idle queue**, using each job's own elapsed time so queue wait is excluded
(an earlier measurement on a busy host overstated the AMD figure by 3.5x).

These are the job's own elapsed time (`started_at` to terminal status), so they
**exclude queue wait** — which matters: with one GPU worker per host, a
submit-to-done wall time includes however long the job sat behind others, and
on a busy host that was measured at 3.5x the job's real cost. To reproduce,
read the job's `started_at` from the status payload rather than timing the
POST. Earlier measurements taken under load were 3.66x / 3.69x on AMD — the
numbers above supersede them.

The AMD lead comes mostly from the **CPU**, not the GPU: the diarization pass
clusters on one core (`onnxruntime` CPU provider) while the GPU handles ASR, and
the AMD host has 12 cores and 112 GiB against the NVIDIA host's 4 cores and
7.9 GiB.

RTF = audio seconds / wall seconds (higher is faster). Both backends returned
the same output SHAPE — 18 segments / 3 speakers on the panel file and
8 / 2 on the single-speaker file — so the difference is not structural.
Shape agreement is not text agreement, though: on the later matched-build
re-run (b0335f9, below) the two transcript TEXTS differ on ~6% of words
(word similarity 0.9442) — the first pass here missed that. See the
correction subsection below before quoting this table as evidence that the
backends agree.

#### The falsified claim: same COUNTS, different words (2026-10-26, card t_ee479fdf)

Re-measured on the later **matched build `b0335f9`** — same 137.5 s clip
(the single-voice `BestPractice_RFQ_CC`), same app layer, both hosts on the
same commit, idle queue:

| host | backend | text_sha | segments | speakers | chars |
|---|---|---|---|---|---|
| NVIDIA | CUDA | `811e0c9b8cc81ee0` | 8 | 2 | 2213 |
| AMD | Vulkan | `11a24a7b2b728066` | 8 | 2 | 2220 |

Word-level similarity: **0.9442** — the transcripts agree on ~94% of words.
The differing runs:

| NVIDIA (CUDA) | AMD (Vulkan) |
|---|---|
| `Soup? We` | `Soup, we` |
| `holders` | `orders` |
| `fine. Just` | `fine just` |
| `it's` | `is` |
| `to touch a justification` | `attach your justification` |
| `does` / `order` | `went` / `holder` |
| `eighty` | `eight and` |
| `dotnasa` | `dotnanasa` |

Nature of the difference: backend ASR variation between CUDA and Vulkan —
same model bytes, same quantisation, different GPU arithmetic. The segment
count and the speaker count are identical (8 / 2), so this is a WORDING
difference, not a structural one, and not a diarization or spell-check
difference. Raw data and the full wave logs:
`~/CODE/autodiarist-planning/NOTES-2026-10-22.md` (section 69); harness
`ab_hosts.py` beside it.

### Concurrency — measured service-level curve (both hosts, 2026-10-26, card t_ee479fdf)

Service-level sweep on the matched build `b0335f9`: the same 137.5 s clip on
BOTH hosts, 3 waves per rung, **in-flight reached N at every rung** on both
hosts — 30/30 jobs completed on AMD (N=1..4) and 12/12 on NVIDIA (N=1..3).
Queue wait was ~4–7 ms at every rung on both hosts, so the
wall time IS the processing time (no queue contamination).

| N | host | wall / job | throughput | scaling efficiency | per-job degradation |
|---|---|---|---|---|---|
| 1 | AMD (12 cores, UMA, Vulkan) | 41.3 s | 0.024 jobs/s | — | — |
| 2 | AMD | 47.9 s | 0.042 jobs/s | 88% | +16% |
| 3 | AMD | 49.5 s | 0.061 jobs/s | 85% | +20% |
| 4 | AMD | 57.2 s | 0.070 jobs/s | 73% | +38% |
| 1 | NVIDIA (4 cores, 4 GB T400, CUDA) | 79.3 s | 0.013 jobs/s | — | — |
| 2 | NVIDIA | 92.6 s | 0.022 jobs/s | 85% | +17% |
| 3 | NVIDIA | 108.9 s | 0.028 jobs/s | 72% | +37% |

Both curves have the same SHAPE: **throughput climbs with N while per-job
latency rises** — concurrency buys service capacity, not faster single jobs.
Head-to-head at the same build and clip, AMD runs 1.92x / 1.93x / 2.20x
faster than NVIDIA at N=1/2/3 (throughput 1.85x / 1.91x / 2.18x).

Safe worker count per host: AMD N=4 (measured, 12/12 jobs complete), NVIDIA
N=3 (N=4 aborts the fourth engine — see the ceiling below).

Why the two hosts stop in different places:

- **NVIDIA (T400, 4 GB VRAM, 4 cores): safe ceiling N=3.** At N=4 the fourth
  engine dies with the verbatim abort `nemotron: sched alloc encoder graph
  failed` (measured 3/4 in EVERY wave; the other three jobs survive and
  complete, and the service stays serving). It is a VRAM ceiling, not a CPU
  one: the card holds 4096 MiB and the measured per-job peak is ~1069 MiB,
  so three engines fit (3207 MiB observed at N=3) and four do not.
- **AMD (Radeon 8060S iGPU, 12 cores, UMA): no VRAM ceiling** — the "VRAM"
  is system RAM (112 GiB), so at these job sizes the engine never hits an
  allocation wall; the first constraint seen is CPU-side, and the curve is
  still climbing at N=4.

AMD caveat: `VRAM_LIMIT_MIB` is INERT on the AMD image. The vulkan build
ships without `nvidia-smi`, so the guard's sampler self-disables at startup
(`VRAM sampling disabled`) and the check can never fire, at any N. The cap
only has meaning on NVIDIA, where the sampler reads the CARD TOTAL — so at
N>1 the effective ceiling is the fleet cap `VRAM_LIMIT_MIB × N` (see
CONFIG.md).

Threads × workers — the interaction that bites operators: the wrapper
resolves `threads = min(max(1, cores // MAX_PROCESSING_GLOBAL),
SHERPA_MAX_THREADS)` (default cap 4), so raising the worker count silently
lowers every job's diarization threads. On the 4-core NVIDIA box at N=3 each
job gets ONE thread and a solo job takes 78.7 s instead of the 43 s measured
at N=1 with 4 threads. If a host is configured for more workers than its
users need, the per-job latency penalty is real and unavoidable at that N;
`SHERPA_NUM_THREADS` pins the count explicitly when the divisor rule is not
what you want. The per-box thread curves behind those numbers are already
documented in this README ("Diarization threading" above) and in CONFIG.md
(card t_71a5768f — optimum 4 threads on BOTH hosts, NVIDIA cliff vs AMD
plateau); they are deliberately not duplicated here.

Notes for anyone repeating this:

- The sherpa diarization pass clusters on **CPU** (onnxruntime `provider=cpu`),
  so total throughput depends on the host's core count as well as its GPU.
  That is why the AMD lead holds (or widens) once `DIAR_METHOD=sherpa` is the
  default: the GPU is no longer the only constraint.
- `CRISPASR_TIMEOUT_SECONDS` (default 1800) is a hard wall-clock budget for
  the whole job, not an idle timeout. Long inputs need it raised — e.g. a
  140-minute file needs ~83 min on the 4-core T400 and ~38 min on the AMD host,
  so run long jobs with `CRISPASR_TIMEOUT_SECONDS=10800`.
- The AMD target is reached with `docker compose -f docker-compose.amd.yml up
  --build` after `scripts/vulkan-host-setup.sh` (one host-side step granting the
  container's uid 10001 access to `/dev/dri` + `/dev/kfd`).


### Long-form inputs — measured (2026-10-22, kanban t_f01b2b89)

Long inputs behave differently from short ones, in two separate ways.

**Throughput.** A 140-minute input takes about as long as its own duration on
the 4-core NVIDIA host, because the fixed per-invocation costs (the engine's
first diarization call being handed the whole file, model loads, and the
single-threaded diarization pass) amortize over a long file. Short inputs are
setup-dominated and run much faster in relative terms.

| input | host | wall | RTF |
|---|---|---|---|
| SEWP Training Day 1 (140 min) | NVIDIA T400 | 147 min | **0.78–0.95x** |
| SEWP Q&A (118 min) | AMD 8060S | 32 min | **3.73x** |

**Diarization accuracy.** The speaker count depends on the kind of audio, and
the current threshold does not fit every kind:

| input | reference speakers | found | direction |
|---|---|---|---|
| SEWP update (249 s, panel) | 5 | 3 | under-clustered |
| SEWP Training Day 1 (140 min, training) | 10 | 7 | under-clustered |
| SEWP Q&A (118 min, overlapping Q&A) | 6–7 | 8 (100 on the raw auto path) | within the reference band |

The Q&A file is the hardest case — rapid overlapping exchanges with several
people talking over each other. On the **raw auto-clustering path** it fragments
the audio instead of grouping it, reaching 100 labels (most covering only one or
two segments) against a reference of 6–7. The shipped service path yields 8 on
the same file, and `DIARIZATION_MAX_SPEAKERS` (default 8) bounds the raw path
too. The transcript itself does not fragment either way (2% of segments are ≤4
words, against 10% in the reference) — this is a speaker-labelling problem, not
a transcription one.

`SHERPA_CLUSTER_THRESHOLD` is a deployment-level environment variable, so a
lower threshold can be set per deployment without rebuilding the image —
raising it merges clusters, lowering it splits them. No single value is
correct for every recording.

`DIARIZATION_MAX_SPEAKERS` (default **8**, card t_38e52d89) bounds the
explosion independently of the threshold: the wrapper counts the distinct
labels of the auto pass and, **only when they exceed the ceiling**, re-runs
clustering forcing that many clusters — stripping the engine's own
`--clustering.num-clusters=0` first, since a bare appended flag is a silent
no-op (sweep 1). Compliant audio (the panel, promo and marklex fixtures) is
replayed byte-identical: a bare fixed cluster count over-splits cleanly
separated speakers (measured: panel 2 -> 4 at NC=8). `0` disables the
ceiling (DIARIZATION_MAX_SPEAKERS=0). The Q&A file drops from 100 labels
to a measured **7** — inside the 6–7-speaker reference band; finer
accuracy than that needs upstream engine work, not more service knobs.
Full rule, measurements and follow-ups: `autodiarist-speaker-count-ceiling.md`.

Other long-input facts worth knowing:

- `CRISPASR_TIMEOUT_SECONDS` (default 1800) is a **hard wall-clock budget for
  the whole job**, not an idle timeout. Long inputs need it raised — e.g.
  `CRISPASR_TIMEOUT_SECONDS=10800`.
- The progress endpoint advances during a long job on real engine work
  boundaries (card t_2e901945). The adapter runs the engine with `-pp`,
  which writes one unbuffered line per completed ~30 s audio slice to
  stderr (`crispasr: progress =  X% (k/N slices)`); the service tails that
  stream and refines the `transcribing` band per slice — measured 28
  distinct advancing values polling `GET /v1/jobs/{token}` every 10 s
  through a 39-minute job. It is still not a timer: if the engine is
  silent, progress honestly freezes (no synthetic heartbeat exists), and
  a `started_at` check remains the fallback during the engine's silent
  blocking stages (the global diarization pre-pass).


### CPU execution — measured (v0.8.43, kanban t_18ee2e43)

Protocol: the pinned `crispasr-linux-x86_64-cuda13.tar.gz` (sha256
`fea397a3…4fdf54`, `ggml backends: cpu,cuda`) re-extracted to a clean dir on
the T400 VM, `libggml-cuda.so` quarantined out of the extraction (the
documented `GGML_BACKEND_DL` fallback — no CUDA runtime can load), and the
60 s spike source (`excerpt60.wav`, sha256 `b5559cf7…afd46ce`, 1,920,078 B)
piped via ffmpeg exactly as the GPU runs. No GPU flags were passed.
Results on 4 vCPUs: exit 0; ASR-only wall 13.7 s median (3 repeats,
±0.02 s; first cold-cache run ~20 s) vs 8 s on CUDA for the same input —
**~1.7x slower wall** (engine-reported 13.67 s vs 6.16 s, ~2.2x); diarized
wall 24.4 s vs 13 s on CUDA; peak RSS ~937 MiB ASR-only, ~1,065 MiB
diarized. Transcript parity vs the CUDA runs on the same clip: 3/3 sections,
identical speaker labels, 97.7% word agreement, ±400 ms segment-boundary
drift. Guards: `nvidia-smi` showed 0 compute apps and 0 MiB used throughout;
`audio.wav` and the spike/VRAM artifacts verified byte-identical pre/post.
**Caveat found:** `--gpu-backend cpu` alone is *not* "no CUDA runtime
loaded" — the plugin still dlopens and `ggml_cuda_init` still enumerates the
GPU before compute falls to CPU; true CPU-only execution requires the runtime
or the backend `.so` to be absent (or a CPU-only build asset).

Fallback rule (per upstream install.md): the Linux `-cuda` and `-cuda13`
tarballs use dynamic backend loading (`GGML_BACKEND_DL`) since v0.8.30 —
if the NVIDIA driver or CUDA runtime is missing, the CUDA backend simply
does not load and the process silently runs on CPU and exits 0. That is
exactly the failure mode the adapter guards against by pinning
`--gpu-backend cuda` and probing the build's registered backends before
the first job (see the silent-CPU-fallback note in *Replication notes*
and TROUBLESHOOTING.md). The `-vulkan` and `-hip` tarballs are statically
linked against their respective runtimes and do **not** fall back to CPU:
without the matching driver they fail loudly instead of quietly
transcribing slowly. Upstream keeps those two release legs static because
flipping them "needs an A/B on real AMD and Vulkan hardware first"
(upstream install.md; quoted — that A/B has not happened upstream).

CANN is the mirror-image caveat: the manual `cann-build.yml` workflow
compiles CLI, shared library, diff and quantizer with the official CANN
SDK images on x86_64/910B (ACL graphs enabled) and arm64/310P (ACL graphs
disabled), but GitHub hosted runners carry no Ascend NPU — a green run is
compile proof, not inference or performance proof. Treat CANN as
unvalidated upstream as well as here.

To check what a given prebuilt asset actually requires before deploying
it:

```bash
readelf -d crispasr | grep NEEDED
```

Engine environment variables follow upstream's
`CRISPASR_<BACKEND>_<FEATURE>` convention (per-backend switches such as
`CRISPASR_OMNIVOICE_CODEC_GPU`; process-global variables drop the backend
segment — `docs/environment-variables.md` is the canonical list). The
adapter's own `CRISPASR_*` names share that prefix but are this service's
settings, documented in CONFIG.md; the adapter hands values to the engine
as explicit CLI flags (e.g. `CRISPASR_CACHE_DIR` -> `--cache-dir`).

### ggml's auto-selection order — and why this service pins the backend

An engine binary that was compiled with several GPU backends picks one
itself at startup. Upstream states the order verbatim (install.md,
"GPU backends"): ggml "will pick the highest-priority compiled backend at
runtime, in ggml's registration order (CUDA > Metal > SYCL > Vulkan > CPU —
MUSA registers through the CUDA slot, so it shares CUDA's position)". This
project never relies on that order: the adapter always passes
`--gpu-backend $CRISPASR_GPU_BACKEND` (a pinned literal per hardware —
`cuda` in `docker-compose.yml`, `vulkan` in `docker-compose.amd.yml`; the
image ENV default follows the `GPU_BACKEND` build ARG, never a silent
fallback — see `pipeline/transcribe.py`), and asserts before the
first job of each process that the pinned backend appears in
`crispasr --version`'s registered-backends list — so there is no silent
fallback surprise: a build or host that cannot run the pinned backend
fails fast at the probe instead of quietly computing on something else,
and no preference-order pick can select a backend other than the one
declared.
The adapter additionally refuses any `CRISPASR_GPU_BACKEND` outside
`cuda`/`cpu`/`metal`/`vulkan` (CONFIG.md), so a pinned backend is either
explicitly one of those or the service fails fast with a typed
`CrispasrConfigError` at startup. The order above only matters if you run
the raw engine CLI without this service.

### Operator how-to for the unverified rows (v0.8.43 assets)

Every bullet below is upstream-documented or an observation about our own
adapter/container — NOT verified on this project's hardware. Nothing here
is a claim that any of these paths run correctly; that is exactly what the
recipe further down establishes for one backend at a time. Exact asset
names and CPU-feature baselines are quoted from upstream install.md's
prebuilt-tarball table; release-asset existence was checked against the
v0.8.43 release listing (names only — no tarballs were downloaded here,
per the docs-only decision).

- **Vulkan** — asset
  [`crispasr-linux-x86_64-vulkan.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-vulkan.tar.gz)
  — **packaged as the AMD target (card t_f82ad22a)**; build it with
  `docker compose -f docker-compose.amd.yml up --build` (Quickstart), not
  by hand. Prerequisite: a Vulkan-capable GPU with its vendor ICD/runtime —
  upstream: "The HIP and Vulkan tarballs still require their respective
  runtimes at link time (no CPU fallback)." Our reference host pairs it
  with Mesa RADV (`mesa-vulkan-drivers`) on amdgpu, so no ROCm stack is
  needed; the image entrypoint exports `VK_ICD_FILENAMES` to exactly one
  `*radeon*|*amdgpu*` ICD. Select with
  `crispasr --gpu-backend vulkan -dev N -m <gguf> -f <wav>` (`-dev N` picks
  the device on hybrid-GPU hosts); service-side, `CRISPASR_GPU_BACKEND=vulkan`
  is a pinned literal in the AMD compose file. Difference that matters:
  statically linked — without the Vulkan runtime the binary fails loudly
  at load instead of degrading to CPU.
- **HIP / ROCm** — asset
  [`crispasr-linux-x86_64-hip.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-hip.tar.gz).
  Prerequisite: AMD GPU + ROCm runtime (same upstream sentence as Vulkan).
  The asset itself is the ROCm build; upstream install.md documents no
  `--gpu-backend hip` example. Service-side there is **no supported path
  yet**: `CRISPASR_GPU_BACKEND` accepts only `cuda`/`cpu`/`metal`/`vulkan`
  (our adapter's whitelist, CONFIG.md), so end-to-end deployment through
  this service needs an adapter change first — a limitation of this repo,
  not of the engine. Difference that matters: statically linked, no CPU
  fallback.
- **CUDA 12** — asset
  [`crispasr-linux-x86_64-cuda.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cuda.tar.gz).
  Prerequisite: NVIDIA driver + CUDA 12 runtime (`libcudart.so.12` and
  CUDA-12 `libcublas`; the image's apt-pinned `-13-0` packages do not
  satisfy it). Select with `--gpu-backend cuda` — the service-side pin is
  already `cuda`. Difference that matters: same `GGML_BACKEND_DL`
  silent-CPU-fallback behavior as the pinned cuda13 asset (upstream): with
  the CUDA 12 runtime absent it "simply runs on CPU" quietly — the same
  trap, same guard (see the fallback rule above).
- **CPU plain / legacy / AVX-512** — assets
  [`crispasr-linux-x86_64.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64.tar.gz),
  [`crispasr-linux-x86_64-cpu-legacy.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-cpu-legacy.tar.gz),
  [`crispasr-linux-x86_64-avx512.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-linux-x86_64-avx512.tar.gz).
  Prerequisites: none beyond a CPU of the right feature baseline — plain is
  an "AVX2 CPU build" that "fails fast with an explanatory error on older
  CPUs since #380/#405", `-cpu-legacy` targets "generic x86-64/SSE2",
  `-avx512` targets AVX-512-capable hosts (all three quotes: upstream
  install.md). Select: run the binary directly, optionally
  `--gpu-backend cpu`. Difference that matters: there is no GPU module to
  miss-load, so the silent-CPU-fallback trap of the `-cuda` assets does not
  apply; throughput differs per ISA. These exact tarballs were never
  downloaded on this host — the CPU *code path* itself is what t_18ee2e43
  measured (rows above), not these assets.
- **Metal** — source build `-DGGML_METAL=ON` on macOS/Apple Silicon
  (upstream install.md flags Metal as the Apple Silicon target; the default
  source build recipe in install.md covers macOS). The v0.8.43 page also
  carries `crispasr-macos-arm64.tar.gz` / `crispasr-macos.tar.gz`, but
  their contents were never opened here, so their Metal status is not
  established by this README. Selection: `--gpu-backend metal`;
  service-side `CRISPASR_GPU_BACKEND=metal` is an accepted pin value, but
  note the difference that matters operationally: **there is no Metal
  release asset for Linux at all** — the containerized path stays
  `cuda`/`cpu`, and Metal is only reachable outside this image (a macOS dev
  machine running the pip-installed service). NOT verified on this
  project's hardware — we own no Apple Silicon.
- **WASM** — prebuilt asset
  [`crispasr-wasm.tar.gz`](https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/crispasr-wasm.tar.gz)
  or a source build via `build-wasm.sh`. Upstream README: "All backends
  compile to WebAssembly (4.3 MB) via build-wasm.sh. Multithreaded, runs
  entirely client-side with COOP/COEP headers." Prerequisites: a browser
  that serves the demo page with COOP/COEP headers for threading; emsdk for
  the source build. Selection: none — this is a browser client-side
  artifact; the service spawns the native CLI, so there is no service path
  to select. Difference that matters: WASM is browser-oriented, not a
  server deployment option for this stack.
- **CANN** — no release asset: the v0.8.43 asset list contains no CANN
  build (checked against the release listing; upstream keeps it a manual
  workflow precisely because "Hosted runners have no Ascend NPU, so a
  successful build does not establish inference or performance" — install.md).
  Build path: `cmake -DGGML_CANN=ON` (or the manual `cann-build.yml`
  workflow: x86_64/910B with ACL graphs enabled, arm64/310P with them
  disabled — upstream install.md, restated in the CANN caveat above).
  Prerequisite: Huawei Ascend NPU + CANN SDK. Select at run time:
  `crispasr --gpu-backend cann -dev N`. Difference that matters: this
  service cannot pin it (`CRISPASR_GPU_BACKEND` rejects `cann`), so the
  engine is used standalone; a green workflow run proves compilation only.

### Pointing the service at a different engine asset

The adapter does not hard-code engine locations. The variable names below
are the real contract — cross-checked against CONFIG.md,
`pipeline/transcribe.py`, `.env.example`, and `docker-compose.yml` at HEAD
(`1c003a8`):

- `CRISPASR_BIN` — path to the crispasr CLI the adapter spawns (image
  default `/opt/crispasr/crispasr`; unset falls back to a bare `crispasr`
  on `PATH`). The adapter probes this exact file with `--version` and
  fails fast when the pinned backend is not in its registered-backends
  list.
- `GGUF_MODEL` — path to the canonical Nemotron Q4_K GGUF (image default
  `/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf`). Set it **empty**
  to switch the adapter to `-m auto --auto-download` instead.
- `CRISPASR_GPU_BACKEND` — the pin described above.
- `CRISPASR_LIB_DIR` — directory prepended to the subprocess
  `LD_LIBRARY_PATH` (image default: the apt CUDA-13 lib dir; unset it when
  your replacement asset carries its own runtime libs).

Baked into the image are the engine at `/opt/crispasr` and the model at
`/opt/models`; to run a different prebuilt asset against the same service
without rebuilding, mount over those paths. For AMD/Vulkan this recipe is
SUPERSEDED by the in-repo target — `docker compose -f
docker-compose.amd.yml up --build` builds the vulkan asset into the same
image shape with the sherpa stack intact; the manual override below is for
the other experimental rows (cuda12, plain CPU):

```bash
# host: extract the tarball into ./vulkan-engine/ (upstream layout: a
# single root dir inside — strip it like the Dockerfile does), and place
# the canonical Q4_K GGUF (or your own) in ./models/
docker run --rm \
  -e CRISPASR_BIN=/opt/crispasr/crispasr \
  -e GGUF_MODEL=/opt/models/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf \
  -e CRISPASR_GPU_BACKEND=vulkan \
  -e CRISPASR_LIB_DIR= \
  -v "$PWD/vulkan-engine:/opt/crispasr:ro" \
  -v "$PWD/models:/opt/models:ro" \
  autodiarist:local
```

Container-side differences to expect (our own observation of this repo's
files, not upstream): the shipped compose file reserves NVIDIA devices
(`driver: nvidia` in `docker-compose.yml`) — a Vulkan/AMD host needs that
`deploy.resources` block overridden and the vendor ICD/ROCm runtime
visible inside the container; the image's apt CUDA-13 packages are inert
for non-CUDA assets but still present. Expect to validate the swap
yourself with the recipe below before trusting it in production.

### Converting a row from NOT-verified to verified

How to verify a backend yourself — run this for the one you own hardware
for, capture everything, and record the result; do not mark a row `yes`
from inference or upstream promises:

1. Download the exact asset from its URL above and record its digest:
   `curl -LO https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.43/<asset>.tar.gz`,
   `sha256sum <asset>.tar.gz`, extract with `--strip-components=1`.
2. Confirm what it links against before running:
   `readelf -d crispasr | grep NEEDED` (see the readelf tip above), and
   `crispasr --version` — capture the full `Registered backends` list.
3. Run inference directly on a real audio file (any WAV the ffmpeg decode
   step accepts — this project uses the 60 s excerpt of its meeting
   fixture), pinning the backend explicitly:
   `crispasr -f <wav> -m <gguf> -l en-US --gpu-backend <backend> -dev 0 -oj`.
4. Capture: wall time, the process's device utilization during the run
   (`nvidia-smi dmon`, `rocm-smi`, `vkcube --gpu N` / `vainfo`, as
   applicable), and the transcript output file.
5. Then exercise it through the service: start the container with the
   `CRISPASR_BIN`/`GGUF_MODEL`/`CRISPASR_GPU_BACKEND` overrides shown
   above, POST one real job, and download the result as both JSON and MD.
   Success requires a non-empty transcript AND the pinned backend
   appearing in the device-utilization capture (the same evidence standard
   t_18ee2e43 used for the CPU row).
6. Record the outcome two places, in the same PR: create a kanban card on
   the `nvidia-asr` board with the captures (model the protocol on the
   "CPU execution — measured" subsection above) **and** add a CHANGELOG.md
   line. Then update the table row: `no — NOT verified on this project's
   hardware` becomes `yes — <hardware>, <date> (kanban <card id>)`, and
   delete that row's entry from the how-to list.

## Replication notes

- Engine: crispasr prebuilt `v0.8.43`, `cuda13` x86-64 release asset
  (archive root dir stripped with `--strip-components=1`), extracted to
  `/opt/crispasr`. The download is validated by `sha256sum -c` against the
  GitHub release asset digest at build time; the `crispasr --version`
  backend assertion runs at job time (see PIPELINE.md), not at build.
- Model: canonical Q4_K GGUF `nemotron-3.5-asr-streaming-0.6b-q4_k.gguf`
  (~408.6 MB) from the `cstr/nemotron-3.5-asr-streaming-0.6b-GGUF` Hugging
  Face repo (the engine's own auto-download target; published LFS OID
  re-verified on the VM), baked at `/opt/models` and pinned by content hash:
  the build asserts
  `sha256 22480dd283a3bf1b39e9d2ab86c048b0c1654b2fc280e46627a4cbfded4b46fd`
  with `sha256sum -c` before accepting the download (Dockerfile lines 99-100;
  the crispasr release tarball is likewise pinned,
  `fea397a3d7eb13741e3c65027fb1f51d0b362f9098257fbfe2d0e97e6c4fdf54`).
  Do not swap in another repo's `q4_k` bytes: different quantization means
  different WER/VRAM behavior (spike verdict A3).
- Uploads are decoded to 16 kHz mono PCM s16 by `ffmpeg` before the engine
  sees them (engine accepts 24 kHz natively; the adapter standardizes).
- CUDA 13 runtime libraries are an explicit image prerequisite: exactly
  `cuda-cudart-13-0` + `libcublas-13-0` from the NVIDIA ubuntu2404 repo
  keyring — no CUDA toolkit, no devel headers, no `nvidia/cuda` base image.
  Their directory is exported as `CRISPASR_LIB_DIR` and prepended (with the
  binary's own directory) to the subprocess `LD_LIBRARY_PATH`.
- Silent-CPU-fallback warning: without those libraries the crispasr CLI
  silently runs on CPU and exits 0 — no error surfaces. Measured slowdown
  is workload-sized: ~1.7x wall on the 60 s clip but ~11.6x on the 39-minute
  fixture (card t_18ee2e43; see "CPU execution — measured" above). The adapter
  therefore pins `--gpu-backend cuda` and probes `crispasr --version`
  before every first job per process, asserting that the pinned backend is
  in the build's `Registered backends` list. A `cuda` build that lacks the
  runtime libs fails that assertion fast with a typed `CrispasrConfigError`
  instead of burning minutes of CPU inference per job.
- VRAM: one whole-file invocation per job (the engine auto-slices audio into
  ~30 s windows internally; measured 1085 MiB peak over the full 39-minute
  fixture), an `nvidia-smi` sampler thread records per-stage allocated/peak
  memory (visible in `docker logs` as `allocated=/peak=` INFO lines), and a
  peak above `VRAM_LIMIT_MIB` raises `TranscriptionOOM` rather than
  thrashing the card.
- Final GPU-box state (post-cleanup, card t_413e2b7b, 2026-10-20): ~48 GB
  reclaimed — `df -h` free went from 25 GB to 73 GB. Purged: the homemade
  `nvidia-asr:local` image (18.5 GB) and leftover build bases, the docker
  build cache (25.0 GB), legacy volumes (4.7 GB), the spike/adapter host
  dirs (`/opt/nvidia-asr-spike`, `/root/spike-crispasr`,
  `/root/nvidia-asr-adapter` — all reports and scripts were archived first
  under `~/CODE/nvidia-asr-planning/archive/vm-cleanup-2026-10-20/`), the
  host `cuda13` apt runtime libs (~1 GB; the final image bundles its own,
  proven by a post-purge GPU job), and stale job artifacts (0.14 GB).
  Still running: the fresh-clone deployment at `/root/replication-test`
  (image `autodiarist:local`; volumes `replication-test_asr-cache`,
  `replication-test_queue-state`, `crispasr-cache`), GPU driver 580.178.04.
  A post-cleanup GPU job (`745556b4…`, 60 s clip, done < 20 s) logged
  `crispasr probe: version-ok backends=['cpu', 'cuda'] pinned=cuda`.

## Tests

```bash
python3 -m pytest tests
```

The API tests (including every endpoint example shape above) run in-process
against `MOCK_PIPELINE=1`, no GPU required. The Docker packaging tests
(`tests/test_docker_packaging.py`) assert the backend pins, runtime-library
prerequisites, GGUF hash verification, and that the torch-era knobs stay
removed. `scripts/vm_integration.py` is the on-VM acceptance harness for the
real CrispASR path. The optional smoke-test script's own tests
(`tests/scripts/`) drive `scripts/smoke-test.sh` against a local stub HTTP
server — they touch neither docker, the GPU, nor the internet, and running
the script itself is never part of activation or the test-suite bootstrap.

## Opt-in LLM cleanup pass (V2, default off)

A caller-injected spelling/punctuation/formatting cleanup pass exists as a
pure seam (`pipeline/llm_cleanup.py`): `process_job_background(..., llm=...)`
takes a `prompt -> response` callable supplied by the embedding caller. The
HTTP path never passes it and no environment toggle hardwires a network
call — same discipline as the `SELF_ID_LLM` / `extra_evidence` seam. The
transcript is chunked on speaker-turn boundaries with a seam overlap, sent
under a versioned rules preface (`pipeline/llm_preface.md`), and every
response is mechanically verified: exact line count/order, byte-identical
speaker labels and timestamps, spelling-only token edits. Any violation
rejects the chunk and keeps the original text; a per-chunk accept/reject
diff report (`llm_cleanup_report.md`) accompanies the job output. Optional
speaker-name evidence flows back through the existing `extra_evidence`
seam. Settings (`CleanupSettings`): chunk budget, overlap, timeout,
retries, opaque model description — with generation temperature pinned to
`0` by assertion. See PIPELINE.md § Opt-in LLM cleanup pass and CONFIG.md
§ v2 seams.

## Documentation map

| File | Covers |
| --- | --- |
| `CONFIG.md` | Every environment variable, defaults, validation, retired knobs |
| `PIPELINE.md` | CrispASR adapter, segment mapping, self-ID, opt-in LLM cleanup pass, render contract |
| `TROUBLESHOOTING.md` | Backend assertion, queue limits, failure post-mortems, restart behavior |
| `CHANGELOG.md` | Release history and QA traceability |
