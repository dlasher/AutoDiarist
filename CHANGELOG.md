# Changelog

## [Unreleased] — global queue concurrency (card t_dbad2235)

**The service can now process N jobs concurrently behind
`MAX_PROCESSING_GLOBAL` (default 1).** `_queue_worker_loop` was serial by
construction — one task awaiting each job to completion before claiming the
next; `MAX_PROCESSING_PER_SESSION` only capped fair-share SELECTION, never
created parallelism. The lifespan now spawns a worker POOL sized from the
knob at boot (plus a dedicated per-job executor of the same width, so the
cap is never silently under-delivered by the default `min(32, cpu+4)`
thread pool), each worker running the same claim/process/wake cycle over the
already-atomic `_claim_next_queued`. The cap is enforced AT CLAIM TIME —
`_select_next_queued` blocks every claim while total in-flight ≥ N — so the
service never starts a job it cannot run, and a job whose engine hard-aborts
(measured at N=4 on the T400: `sched alloc encoder graph failed`) fails ONLY
that job with the real stderr tail while every worker keeps serving. The
single shared wake `asyncio.Event` (whose `clear()` one worker could steal
from another waiter) is replaced by `_WakeHub`: per-worker events, `set()`
fans out to all, nobody can strand anybody. Fair-share round-robin, the
per-session cap and `VRAM_LIMIT_MIB` semantics are untouched at any N.
Planner addition on this card: the nvidia-smi sampler reads the CARD TOTAL,
so under concurrency the VRAM guard compares against the FLEET cap
`VRAM_LIMIT_MIB × N` (per-job share x workers) — a job is no longer failed
for its siblings' memory; unset/invalid N keeps the old serial math, and on
the AMD image the sampler is inert either way. Default stays 1 on BOTH
compose profiles (`MAX_PROCESSING_GLOBAL: ${MAX_PROCESSING_GLOBAL:-1}`) —
the NVIDIA operating point is measured (T400: N=3 sustains 3/3 engines at
3207/4096 MiB; throughput 0.014 -> 0.032 jobs/s) but the AMD value must come
from measurement, and the rebuild of the AMD image follows this merge.
`validate_config()`, the boot log and `GET /v1/jobs`
(`processing_count`) all expose the knob and the live in-flight count.
(Tests: tests/api/test_concurrency_global.py, VRAM fleet-cap cases in
tests/pipeline/test_crispasr.py, compose asserts in
tests/test_docker_packaging.py.)

## [Unreleased] — domain lexicon that actually runs (card t_72405565)

**The spell-check layer now runs in the shipped service and carries the
measured NASA/SEWP vocabulary.** Four blockers made it inert: the API gate
required `SPELL_CHECK` to be exactly `1` (no compose file set it, so the
pass never ran), no `vocab.txt` existed, the default path was the relative
`vocab.txt` (unresolvable against the service CWD), and the matching rules
(>=6-character single words only) could not catch the measured Q4_K error
classes. The API seam and the module now share one rule
(`spell_check_enabled()`): ON by default, `{0,false,no,off}` opts out.
`vocab.txt` ships in the repo and is baked to `/app/vocab.txt`; the module
default is absolute (package-relative, no CWD dependence). New rules, all
unique-match-only: near-miss floor lowered to 4 chars ('gwec'->'GWAC'),
explicit measured-garble aliases ('sout, soupes => SEWP',
'soubcontracts => soup contracts', 'acceptional => exceptional'),
term-boundary split/join ('contractholders'->'contract holders',
'its av'->'ITCAV') restricted against prose windows ('as a', 'it can' are
never fused), and repeated single-character token runs collapse (the
measured `c c c c c c c c c cuture` hallucination). Word count inside a
segment may now change where the measured garbles changed it (docstring
promise revisited); segment count, order and timing are untouched, and
all-lowercase canonicals preserve source casing. Corpus probe across the
four source transcripts: 40 corrections, all genuine domain fixes, zero
prose damage. Both compose files set `SPELL_CHECK: "1"` explicitly.
(Tests: tests/pipeline/test_spellcheck.py, tests/api/test_spellcheck_seam.py.)

## [Unreleased] — speaker-count ceiling (card t_38e52d89)

**`DIARIZATION_MAX_SPEAKERS` (default 8) bounds the diarization label
count.** The NASA SEWP Q&A file (118 min, reference 6-7 speakers) exploded
to 100 distinct labels through the auto-threshold clustering path. The
sherpa wrapper now counts the distinct labels of the auto pass and, ONLY
when they exceed the ceiling, re-runs clustering with
`--clustering.num-clusters=<ceiling>` injected — after REMOVING the
engine's own `--clustering.num-clusters=0` from the argv, because the CLI
gives a positive num-clusters outright precedence over the threshold and
an appended flag would be a silent no-op (measured: byte-identical output
for every injected N until the strip was added). The ceiling is
conditional on purpose: a bare fixed num-clusters over-splits
cleanly-separated audio (measured: panel 2 -> 4 at NC=8), so compliant
fixtures replay byte-identical and nothing re-runs for them.
`0` disables the ceiling (pre-card behaviour). One wrapper script is the
single shared layer, so the CUDA and Vulkan images carry the change
identically; the knob is documented in README, CONFIG.md and
`.env.example`. Rule, sweeps, measurements and follow-ups:
`autodiarist-speaker-count-ceiling.md`. VM-measured: Q&A 100 -> **7**
speakers (326 segments, inside the 6-7 reference band); panel stays 3;
compliant fixtures replay byte-identical and never re-run. (Tests:
tests/pipeline/test_speaker_count_ceiling.py.)

## [Unreleased] — live job progress on long inputs (card t_2e901945)

**`GET /v1/jobs/{token}` now moves during long transcriptions instead of
freezing at 0.30.** Measured live: a 140-minute job published exactly
three progress values over two hours of engine time, because crispasr's
diarized stdout transcript is app-held until the final merge stage — a
pipe received zero bytes for ~80% of the run and neither `stdbuf -oL` nor
a pty released it earlier (the card's block-buffering diagnosis was
tested against the pinned 0.8.43 binary and refuted). The adapter now
passes `-pp` and tails the engine's unbuffered stderr stream, where
crispasr reports one real work boundary per completed audio slice
(`progress = X% (k/N slices)`), and publishes those as transcribing-band
progress — 86 distinct advancing events over a 39-minute input, live on
the NV host. The no-timer guarantee is unchanged: a silent engine still
means honestly frozen progress, and short jobs behave exactly as before.
(PIPELINE.md "Live progress on long jobs"; regression tests in
tests/pipeline/test_engine_streaming.py.)

## [Unreleased] — one-command deploy packaging (card t_f82ad22a)

**Contract: `git clone`, pick your compose file, `docker compose up` —
zero edits, on both hardware targets.** The user knows their hardware:
`docker-compose.yml` = NVIDIA CUDA, new `docker-compose.amd.yml` =
AMD/Vulkan. Nothing past `up` is part of activation.

- **One Dockerfile, two GPU targets.** The `GPU_BACKEND` build ARG
  (`cuda` | `vulkan`, default `cuda`) selects ONLY the host-runtime apt
  layer (CUDA-13 keyring + `cuda-cudart-13-0`/`libcublas-13-0` vs
  `libvulkan1` + `mesa-vulkan-drivers` = Mesa RADV on amdgpu, no ROCm)
  and the sha256-pinned CrispASR v0.8.43 release asset (`cuda13` digest
  unchanged; `vulkan` digest `d5c676e0…ca46` pinned per AC2). The app
  layer — wheels, ffmpeg, the Nemotron Q4_K GGUF, the whole sherpa
  diarization stack (card t_5a287979: CLI + both ONNX models + threshold
  wrapper + build-time self-check), the entrypoint, the non-root `app`
  user (uid 10001) — is ONE source of truth that cannot diverge between
  backends. This closes the regression class where the AMD operator
  script GENERATED a second `Dockerfile.amd` (foxnose-era, no sherpa):
  `amd-bringup.sh` is superseded — its job is now `git pull` + the AMD
  compose file; rebuilding via the old script would silently undo
  t_5a287979. **The AMD host still runs the pre-fix image until the
  operator rebuilds it with this path (pending).**
- **Fresh-clone blocker removed.** `env_file: .env` is gone from BOTH
  compose files (compose errors on a missing env_file target); every
  optional knob carries an inline `${VAR:-default}`; `.env.example` is
  now documented as optional overrides only.
- **AMD compose file is fully STATIC** — no host-specific gid/path
  anywhere (compose cannot interpolate `group_add` from `.env`; per-host
  device groups differ across driver stacks). Device access = `/dev/dri`
  + `/dev/kfd` devices + host-side `setfacl` grants for uid 10001 via
  the new idempotent `scripts/vulkan-host-setup.sh` (setfacl primary,
  announced 0666-mode fallback, `--persist` udev rule for reboots).
  `CRISPASR_GPU_BACKEND: vulkan` is a pinned literal, so a missing
  Vulkan runtime fails the `--version` probe instead of silently
  CPU-falling-back. `CRISPASR_LIB_DIR` empty; `VRAM_LIMIT_MIB: 7168`
  mirrors the proven bring-up on the 8 GB reference card.
- **One entrypoint.** New `scripts/docker-entrypoint.sh` (executed as
  the image `ENTRYPOINT` on both targets) exports `VK_ICD_FILENAMES` to
  the single `*radeon*|*amdgpu*` ICD under `/usr/share/vulkan/icd.d/` —
  selecting ALL ICDs made the engine die at init with `vulkan:11` on the
  proven bring-up — and execs the same uvicorn command the CUDA path
  used before. No-op on the CUDA image.
- **Tests** (tests/test_docker_packaging.py, 11 -> 22): both engine
  assets + digests in the single Dockerfile; structural assert that the
  app layer never branches on `GPU_BACKEND`; the AMD contract (pinned
  vulkan/sherpa literals, device list, forbidden
  group_add/privileged/rocm/`ENV{`); no `env_file` and no bare
  interpolations in either compose file; entrypoint ICD glob; and a
  cross-check that EVERY env key set by either compose file is actually
  consumed by `api/`+`pipeline/` sources or a baked script (invented
  keys are silent config regressions — planner review).

## [Unreleased] — multi-speaker diarization fix (card t_5a287979)

**Diarization default changed: `DIAR_METHOD=foxnose` -> `sherpa`.**
foxnose's global BIC speaker counter collapses panel recordings — live
engine measurement on the 249 s NASA SEWP file:
`foxnose windows=357 skipped=0 -> 1 speakers
(cosine_similarity_single_speaker)`. The engine's own help names
`sherpa`/`ecapa` ("external sherpa-onnx subprocess with segmentation +
speaker embedding + clustering") as the reliable path; on the image it
was dead — no sherpa CLI, no ONNX models — AND failing silently:
label-less segments normalized into a fake single-speaker transcript.

* `pipeline/transcribe.py`: `DIAR_METHOD_DEFAULT=sherpa`; the two model
  paths + optional `SHERPA_BIN` wire into the engine command;
  `sherpa`/`ecapa` without their models now raise `CrispasrConfigError`
  at startup (AC4 — silent degradation closed); label-less engine output
  for a labeling method raises `CrispasrError` at runtime; a present
  single label still maps to exactly one speaker (AC7). `foxnose`/`none`
  stay selectable.
* `scripts/sherpa-diarization-wrapper.sh` + `SHERPA_CLUSTER_THRESHOLD`
  (default 0.40): injects the clustering distance threshold the engine
  cannot pass; calibration table in README "Diarization models".
  Shipped default keeps panel >= 3 speakers (reference is 5; exact 5
  needs 0.35 which costs single-speaker files — recorded honestly).
* `Dockerfile`: three new vendored, sha256-pinned artifacts from
  k2-fsa/sherpa-onnx releases (CLI tarball -> `/opt/sherpa`, reverb
  segmentation + eres2net embedding ONNX -> `/opt/models`, wrapper ->
  `/opt/sherpa/sherpa-diarization-wrapper`). The layers are
  backend-independent (CPU onnxruntime), so CUDA and Vulkan images carry
  the SAME files and the same default (AC5); packaging card t_f82ad22a
  folds the GPU variant into a build arg around them.
* `tests/pipeline/test_diarization_speakers.py`: RED-on-main regression
  suite (9 failed / 1 passed before the fix) — startup refusals, command
  wiring, silent-degradation guard, single-speaker preservation, plus
  engine-gated integration tests asserting the panel fixture yields
  >= 3 distinct speakers and the pyannote 2-speaker control keeps >= 2.
* marklex 44.1 kHz stereo note: the service decode path (`ffmpeg -ac 1
  -ar 16000`) already normalizes any input before the engine/sherpa
  ever see it; the raw sherpa CLI's "Expect sample rate 16000" abort
  only affects bypassing calls. Not a service bug.

- **AMD vs NVIDIA measured A/B** (2026-10-22, kanban t_97bbd597). Two deploys of
  the same image family, same model bytes, same `DIAR_METHOD=sherpa`, same
  diarization models; only the engine build and the GPU differ. Measured as a
  pure consumer over the HTTP API.
  - `whatIsNASASEWP_2026Update-CC` (249.0 s): NVIDIA 222.7 s (1.12x) |
    AMD 105.4 s (**2.36x**) — AMD 2.1x faster
  - `BestPractice_RFQ_CC` (137.5 s): NVIDIA 78.4 s (1.75x) |
    AMD 39.2 s (**3.50x**) — AMD 2.0x faster
  - Both backends returned identical output shape (18 seg / 3 spk and
    8 seg / 2 spk) — the difference is throughput, not transcript content.
  - RE-MEASURED on the MATCHED build (`000bc2f` — ceiling, lexicon, progress
    fix), both hosts on the same app layer, on an IDLE queue, using each job's
    OWN elapsed time (started_at to terminal status) so queue wait is excluded.
    The earlier figures (AMD 12.78x / 3.70x) were measured on a busy host and
    were inflated by up to 3.5x.
  - Output shape is IDENTICAL on both hosts: 18 seg / 3 spk (panel) and
    8 seg / 2 spk (single-voice). Only the time differs.
  - The domain lexicon is verified live on BOTH hosts: GWAC, SEWP, ITCAV, NASA,
    contract holders and soup contracts all present; the measured garbles
    (gwec, sout, soupes, soubcontracts, acceptional, contractholders) absent.
  - The AMD lead comes mostly from the CPU: diarization clusters on one core
    while the GPU does ASR, and the AMD host has 12 cores / 112 GiB against
    the NVIDIA host's 4 cores / 7.9 GiB.
  - The sherpa diarization pass clusters on CPU, so throughput depends on the
    host's core count as well as its GPU — this is why the AMD lead holds
    (and widens) once sherpa is the default method.
- **README "Other backends" Vulkan row** no longer credits the retired
  `amd-bringup.sh` operator script; it now carries the measured speed, the
  hardware identity, and the fact that the AMD target is reached from a fresh
  clone with `docker compose -f docker-compose.amd.yml up --build`.
- **Operational note added to the README:** `CRISPASR_TIMEOUT_SECONDS` (default
  1800) is a hard wall-clock budget for the whole job, not an idle timeout.
  A 140-minute input needs ~83 min on the 4-core T400 and ~38 min on the AMD
  host, so long jobs must raise it (e.g. `CRISPASR_TIMEOUT_SECONDS=10800`).

## [1.5.0] - 2026-10-21 — V1.5 wave: multi-user behaviour on one service (integrated by card t_f41ad4f2)

One deployment shared by several people at once: every user gets a
server-issued session id, every submission is accepted and queued instead
of rejected per session, the queue drains fairly across sessions, each
user can list their own jobs, and every queued job carries an honest wait
estimate. No pipeline/diarization changes in this release — the behaviour
delta lives in `api/main.py`. The card sections below (newest first, as
stacked on the release branch) are preserved verbatim from the branches.

**Config changes in this release (new/changed env vars):**

| Variable | Change | Card |
|---|---|---|
| `SESSION_HEADER` | session header default changed to `X-Session` (explicit config still honored; `Authorization` works only when configured) | t_920fb7dc |
| `SESSION_COOKIE_NAME` | new, default `autodiarist_session` — server-issued session cookie | t_920fb7dc |
| `MAX_JOBS_PER_SESSION` | deprecated: still read from `.env`, ignored at runtime with a boot warning; the per-session 429 rejection path is gone — `MAX_QUEUE_DEPTH` is the only admission rejection (503) | t_82bfbc15 |
| `MAX_PROCESSING_PER_SESSION` | new, default `1`, scheduler-enforced per-session GPU cap | t_4b40acfd |
| `PROCESSING_SECONDS_PER_AUDIO_SECOND` | new, default `0.185` (measured on this T400 rig; hardware-specific; `0` disables ETA fields) | t_b3ae9366 |
| `AUDIO_DURATION_FALLBACK_SECONDS` | new, default `300`, substituted in the ETA prefix sum only for unprobeable jobs queued ahead | t_b3ae9366 |

`queue_position` semantics also changed (an API field, not config): rank
in fair execution order, no longer arrival order — see the t_4b40acfd
section. New API surface: `GET /v1/jobs` session listing (t_f97d12a8),
`duration_seconds` / `estimated_wait_seconds` / `estimated_completion_at`
on the 202, every poll and queued listing entries (t_b3ae9366), and the
`session` field + session cookie on `POST` responses (t_920fb7dc).

**Known limitations (accepted by maintainer ruling, release card
t_f41ad4f2 — no hotfix card for 1.5.0):**

- **Job-artifact authorization is by token possession.** Session identity
  gates *listing*; it does not gate per-job operations. A
  `DELETE /v1/jobs/{token}` (or status/download GET) carrying a valid job
  token is honored even when it comes from a different session — live
  observation on the canonical `:8000` deployment: session A's DELETE of
  session B's job token returned 202 and the job was cancelled
  (`Cancelled by client while queued`). This is unchanged, deliberate
  behavior from the identity card (t_920fb7dc), which explicitly scoped
  itself to session issuance/precedence/listing and declared per-job
  authorization out of scope; no wave card implemented session-authz on
  the job endpoints. Release-card AC(e) is amended accordingly: own-cancel
  evidence is `202` → `cancelled`, and the cross-session DELETE is
  recorded as this known limitation.

### V1.5 session-scoped job listing: GET /v1/jobs (card t_f97d12a8)

Building on session identity (card t_920fb7dc) and accept-and-queue admission
(card t_82bfbc15): a client can now answer "what did I submit?" — the
user-facing requirement to wait for a job or cancel it needs a way to find
its own tokens first.

- **`GET /v1/jobs` — session-scoped listing.** Entries carry `token`,
  `status`, `progress`, `stage`, `source` (uploaded filename) and
  `created_at`; `queued` entries add `queue_position` (the same fair rank
  the poll endpoint and the POST 202 report, computed from one shared
  simulation), `failed` entries add `error`, and `cancelled` jobs stay
  listed (terminal states are still part of what the user submitted).
  Filtered by the same hashed `session_ident` admission stamps on every job
  — no cross-session leakage is possible and there is no admin mode.
- **Ordering & paging.** Newest-first (`created_at` desc, stable `seq` desc
  tiebreak). `?limit=` defaults to `50` and clamps to `[1, 200]` (clamped,
  never an error; non-integer `limit` is `422`). `total` reports the
  session's job count before truncation. Deliberately knob-free: the two
  limits are constants, not env vars.
- **`created_at` joins the job state at admission** as the listing's stable
  clock; `queued_at` keeps its queue-facing meaning (boot requeue stamps
  it). Legacy jobs without `created_at` fall back to `queued_at`, so an
  upgrade lists pre-existing history without re-admission.
- `GET /v1/jobs/{token}` is unchanged — token-based access stays as it is;
  the listing is the session-based view. The response echoes the session
  identity exactly like `POST` does (cookie + `X-Session` default header).
- Tests: 9 new acceptance tests (`tests/api/test_jobs_listing.py`); full
  suite 254 green. README documents the endpoint; CONFIG.md unchanged.

### V1.5 honest wait estimate: duration probe + queue-aware ETA (card t_b3ae9366)

Building on fair-share scheduling (card t_4b40acfd): a queued client can now
decide *wait or cancel* from an honest number instead of a bare position.

- **Duration probe at admission.** `POST /v1/jobs` runs a cheap ffprobe
  container-header read (no decode) on the stored upload and persists the
  result as `duration_seconds` in the job state; a failed probe stores
  `null` (unknown — never a fabricated 0). The job's own run refreshes the
  value once from the decoded 16k wav. Polls NEVER re-probe and never read
  the audio file.
- **`PROCESSING_SECONDS_PER_AUDIO_SECOND` (default `0.185`).** The linear
  model rate (GPU seconds per audio-second). The default is documented at
  its source — measured on the T400 rig: 39-min audio ran 423–434 s wall on
  CUDA — and is hardware-specific; CONFIG.md gives the re-derivation
  procedure. Exact `0` disables ETA emission; negative/non-numeric warn at
  boot and fall back to the default (also in the boot config summary as
  `eta_rate=... eta=on|OFF`). `AUDIO_DURATION_FALLBACK_SECONDS` (default
  `300`) substitutes for queue members AHEAD whose probe failed, so one
  unprobeable file cannot poison the tail.
- **Queue-aware ETA on every session surface.** Queued jobs carry
  `estimated_wait_seconds` (seconds from now until the RESULT should exist:
  the running job's remaining estimate + every fair-order job ahead + own
  estimate) and `estimated_completion_at` (`YYYY-MM-DDTHH:MM:SSZ`) — on the
  202 admission body, on every `GET /v1/jobs/{token}` poll, and on every
  `queued` entry of the session job listing `GET /v1/jobs` (one shared
  queue snapshot, so the surfaces cannot disagree). Positions
  get distinct, non-overlapping estimates by the prefix-sum construction;
  completion times chain (each position finishes exactly its own estimate
  after the one ahead). Polling alone keeps the estimate current across
  admissions, cancellations and completions — everything recomputes from
  live state in the fair execution order, and the processing job's
  remaining share shrinks monotonically with elapsed time (negatives
  clamp to 0).
- **Honesty contract.** The fields are explicitly ESTIMATES from a linear
  model: diarization variance, chunk overhead and fixed startup cost are
  not modeled and observed wall time can and will differ; a `processing`
  job returns the fields as `null` (stage/progress carry the truth), a
  finished job omits them, and a queued job with unknown own duration gets
  no estimate rather than a fabricated one.

Tests: `tests/api/test_wait_estimate.py` (30, CPU-only, no real audio — the
probe is stubbed where needed): empty-queue estimate, N-jobs-ahead prefix
sums, fair-order (not admission-order) ranks, decrease as jobs ahead
complete/cancel, explicit nulls while processing, negative clamps, no
re-probe at request time, rate validation (0 disables, invalid warns), and
the 202/poll/listing HTTP surfaces (listing entries agree with the poll
endpoint by construction). Full suite green: 284 passed (stacked on
card t_f97d12a8's listing branch).

### V1.5 fair-share scheduling: round-robin per session + processing cap (card t_4b40acfd)

Building on accept-and-queue admission (card t_82bfbc15): the shared GPU is
now allocated fairly across sessions, so a session that submits many jobs
can no longer starve one that submits a single job.

- **Round-robin claim order.** The worker picks the next job from the
  session whose turn is longest overdue (FIFO by `seq` inside each
  session). The pointer — a global `serve_counter` plus per-session
  `last_served` stamps — persists to `JOBS_DIR/_scheduler.json` and is
  re-read on every claim, so a restarted service continues the same
  fairness round instead of letting an established submitter re-jump the
  line. A missing or corrupt pointer heals to a fresh round with a warning
  (worst case: one round ordered by arrival); service start never fails on
  it.
- **`MAX_PROCESSING_PER_SESSION` (default `1`)** caps how many jobs of one
  session may be `processing` concurrently. The cap is enforced by the
  scheduler at claim time — a session at the cap is skipped even when its
  queued jobs sit at the head of its sub-queue — so it holds under a future
  multi-worker scale-up, not by accident of the single worker. Values `< 1`
  or unparseable log a warning and clamp to `1` (a zero cap would freeze
  the queue).
- **`queue_position` semantic change (explicit):** the field is now the
  rank in the fair-scheduled **execution order**, not raw admission order.
  A job can therefore report a position behind a later arrival from a
  session that has waited longer, and — unlike under FIFO — a new arrival
  from a never-served session can jump ahead, so a queued job's reported
  position can rise as well as fall. Within-session FIFO is guaranteed:
  jobs of one session never overtake each other. `POST` 202 and `GET`
  polling share one deterministic simulation of the worker's real pickup
  order.
- **Preserved guarantees:** monotonic `seq` (now documented as the arrival
  ticket it always was), FIFO-within-session, on-disk persistence across
  restart, boot requeue of orphaned `processing` jobs with `seq` intact,
  and never two `processing` jobs for one session — now a scheduler
  invariant, not a worker-count accident.
- Out of scope by maintainer decision: duration/size-aware priority and
  preemption of a running job; fairness applies at selection only.
- Docs updated: README (architecture block, admission + GET schemas with
  the semantic-change note, new *Fair-share scheduling* section), CONFIG.md
  knob table, TROUBLESHOOTING.md restart & endpoint gotchas, PIPELINE.md
  worker note, `.env.example`.
- `tests/api/test_fair_share.py` (AC1-AC6, 14 cases, no GPU): fairness
  ordering, cap enforcement independent of worker count, position ranks,
  restart preserving fairness with no starvation, corrupt-pointer recovery,
  boot requeue, cancelled-job order removal.

### V1.5 admission: accept-and-queue, no per-session rejection (card t_82bfbc15)

- **POST /v1/jobs now accepts and queues instead of rejecting.** A session
  may hold any number of active jobs; every submission that fits under
  `MAX_QUEUE_DEPTH` answers `202` with the `token`, the `session` id and
  the `queue_position` (1-based FIFO rank at admission — the same rank
  `GET /v1/jobs/{token}` reports while `queued`). The per-session `429`
  ("Session limit reached") response path is removed; the service no
  longer emits `429` anywhere.
- **`503` is the only rejection path** (queue genuinely full): `Retry-After`
  header as before, and the detail now names both the depth and the
  Retry-After value — `Queue is full (n/limit jobs queued). Retry after Xs.`
- **`MAX_JOBS_PER_SESSION` is deprecated but accepted**: existing `.env`
  files keep loading; when set, startup logs
  `config: MAX_JOBS_PER_SESSION=... is DEPRECATED and IGNORED` and the
  value is dropped from the boot config summary. No job is rejected on its
  basis.
- Docs updated: README HTTP API admission table + examples, CONFIG.md
  queue/admission rows, TROUBLESHOOTING.md queue failures & restart
  semantics, `.env.example` no longer ships the deprecated knob.
- `tests/api/test_admission_202.py` (AC1-AC6); the retired-quota cases in
  `test_queue_hardening.py` / `test_session_issue.py` were rewritten to the
  accept-and-queue contract.

### V1.5 server-issued session identity (card t_920fb7dc)

Quotas and per-job attribution become per-*person* rather than per-IP:
the server now mints a random high-entropy session id (uuid4 hex) on
first contact and returns it BOTH as the `autodiarist_session` cookie
(HttpOnly, SameSite=Lax) AND as the `session` field of the
`POST /v1/jobs` body, so API clients without a cookie jar can capture
and reuse it. The server stays stateless — no session table; identity
is derived purely from what the client presents.

- **`api/main.py`** — `_session_id` replaced by `_resolve_session`, a
  precedence chain: `SESSION_HEADER` header (default now **`X-Session`**;
  `Authorization` remains honored when configured explicitly) > server-issued
  cookie > client IP > `anonymous`. Every accepted job persists the identity
  digest under `session` (unchanged domain-separated SHA-256, first 32 hex —
  raw ids/cookies/IPs never hit disk) plus the dedicated `session_ident` key
  (hash of the session id itself, including the first-contact issuance) so
  the scheduler and job listing can group by person; `session_source` records
  which branch resolved (`header`/`cookie`/`ip`/`anonymous`). Cookie name is
  overridable via `SESSION_COOKIE_NAME`; `validate_config()` runs at boot and
  logs the effective `SESSION_HEADER`, `SESSION_COOKIE_NAME`,
  `MAX_JOBS_PER_SESSION`, `MAX_QUEUE_DEPTH`, `RETENTION_DAYS`, warning and
  falling back on invalid values.
- **`tests/api/test_session_issue.py`** — new: AC1-AC8 coverage — issuance
  in Set-Cookie + body, entropy/uniqueness, cookie-reuse stability,
  header-beats-cookie precedence, configured `Authorization`, IP/anonymous
  fallback, raw-value leak check, two-session distinguishability with
  per-session quota, statelessness (no files but job dirs; identity
  never remembered server-side), and the boot config report incl.
  `SESSION_COOKIE_NAME` override + invalid-name warn/fallback.
- **Docs** — README *Session identity* rewritten for the issuance/reuse
  flow, `CONFIG.md` `SESSION_HEADER` default and chain, TROUBLESHOOTING
  shared-bucket row.

## [1.3.1] - 2026-10-21 — conversational-word rejection in self-ID capture (card t_ac7889e4)

Live strict-tier defect found during the (b) verification (t_90e1752b):
CrispASR case-restores mid-sentence prose, so the spoken decoy "I'm sorry"
arrives in the diarizer output as "I'm Sorry". The capitalized capture is
load-bearing on the strict tier (it is how "I'm Shelley Festing" must work),
"Sorry" is 5 chars so it clears the <4-char weak-single-word guard, and it
is not in the phrase blocklist — the live clip resolved Speaker 2 as the
name "Sorry" (conf 0.70). Pre-existing defect, not an option (b) regression.

- **`pipeline/self_identification.py`** — new conservative
  `_CONVERSATIONAL_WORDS` set (sorry, going, coming, trying, working,
  looking, thinking, afraid, glad, sure, done, here, there, fine, okay,
  ready, really, currently, basically, honestly) applied as a secondary
  filter inside `_candidate_name` — the single choke point every capture
  tier funnels through (strict self-intro, option (b) lowercase, handoff,
  third-party intro). A candidate is rejected when ANY word is
  conversational: the single word ("I'm Sorry" -> `Sorry`), the surname
  position (AC5: "Smith Sorry"), or the given-word position (the card's
  "Sorry Smith"). The set deliberately excludes plausible surnames (May,
  Young, Poe, Hart, Cook, Good, Small, Strange, Rich, Wise, Brown, White,
  Green, Black), so "I'm Sarah Cook" / "I'm Mary Young" / "I'm Fred Rich"
  keep resolving; the two-word given+surname shape guard remains the
  primary defense. Rejections log `contains conversational word(s)`.
- **`tests/pipeline/test_self_id_conversational_words.py`** — new
  regression file: the live clip's verbatim decoy string and its
  lowercase spelling resolve nothing; "I'm Going To Explain That" and
  "This Is Going To Be Quick" (which truncate through the function-word
  retry to `Going`) resolve nothing on both tiers; two-word captures with
  a conversational given or surname word are rejected; "I'm Shelley
  Festing", "I'm Jane Smith", "I'm Sarah Cook", "I'm Mary Young", "I'm
  Fred Rich" and the lowercase-tier equivalents still resolve.

## 1.3.0 - 2026-10-21 (issue #1 option (b) — integrated by card t_e2b82e85)

Follow-up to the 1.2.0 issue #1 fix: the maintainer's final ruling is
option (b) — allow lowercase captured names after the two unambiguous
self-introduction cues — landed on top of the strict-only variant that
shipped in 1.2.0. No breaking API changes.

- **`pipeline/self_identification.py`** — two-tier capture grammar:
  `my name is` / `this is` now take a lowercase-tolerant ASCII name capture
  (`_NAME_CASE_TOLERANT`, scoped `(?i:...)`), while the ambiguous phrases
  `I'm` / `I am` / `<name> here` keep the 1.2.0 strict `(?-i:...)` capture.
  The allowance is not a back door for prose: captures run through the same
  validation pipeline (function-word rejection with truncation retry,
  blocklist, weak-single-word rule, ASCII-only check) plus a new name-shape
  guard — a lowercase-dependent capture must be exactly two words
  (given+surname), so `my name is jane wu` yields `jane wu` while
  `this is strictly about ability one` and `my name is wrong` yield no
  suggestion. Environment-override patterns keep the strict wrap; a `(?i:`
  marker inside an override opts into the lowercase path explicitly.
- **Truncation retry reaches the lowercase path**: `my name is shelley
  festing the rest …` resolves to `shelley festing`. The
  `non-name function words` log still fires for genuinely prose-like
  captures (e.g. `My name is New Manager`).
- **Known limitation**: single-word lowercase names (`my name is alice`)
  and three-word lowercase names are not accepted — case is the only cheap
  signal separating `my name is wrong` from a name; capitalize the ASR
  output or extend `SELF_ID_NAME_BLOCKLIST`-style tuning for edge cases.
- **Correction to the 1.2.0 note**: option (b) was not "retracted by the
  card banner before merge" — 1.2.0 shipped strict-only behaviour
  (`this is jane wu` yielded no suggestion), and this release supersedes
  that as the maintainer's chosen design.
- **`tests/pipeline/test_self_id_strict_name_capture.py`** — option (b)
  acceptance rows: lowercase name after both unambiguous phrases, lowercase
  after `I'm` / `I am` rejected, the prose back-door guard, truncation retry
  on the lowercase path, and the function-word floor on two-word lowercase
  captures; the grammar-shape test now asserts the two-tier split.
  `test_self_identification.py`'s `this is jane wu` case is re-encoded from
  the strict-only expectation back to option (b).
- **`PIPELINE.md`** — the self-ID section documents the two-tier capture.

## 1.2.0 - 2026-10-20 (issue #1 self-ID capture fix — integrated by card t_ba40a8fd)

Fixes GitHub issue dlasher/AutoDiarist#1: `suggest_speaker_names()` returned `{}`
on real webinar audio — the any-case name capture let prose match as names, and a
single trailing function word discarded otherwise-valid whole captures. Rebased
onto the 1.1.0 tree; both the v2 stage/journal/retention/LLM-cleanup behaviour
and the issue #1 fix ship together. No breaking API changes.

### Strict self-intro name capture + function-word truncation retry (card t_16c871a6)

- **`pipeline/self_identification.py`** — self-introduction name captures
  are now case-sensitive (`_NAME_STRICT`, scoped `(?-i:...)`) while the lead
  phrase stays case-insensitive, so lowercase prose like
  "my name is rajiv is from" no longer matches as a name, and every
  `(?P<name>...)` group in `SELF_ID_PATTERNS` environment overrides is
  auto-wrapped to keep that guarantee under `re.IGNORECASE`.
- **Function-word truncation retry**: a captured trailing function word
  ("My name is Shelley Festing the") now shortens the capture from the tail
  instead of discarding the whole match; rejection only when the single-word
  floor is itself a function word. Handoff/third-party-intro captures are
  unchanged.
- **Maintainer decision**: the shipped design is the issue's strict variant —
  case-sensitive capture on every self-intro phrase (`my name is` / `I'm` /
  `I am` / `this is` / `<name> here`). The alternative option (b) (allowing
  lowercase names after the unambiguous cues `my name is` / `this is`) did
  not land here: 1.2.0 shipped strict-only behaviour, so `this is jane wu`
  yields no suggestion and `I'm sorry` never matches. *(Corrected by 1.3.0,
  card t_1a02ecbf: the maintainer's final ruling is option (b); see the
  1.3.0 section — this release is accurate history, not the final design.)*
  The function-word
  rejection path stays reachable and keeps logging `non-name function words`
  for genuinely prose-like captures.
- **`tests/pipeline/test_self_id_strict_name_capture.py`** — issue #1
  regressions; the two `test_self_identification.py` cases whose assertions
  encoded the old lowercase-capture behavior were updated per the card.

## 1.1.0 - 2026-10-20 (V2 wave — integrated by card t_bfd9ca2e)

The second feature wave, merged from per-card branches into main with all
conflicts resolved without dropping any card's behaviour; the full offline
suite is green on the merged tree (see below). No breaking API changes.

### Stage-based progress + status journal (card t_14ac6b49)

- Every status publish carries the machine-readable `stage` enum
  (`queued` → `decoding` → `transcribing` → `diarizing` → `spellcheck` →
  `llm_cleanup` → `self_identification` → `rendering` → `done`);
  `progress` moves within each stage's band (README "Progress stages",
  PIPELINE.md stage table).
- `progress` is monotonic non-decreasing across the whole job — the
  boot-reclaim requeue → claim path never rewinds it (review rework) —
  and is published only at real work boundaries, never from timers;
  `progress == 1.0` iff `status: "done"`.
- Append-only `<job_dir>/status_journal.jsonl` records every state
  transition (including the boot-reclaim requeue line) as an audit trail
  beside the mutable `status.json`. Tests: `tests/api/test_stage_progress.py`.

### Per-chunk engine progress (card t_1eb108e0)

- The CrispASR adapter streams the engine's stdout line-by-line and maps
  recognised `asr`/`diar` chunk boundaries onto the `transcribing`/
  `diarizing` bands (`api/main.py::_ENGINE_STAGES`), so progress refines
  per pass/chunk during the single whole-file invocation; unrecognised
  output is not a work boundary. Tests: `tests/pipeline/test_engine_streaming.py`.
- Root `conftest.py` pins suite imports to the checked-out repo root, so
  `pytest` is green from any worktree even under a PEP-660 editable
  install (QA rework).

### Opt-in LLM cleanup pass (card t_48c2efae)

- **`pipeline/llm_cleanup.py`** — opt-in, caller-injected cleanup pass over
  the transcript: fixes spelling/punctuation/capitalization only, never
  source content. Chunks whole speaker-turn lines to a character budget
  (oversized turns get their own chunk, never split) and carries a
  read-only seam overlap into each subsequent prompt so chunk boundaries
  contain the previous seam's text, not just indices.
- **Mandatory verification guard**: each chunk response must preserve the
  exact line count, order, `N|` numbering, and byte-identical
  `Speaker N [t0-t1] —` prefixes; token-level checks allow only
  case/punctuation normalization and Damerau-Levenshtein near-misses
  (spelling fixes) — any wording change rejects the whole chunk and the
  original text is kept. Temperature is pinned to `0.0` by assertion.
- **`pipeline/llm_preface.md`** — versioned AGENT rules preface
  (`PREFACE VERSION: 1`) stating exactly what may and may not change.
- **Per-chunk accept/reject diff report**: counts plus before/after for
  every permitted edit and rejection reasons; through the service seam it
  lands in `llm_cleanup_report.md` and the rendered report gains a
  `## LLM transcript cleanup` summary section (`meta['cleanup_summary']`).
- **Service seam (AC4)**: `process_job_background(..., llm=...,
  cleanup_settings=...)` — the HTTP/queue path never injects a client; the
  module performs no network I/O; no env toggle hardwires a call
  (`SELF_ID_LLM`/`extra_evidence` precedent). `CleanupSettings.from_env()`
  documents budget knobs only (`LLM_CLEANUP_CHUNK_CHARS`,
  `LLM_CLEANUP_OVERLAP_LINES`, `LLM_CLEANUP_TIMEOUT_S`,
  `LLM_CLEANUP_MAX_RETRIES`, `LLM_CLEANUP_MODEL`,
  `LLM_CLEANUP_REVIEW_NAMES`).
- **Optional speaker-name review** (default off): quoted verbatim evidence
  lines are harvested only from ACCEPTED chunks and returned as
  `Evidence`-shaped mappings for the existing
  `suggest_speaker_names(..., extra_evidence=[...])` seam.
- **Tests**: 25 new offline tests (`tests/pipeline/test_llm_cleanup.py`,
  `tests/api/test_llm_cleanup_seam.py`) covering turn-aware chunking,
  seam-overlap prompts, preface rules, spelling-accept/wording-reject,
  label/timestamp immutability, line-count/order enforcement,
  retries/timeout, temperature assertion, evidence→`extra_evidence` flow,
  and an end-to-end rendered-report diff showing only the permitted change
  (suite: 150 passed, no network).

### Job retention + disk-pressure eviction (card t_5e6200fc)

- `RETENTION_DAYS` is now a live knob (default `7` → `30`): a retention
  sweep deletes finished job dirs (`done`/`failed`/`cancelled` whose
  `status.json` mtime exceeds the window) synchronously at service boot and
  hourly after that; every removal logs the token with `reason=age-expiry`.
  `RETENTION_DAYS=0` disables age-based expiry. The live pair
  `queued`/`processing` is never age-swept (boot requeue must not be
  starved by an expired backlog; an in-flight job must never vanish).
- New disk-pressure eviction: `DISK_MIN_FREE_MIB` (default 8192) +
  `DISK_CHECK_INTERVAL_SECONDS` (default 300). While free space on the
  `JOBS_DIR` filesystem sits below the watermark, job dirs are evicted
  oldest-mtime-first until it recovers; a `processing` job is never evicted
  and every eviction logs token + `reason=disk-pressure`. Scope is
  `JOBS_DIR` only — the mounted CrispASR cache volume is untouched.
- Docs updated: CONFIG.md gains a retention/disk-pressure table and the
  knob is no longer documented as reserved; README and
  TROUBLESHOOTING retention prose now describe shipped behaviour;
  `.env.example` / `docker-compose.yml` expose the three knobs.

### Operator how-to for unverified backends (card t_a13c1342)

- README "Other backends" now carries per-row prerequisites, selection flags,
  operational differences, and exact v0.8.43 asset URLs for CUDA-12,
  Vulkan, HIP/ROCm, CPU plain/legacy/AVX512, Metal, WASM, and CANN — all
  explicitly marked NOT verified on this project's hardware — plus the
  upstream ggml preference-order quote vs. our `--gpu-backend` pin, the
  `CRISPASR_BIN`/`GGUF_MODEL` asset-swap recipe (incl. container mount
  for non-CUDA builds), and a "convert a row to verified" checklist.
  Documentation only: no builds, no new test claims.

### Restart-recovery verification (cards t_3575f1f8, t_8f56bc92)

- Live restart-recovery proof on the VM (t_3575f1f8) and an independent
  audit of that evidence (t_8f56bc92) close out the V2 wave: boot reclaim
  and requeue of in-flight jobs were verified on live runs under the new
  stage/journal model, not just offline tests. Verification-only cards;
  no shipped-code changes from them.

**Merged-tree verification (this release):** full offline suite
**177 passed** (no network), covering every card's tests together —
stage progress, per-chunk engine streaming, LLM cleanup + service seam,
retention lifecycle, queue hardening, docker packaging.

## 1.0.2 - 2026-10-20 (docs final — no code changes)

- CPU-only validation of the pinned `cuda13` v0.8.43 asset (card
  t_18ee2e43, README commit 2bffe0e): with `libggml-cuda.so` quarantined —
  the documented `GGML_BACKEND_DL` fallback — the 60 s spike source ran on
  4 vCPUs at ~1.7x CUDA wall (ASR-only 13.7 s median vs 8 s; diarized
  24.4 s vs 13 s; peak RSS ~1,065 MiB) with 97.7% word parity, 3/3
  sections and identical speaker labels; the full 39-minute fixture took
  142.9 s vs 12.3 s (~11.6x) at 71.2% parity. `nvidia-smi` showed 0
  compute apps / 0 MiB GPU throughout; caveat documented:
  `--gpu-backend cpu` alone still dlopens the CUDA plugin. This
  measurement scopes the older `~11x` estimate in the guard message
  (`pipeline/transcribe.py`): it holds at production-file lengths, not on
  short clips.
- Live fresh-clone replication proof (card t_99d78ff4): the public repo
  (`github.com/dlasher/AutoDiarist`) was cloned at 88dee5d to
  `/root/replication-test` on the T400 VM and `docker compose build` +
  `up` succeeded from the clone alone — both `sha256sum -c` verifications
  passed (model + engine), the backend assertion logged
  `crispasr probe: version-ok backends=['cpu', 'cuda'] pinned=cuda`, and
  2 E2E jobs ran on the 39-minute fixture (252/252 segments, 3 speakers,
  wall 437 s; peak `allocated=1083 MiB / peak=1085 MiB`), zero ERROR
  lines. Host driver 580.178.04 vs image CUDA userland 595.71.05
  compatibility confirmed.
- VM cleanup complete (card t_413e2b7b): ~48 GB reclaimed on the
  validation box (`df -h` free 25 GB -> 73 GB) — homemade
  `nvidia-asr:local` image (18.5 GB), docker build cache (25.0 GB),
  legacy image layers/volumes (4.7 GB), spike/adapter dirs (10.1 GB),
  host `cuda13` apt runtime (~1 GB, image-bundled now), stale job
  artifacts (0.14 GB). All reports/patches were copied first to
  `~/CODE/nvidia-asr-planning/archive/vm-cleanup-2026-10-20/` (sealed
  tarball sha256 `f73eb3d4…`). Deployment survived the purge: post-cleanup
  GPU job `745556b4-…` completed in the kept
  `/root/replication-test` stack; `/opt/nvidia-asr` (131 MB) retained as
  reviewer-approved fallback.

## 1.0.1 - 2026-10-19

- Project rebrand `nvidia-asr` -> `AutoDiarist` (commit 1dc36af).
- Renderer pipeline stamp now reports the actual backend identity
  (card t_3275e056, follow-up to the docs rewrite):
  `autodiarist v1 (CrispASR nemotron Q4_K @<version>)`, with `<version>`
  parsed from the `crispasr --version` probe output and cached alongside
  the backend registration; `@unknown` only until a probe has run.
  `MOCK_PIPELINE=1` renders the deterministic `autodiarist v1 (mock)`.
  The legacy `NeMo Speech @<sha>` label and the `nemo_speech_sha`/
  `nemo_sha` meta keys are retired from `pipeline/renderer.py`; golden
  fixture, `tests/pipeline/test_renderer.py`, and the PIPELINE.md
  "As-built quirk" note updated to the new contract.

## 1.0.0 - 2026-10-19 (post-pivot release)

- Backend pivot: transcription + diarization now runs through the
  **CrispASR subprocess backend** (`crispasr` v0.8.43 CUDA-13 CLI, Nemotron
  3.5 ASR streaming 0.6B Q4_K GGUF, sha256 `22480dd2…`, engine default
  ~30 s internal windowing). The in-process NeMo/torch path and its
  torch-era environment knobs (`DEVICE_DTYPE`, `PYTORCH_*`, `HF_HOME`) are
  retired; no `nemo`/`torch` import remains on the real path.
- Diarization default is `DIAR_METHOD=foxnose` (globally-stable speaker
  IDs); `ecapa`, `sherpa`, `sortformer`, `none` selectable.
- Silent-CPU-fallback guard: the adapter pins `--gpu-backend cuda` and
  asserts backend registration from `crispasr --version` before each
  first job per process; missing CUDA-13 runtime libraries fail jobs fast
  instead of silently running ~11x slower on CPU.
- VRAM discipline: per-stage `nvidia-smi` sampling with `allocated=/peak=`
  INFO lines, `TranscriptionOOM` on breach of `VRAM_LIMIT_MIB` (3584),
  `CRISPASR_TIMEOUT_SECONDS` (1800) wall-clock budget.
- Post-merge pipeline hardening: sub-`MIN_FRAGMENT_SECONDS` (2.0 s)
  fragment merge against 30 s-window mis-clustering, `<en-US>`/NST
  suppression (`--suppress-nst` + parser-side tag stripping), hard failure
  on empty/missing engine output (never a silent fallback transcript).
- Queue hardening (card t_daebf369 lineage): disk-persisted FIFO queue,
  `MAX_QUEUE_DEPTH`/`MAX_JOBS_PER_SESSION` admission (503/429 +
  `Retry-After`), queued-job cancellation with restart revalidation,
  `queue_position` in status.
- Self-identification v2: weighted-evidence suggestions
  (`self_intro` 0.7 / `third_party_intro` 0.5 / `handoff` 0.4, tight/loose
  attribution bonuses, conflict penalty, name-plausibility blocklist,
  cross-label collision dedup) with the `extra_evidence` seam reserved for
  the (default-off) E4 LLM review pass.
- Render contract per SPEC 08 "Render contract — speaker naming"
  (supersedes annotate-only): direct-name transcript body for resolved
  speakers, `## Speaker mapping` table, `## Speaker identification`
  evidence section, raw-segment appendix.
- Language fixed to `en-US` (SPEC 08 "Languages").
- Docker deliverable rebuilt: crispasr prebuilt + GGUF baked by verified
  sha256, apt CUDA-13 runtime libraries, foxnose embedder + job state on
  named volumes, GPU reservation pinned.
- Documentation set (README/CONFIG/PIPELINE/TROUBLESHOOTING) rewritten to
  the shipped post-pivot reality; CONFIG.md cross-checked line-by-line
  against every environment lookup in `api/main.py` and `pipeline/*.py`.

### QA traceability

Accepted by real end-to-end verification (kanban task t_c17fd4fb,
"QA verification: real E2E run of /root/audio.wav on VM, no mock",
verdict PASS): job token **`decb8249-bfd4-43ee-a7c9-b9ac05ed02a9`** —
`POST /v1/jobs` 202, polling reached `done` on `MOCK_PIPELINE=0`,
downloaded Markdown contains 107 segments of genuine transcribed speech
(zero "Mock transcript"/"fallback segment" occurrences), metadata duration
2340.01 s (~39 min) matching the source, speaker attribution and the raw
diarizer appendix verified. Evidence comment retained on the QA card.

## 0.x (pre-pivot development, 2026-10-16)

- Added the HTTP job API, deterministic mock-pipeline integration path,
  and pipeline helper modules for chunking/reconciliation, vocabulary-only
  spelling correction, advisory speaker self-identification, and Markdown
  rendering (NeMo/torch in-process experiments superseded by the CrispASR
  backend above; the spike report carries the pivot evidence, card t_7e17589e).
