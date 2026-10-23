# Diarization speaker-count ceiling (card t_38e52d89)

**TL;DR** — `DIARIZATION_MAX_SPEAKERS` (default **8**, `0` disables) bounds
the number of distinct speaker labels the sherpa diarization backend can
emit. The wrapper runs the historical auto-threshold pass first; **only
when its distinct label count exceeds the ceiling** does it re-run
clustering with `--clustering.num-clusters=<ceiling>` injected. Compliant
audio is never re-run and never over-split; the 118-min NASA SEWP Q&A
file that exploded to **100** labels (reference 6–7) is capped at ≤ 8.

## Why a ceiling at all

Measured before this change (`SHERPA_CLUSTER_THRESHOLD=0.40` shipped
default, full engine contract):

| file | duration | reference | shipped |
|---|---|---|---|
| SEWP Q&A (overlapping questions) | 118 min | 6–7 | **100 labels** |
| SEWP panel | 249 s | 5 | 3 labels (collapse path on some runs: 1) |
| BestPractice promo (single voice) | 137 s | 1 | 1–2 labels |
| marklex control (published GT) | 79 s | 2 | 2 speakers service-level |

On rapid overlapping exchanges FastClustering fragments the timeline: most
of the 100 labels cover one or two segments (README "Long-form inputs").
The product owner's constraint: meetings with >10 participants are not
expected, so an 8-or-fewer default is acceptable.

## The mechanism, and the two traps that shaped it

The engine (`crispasr` 0.8.43) invokes the sherpa CLI through this repo's
wrapper (`scripts/sherpa-diarization-wrapper.sh`, baked
`/opt/sherpa/sherpa-diarization-wrapper`, `SHERPA_BIN`). Per the CLI's own
`--help`:

> `--clustering.num-clusters`: If greater than 0, then cluster threshold is
> IGNORED. Please provide it if you know the actual number of clusters in
> advance. (default = -1)

**Trap 1 — append-only injection is a silent no-op.** The engine always
passes its own `--clustering.num-clusters=0`. Sweep 1 added
`--clustering.num-clusters=N` to the wrapper and swept N = 0, 4, 5, 6, 8,
10 on the panel file: **byte-identical output for every N** — the engine's
value won. Fix: the wrapper REMOVES every `--clustering.num-clusters`
(both `--flag=v` and `--flag v` shapes) from `"$@"` before injecting its
own (AC7; the strip loop is the exact code sweep 2 proved works).

**Trap 2 — a bare fixed `num-clusters` over-splits clean audio.** Sweep 2
(strip + inject, panel file, reference 5):

| injected NC | segments | speakers |
|---|---|---|
| 0 (auto) | 9 | 1 ← the collapse this file also exhibits |
| 5 | 19 | 4 ← close to reference |
| 8 | 24 | 7 |

and on the 137 s single-speaker promo (reference 1):

| injected NC | segments | speakers |
|---|---|---|
| 0 (auto) | 5 | 1 |
| 1 | 4 | 1 |
| 5 | 11 | 3 ← forced k splits one real speaker |

Forcing exactly *k* clusters on audio with fewer true speakers carves one
voice into several labels. A ceiling that always injected would therefore
break the clean fixtures the product owner also cares about.

## The final rule (implemented in the wrapper)

```
DIARIZATION_MAX_SPEAKERS = ${DIARIZATION_MAX_SPEAKERS:-8}
non-digit / negative values           -> ceiling OFF (fail open, historical behaviour)

ceiling == 0:
    exec real CLI with injected cluster-threshold, unbuffered   # exactly pre-card

ceiling > 0:
    pass 1 = pre-card command, stdout captured
    pass 1 non-zero exit  -> replay captured stdout, exit with its code (no ceiling pass)
    distinct speaker_XX labels in pass 1:
        <= ceiling (or zero parseable)  -> replay pass-1 stdout BYTE-IDENTICAL
        >  ceiling                      -> re-run:
              engine's --clustering.num-clusters stripped from "$@"
              --clustering.num-clusters=<ceiling> injected
              (cluster-threshold still passed; the CLI ignores it while num-clusters > 0)
```

Properties:

* **Non-interference.** Below the ceiling nothing changes for any
  deployment — same command, same bytes, one process, unbuffered stderr
  streaming preserved. The explosion path costs a second *clustering
  pass* (segmentation + embedding + clustering); the engine's ASR decode
  is NOT re-run (the wrapper only backs the diarization timeline CLI).
* **The ceiling is an upper bound on the emitted labels**, enforced where
  the labels are born (the CLI), so every consumer — engine transcript,
  renderer, `output_result.json`, markdown — sees the capped set.
* **One change, both images.** CUDA and Vulkan are one Dockerfile with
  `ARG GPU_BACKEND`; the wrapper COPY is in the shared runtime layer
  (`tests/test_docker_packaging.py` pins the single COPY;
  `tests/pipeline/test_speaker_count_ceiling.py` pins the knob in both
  compose files).
* Out of scope, deliberately: `SHERPA_CLUSTER_THRESHOLD` stays 0.40
  (separate measured decision), and the threshold injection stays on the
  ceiling pass even though the CLI ignores it there.

## Env knob

| Where | What |
|---|---|
| `scripts/sherpa-diarization-wrapper.sh` | reads `DIARIZATION_MAX_SPEAKERS`, default 8 |
| `Dockerfile` ENV | `DIARIZATION_MAX_SPEAKERS=8` (both GPU targets) |
| `docker-compose.yml`, `docker-compose.amd.yml` | `${DIARIZATION_MAX_SPEAKERS:-8}` passthrough |
| `.env.example`, `CONFIG.md`, `README.md` | documented, incl. how to raise/disable |

Raise it for a bigger room; set `DIARIZATION_MAX_SPEAKERS=0` to disable
(pre-card behaviour: unbounded auto-threshold clustering).

## Unit coverage (hermetic, no GPU)

`tests/pipeline/test_speaker_count_ceiling.py` drives the real script
against a stub CLI that mirrors the num-clusters-precedence semantics:

* explosion (100 labels) → final ≤ 8, ceiling pass injected;
* default = 8 with a 9-label auto pass; env raise to 12 leaves 12 alone;
  env lower to 4 caps to 4; `0` disables (one call, `num-clusters=0`
  untouched, engine command byte-faithful);
* labels 1/2/3/5/8 → single call, stdout replayed, no injection;
* threshold present on every pass (additive ceiling, out-of-scope guard);
* pass-1 failure propagates exit code without a ceiling pass;
* **AC7 strip proof**: the ceiling pass's argv contains exactly one
  `--clustering.num-clusters=8` (engine's `=0` removed);
* knob wired through Dockerfile/compose/docs; single shared wrapper COPY.

TDD order was honoured: the behavioural tests were written first and RED
against the pre-change wrapper (13 behavioural failures recorded in the
card thread before implementation).

## VM verification runs (NVIDIA host 10.4.10.58, container `autodiarist-04e2656-autodiarist-1`, engine v0.8.43, wrapper hot-installed from this branch; rollback copy at `/opt/sherpa/wrapper.bak-pre-ceiling`)

Numbers below are the measured results requested by the card (final
service-path speaker counts and, where taken, raw wrapper label counts):

| fixture | AC | reference | before | after (this change) |
|---|---|---|---|---|
| SEWP Q&A 118 min (`/tmp/qa.wav` pipeline-direct run, HTTP layer bypassed only; deployed job budget is 1800 s < this file's wall) | ≤ 8 | 6–7 | 100 | TBD |
| SEWP panel 249 s (service API job) | ≥ 2, not worse than 3, ideal 4–5 | 5 | 3 | TBD |
| BestPractice promo 137 s (service API job) | = 1 | 1 | 1 | TBD |
| marklex control 79 s (service API job) | = 2 | 2 | 2 | TBD |
| wrapper-vs-CLI replay diff on promo (compliant audio must be byte-identical to the pre-card command) | identical | — | — | TBD |

## Known limits and follow-ups

1. **Ceiling ≠ accuracy.** Cap 8 on the Q&A file yields *at most* 8
   labels; landing on the 6–7 reference exactly needs the clustering
   itself to be better — filed upstream against CrispASR (same ticket as
   the panel under-clustering, README "Diarization models").
2. **Acoustic near-duplicate merging.** The card's preferred mechanism
   (cap, then merge clusters that are acoustically near-duplicate) is
   not reachable at the wrapper boundary: the sherpa CLI exposes neither
   embeddings nor per-cluster centroids. A follow-up could request an
   embedding-dump flag upstream, or move the ceiling into the pipeline
   with label-averaged embedding comparison. Until then the conditional
   re-run is the measured-honest mechanism: it can only reduce the
   explosion, never distort compliant audio.
3. **Double clustering cost on explosion files.** Pass 2 repeats
   segmentation + embedding + clustering for over-ceiling inputs. Bounded
   (the expensive ASR decode is untouched) and rare by product
   constraint; if it ever hurts, the lever is `SHERPA_CLUSTER_THRESHOLD`
   (raise to merge earlier) or a smarter upstream counter.
4. **`DIAR_METHOD=foxnose` path is not capped** (it never produced label
   explosions: its failure mode is the opposite collapse). Documented
   behaviour, not a gap to close here.
5. **Image rebuild pending.** The verification above hot-installed the
   wrapper into the running container (exact file this branch COPYs).
   The next image build from `main` bakes it for both GPU targets;
   until then deployed containers can be patched with the same
   `docker cp` + `chmod 755` + `chown app:app` used for the runs.
