"""The complete environment-variable catalogue of the AutoDiarist app.

Card t_1799d07f, AC5: one place naming EVERY knob the app reads, with
what it affects and the default the code actually falls back to. The
catalogue is MECHANICALLY verified against the source tree by
``scripts/check-config-catalog.py`` — that script fails CI if a knob is
read anywhere in api/, pipeline/, or the container entrypoint/wrapper
without being catalogued here (or catalogued without being read), or if a
recorded default drifts from the code's literal default.

Lives under ``pipeline/`` deliberately: the config *documentation* card
(t_16c871a6) owns README/CONFIG.md prose; this module is the machine
source of truth, and prose should link here, not duplicate it.

Fields
------
name     the environment variable, exactly as read.
default  the fallback the CODE uses, as text (None = unset means "off",
         "auto", or a computed path; the effect says which). Values here
         are verified literal-for-literal where the code's default is a
         scalar literal (see the checker).
where    files that read it (authoritative set is discovered by the
         checker; this column is human orientation).
effect   one plain sentence: what setting it changes, and — for the
         capacity knobs — the failure code the operator gets when the
         value is wrong (card t_1799d07f: OUT_OF_MEMORY says "lower
         MAX_PROCESSING_GLOBAL", TIMEOUT says "raise
         CRISPASR_TIMEOUT_SECONDS").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigVar:
    name: str
    default: str | None
    where: str
    effect: str


CATALOG: tuple[ConfigVar, ...] = (
    # ---- queue / concurrency / admission ------------------------------------
    ConfigVar("JOBS_DIR", "/tmp/autodiarist_jobs", "api/main.py; entrypoint; wrapper",
              "Root directory for per-job state (status.json, journal, artifacts)."),
    ConfigVar("MAX_UPLOAD_BYTES", "524288000", "api/main.py",
              "Upload size cap; bodies over it are rejected with HTTP 413 and the "
              "job dir is removed (413 text classifies as the "
              "INPUT_TOO_LARGE error code)."),
    ConfigVar("MAX_QUEUE_DEPTH", "50", "api/main.py",
              "Total queued jobs admitted before POST starts answering 429."),
    ConfigVar("MAX_JOBS_PER_SESSION", None, "api/main.py",
              "Deprecated as of V1.5: accepted but IGNORED (unset or set, no "
              "effect); per-session admission is governed by MAX_QUEUE_DEPTH "
              "fair scheduling and MAX_PROCESSING_PER_SESSION."),
    ConfigVar("MAX_PROCESSING_GLOBAL", "1", "api/main.py; pipeline/transcribe.py; wrapper",
              "Service-level concurrency: spawns N engine workers and scales the "
              "VRAM fleet cap. Measured ceiling 3 on the 4 GB NVIDIA host — N=4 "
              "deterministically fails the 4th engine with OUT_OF_MEMORY; LOWER "
              "THIS when a job's error_code is OUT_OF_MEMORY. Also divides the "
              "auto-resolved sherpa thread count."),
    ConfigVar("MAX_PROCESSING_PER_SESSION", "1", "api/main.py",
              "Concurrently-processing jobs allowed for ONE session."),
    ConfigVar("QUEUE_WORKER_ENABLED", "1", "api/main.py",
              "Set 0 to freeze the queue (admission without execution; tests use it)."),
    ConfigVar("QUEUE_POLL_SECONDS", "0.2", "api/main.py",
              "How often the queue worker looks for the next claimable job."),
    ConfigVar("RETRY_AFTER_SECONDS", "30", "api/main.py",
              "Retry-After header value sent with 429 admission refusals."),
    ConfigVar("PROCESSING_SECONDS_PER_AUDIO_SECOND", "0.185", "api/main.py",
              "ETA model rate: seconds of compute per second of audio."),
    ConfigVar("AUDIO_DURATION_FALLBACK_SECONDS", "300.0", "api/main.py",
              "Duration assumed for ETA when the admission probe fails (unset "
              "falls back to 300)."),
    # ---- storage / retention -------------------------------------------------
    ConfigVar("RETENTION_DAYS", "30", "api/main.py; entrypoint",
              "Days before finished job dirs are swept (0 disables the age sweep)."),
    ConfigVar("DISK_MIN_FREE_MIB", "8192", "api/main.py; entrypoint",
              "Free-disk watermark: below it the retention sweep evicts finished "
              "jobs early rather than letting the volume fill."),
    ConfigVar("DISK_CHECK_INTERVAL_SECONDS", "300", "api/main.py; entrypoint",
              "How often the disk watermark is re-checked."),
    # ---- engine (crispasr) ---------------------------------------------------
    ConfigVar("CRISPASR_BIN", None, "pipeline/transcribe.py; entrypoint",
              "Path to the crispasr engine binary; unset = the packaged default "
              "location. A missing/unexecutable value fails jobs as "
              "CrispasrConfigError (UNKNOWN code)."),
    ConfigVar("GGUF_MODEL", None, "pipeline/transcribe.py",
              "Path to the Q4_K GGUF model; unset = engine auto-downloads."),
    ConfigVar("CRISPASR_TIMEOUT_SECONDS", "1800", "pipeline/transcribe.py",
              "Wall-clock ceiling per engine run; expiry kills the process group "
              "and the job records error_code=TIMEOUT — raise this (hosts run "
              "10800) when TIMEOUT appears."),
    ConfigVar("CRISPASR_SHERPA_TIMEOUT_SEC", None, "pipeline/transcribe.py",
              "Override for the pipeline-side diarization subprocess timeout "
              "(split path, card t_7685c6da); unset = engine parity — "
              "max(120, ceil(audio_s) * 5), the formula crispasr applies to "
              "its own sherpa pre-pass."),
    ConfigVar("VRAM_LIMIT_MIB", "3584", "pipeline/transcribe.py",
              "Per-engine VRAM cap (card total from nvidia-smi); the sampler "
              "kills runs crossing it with TranscriptionOOM -> error_code"
              "=OUT_OF_MEMORY."),
    ConfigVar("CRISPASR_VRAM_SAMPLE_SECONDS", "1.0", "pipeline/transcribe.py",
              "Interval of the VRAM sampler during engine runs."),
    ConfigVar("CRISPASR_CACHE_DIR", None, "pipeline/transcribe.py",
              "Engine model cache directory; unset = engine default (~/.cache)."),
    ConfigVar("CRISPASR_LIB_DIR", None, "pipeline/transcribe.py",
              "Extra directory prepended to LD_LIBRARY_PATH for the engine's "
              "bundled CUDA libs; unset = packaged locations."),
    ConfigVar("CRISPASR_GPU_BACKEND", "cuda", "pipeline/transcribe.py",
              "Engine GPU backend (cuda | vulkan)."),
    ConfigVar("CRISPASR_LANGUAGE", "en-US", "pipeline/transcribe.py",
              "ASR target language tag passed to the engine."),
    ConfigVar("DIAR_METHOD", "sherpa", "pipeline/transcribe.py",
              "Diarization backend; any other value is a CrispasrConfigError."),
    ConfigVar("MIN_FRAGMENT_SECONDS", "2.0", "pipeline/transcribe.py",
              "Fragments shorter than this are merged into neighbours at render."),
    ConfigVar("DIARIZATION_MIN_REGIONS_PER_MINUTE", "0.5", "pipeline/transcribe.py",
              "Speaker-region plausibility floor per audio minute (card "
              "t_dc5c2abb). Measured calibration: the collapsed 8411 s run "
              "produced 0.0143 regions/min vs 3.37 for the clean run; the "
              "gap is 235x, so 0.5 is nowhere near a knife's edge."),
    ConfigVar("DIARIZATION_MIN_CHECK_SECONDS", "600.0", "pipeline/transcribe.py",
              "Small-audio exemption for the degradation guard: shorter "
              "files are never judged by the region floor (a 137 s "
              "single-speaker recording at 0.44 regions/min must PASS)."),
    ConfigVar("SILENCE_SPLIT_SECONDS", "1.5", "pipeline/renderer.py",
              "Silence gap that starts a new speaker run in the rendered report."),
    # ---- sherpa subprocess (diarization engine) -------------------------------
    ConfigVar("SHERPA_BIN", "/usr/local/bin/sherpa-diarization-wrapper.sh",
              "pipeline/transcribe.py; wrapper",
              "The diarization subprocess command."),
    ConfigVar("SHERPA_REAL", "/opt/sherpa/bin/sherpa-onnx-offline-speaker-diarization",
              "wrapper",
              "The REAL sherpa binary behind the thread-resolving wrapper."),
    ConfigVar("SHERPA_SEGMENT_MODEL", "computed: /opt/models/sherpa-onnx-reverb-diarization-v1.onnx",
              "pipeline/transcribe.py",
              "ONNX segmentation model path for sherpa."),
    ConfigVar("SHERPA_EMBEDDING_MODEL", "computed: /opt/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
              "pipeline/transcribe.py",
              "ONNX speaker-embedding model path for sherpa."),
    ConfigVar("SHERPA_NUM_THREADS", None, "wrapper",
              "Explicit sherpa thread count; unset = auto-resolved as "
              "cores // MAX_PROCESSING_GLOBAL, capped by SHERPA_MAX_THREADS."),
    ConfigVar("SHERPA_MAX_THREADS", "8", "wrapper",
              "Cap for the auto-resolved sherpa thread count (4 is the measured "
              "optimum on both shipped hosts)."),
    ConfigVar("SHERPA_CLUSTER_THRESHOLD", "0.40", "wrapper",
              "Sherpa clustering threshold: higher merges fewer speakers."),
    ConfigVar("DIARIZATION_MAX_SPEAKERS", "8", "wrapper",
              "Speaker ceiling: the most speakers FastClustering may emit."),
    ConfigVar("DIAR_HEARTBEAT_INTERVAL_SECONDS",
              "2.0", "pipeline/transcribe.py",
              "Seconds between progress ticks published while the split "
              "diarization subprocess runs (card t_868131e3). Ticks are an "
              "elapsed-time ESTIMATE, labelled estimated=true on the job "
              "status; the real completion boundary is still published. "
              "Non-numeric or non-positive warns and keeps 2.0."),
    ConfigVar("DIAR_CHUNK_ESTIMATE_SECONDS",
              "2.0", "pipeline/transcribe.py",
              "Estimated CPU seconds of diarization work per heartbeat tick "
              "(card t_868131e3): total = ceil(audio x rate / this), rate = "
              "DIAR_SECONDS_PER_AUDIO_SECOND (card t_79f9cdc6 re-priced the "
              "stage at its own measured rate; was the ASR constant). "
              "Non-numeric or non-positive warns and keeps 2.0."),
    ConfigVar("DIAR_SECONDS_PER_AUDIO_SECOND", "0.3661",
              "pipeline/transcribe.py; api/main.py",
              "Seconds of diarization wall time per second of audio. "
              "Measured on the NV host: 0.3661 s/s-audio for the 8411 s "
              "SEWP file and 0.4781 for the 249 s panel, against the ASR "
              "stage's 0.1016 and 0.1245 - a stable ~3.6-3.8x across a 34x "
              "spread of audio length. The heartbeat's tick total and the "
              "running-phase ETA for the diarizing stage use THIS, not the "
              "ASR rate (card t_79f9cdc6). Non-numeric or non-positive "
              "falls back to the measured default."),
    ConfigVar("SHERPA_LD_LIBRARY_PATH", "/opt/sherpa/lib", "wrapper",
              "Library path the wrapper uses for the sherpa subprocess."),
    # ---- serving --------------------------------------------------------
    ConfigVar("PORT", "8000", "entrypoint",
              "uvicorn listen port."),
    ConfigVar("VK_ICD_FILENAMES", None, "entrypoint",
              "Vulkan ICD pin for the vulkan GPU backend; unset = entrypoint "
              "pins the single RADV/amdgpu ICD when present."),
    ConfigVar("MOCK_PIPELINE", "0", "api/main.py; pipeline/transcribe.py; pipeline/renderer.py; wrapper; entrypoint",
              "1 = deterministic mock transcript, no ffmpeg/engine — GPU-free CI."),
    # ---- vocabulary pass --------------------------------------------------
    ConfigVar("SPELL_CHECK", "true", "pipeline/spellcheck.py",
              "Vocabulary correction on/off (0|false|no|off disables)."),
    ConfigVar("VOCAB_PATH", None, "pipeline/spellcheck.py",
              "Custom vocabulary file; unset = packaged vocab.txt."),
    ConfigVar("SELF_ID_SECTION", "1", "pipeline/renderer.py",
              "0 omits the self-identified-speakers section from the rendered "
              "report; speaker appendix remains authoritative either way."),
    # ---- self-identification ------------------------------------------------
    ConfigVar("SELF_ID_RESOLVE_THRESHOLD", "0.7", "pipeline/self_identification.py",
              "Cosine similarity at which a suggested speaker identity resolves."),
    ConfigVar("SELF_ID_ANNOTATE_THRESHOLD", "0.4", "pipeline/self_identification.py",
              "Similarity floor below which the annotation is dropped."),
    ConfigVar("SELF_ID_BLOCKLIST_PATH", "computed: packaged name blocklist path",
              "pipeline/self_identification.py",
              "File of names never to suggest; unset = packaged default."),
    ConfigVar("SELF_ID_NAME_BLOCKLIST", None, "pipeline/self_identification.py",
              "Inline extra blocked names (comma/newline separated); unset = none."),
    ConfigVar("SELF_ID_PATTERNS", "computed: packaged strict-capture regex list",
              "pipeline/self_identification.py",
              "Override patterns for the self-introduction capture (JSON array or "
              "plain string)."),
    ConfigVar("SELF_ID_HANDOFF_PATTERNS", "computed: packaged handoff regex list",
              "pipeline/self_identification.py",
              "Override patterns for speaker-handoff detection."),
    ConfigVar("SELF_ID_INTRO_PATTERNS", "computed: packaged third-party intro regex list",
              "pipeline/self_identification.py",
              "Override patterns for third-party introductions."),
    ConfigVar("OVERLAP_SECONDS", "30", "pipeline/chunking.py",
              "Re-roll window shared between chunk boundaries."),
    ConfigVar("CHUNK_SECONDS", "600", "pipeline/chunking.py; api/main.py",
              "Chunk length the engine is fed; 0 disables chunking."),
    # ---- session identity ----------------------------------------------------
    ConfigVar("SESSION_HEADER", "X-Session", "api/main.py",
              "Header carrying the session identity for queue fairness/isolation."),
    ConfigVar("SESSION_COOKIE_NAME", "autodiarist_session", "api/main.py",
              "Cookie name carrying the same identity when the header is absent."),
    # ---- LLM cleanup seam (opt-in injection; env tunes the pass) -------------
    ConfigVar("LLM_CLEANUP_MODEL", None, "pipeline/llm_cleanup.py",
              "Override model name for the cleanup call; unset = library default."),
    ConfigVar("LLM_CLEANUP_CHUNK_CHARS", None, "pipeline/llm_cleanup.py",
              "Character budget per cleanup chunk; unset = library default."),
    ConfigVar("LLM_CLEANUP_OVERLAP_LINES", None, "pipeline/llm_cleanup.py",
              "Lines re-sent between adjacent chunks; unset = library default."),
    ConfigVar("LLM_CLEANUP_TIMEOUT_S", None, "pipeline/llm_cleanup.py",
              "Per-call timeout seconds; unset = library default."),
    ConfigVar("LLM_CLEANUP_MAX_RETRIES", None, "pipeline/llm_cleanup.py",
              "Retry count per chunk; unset = library default."),
    ConfigVar("LLM_CLEANUP_REVIEW_NAMES", None, "pipeline/llm_cleanup.py",
              "Name-review pass on/off (0|false|no|off disables); unset = enabled."),
    # ---- cache / ETA / mock ------------------------------------------------
    ConfigVar("CACHE_DISABLED", None, "api/main.py",
              "Pure KILL SWITCH for the transcript cache (card t_ac748379). "
              "Unset/empty: the per-request X-Cache opt-in governs. Any "
              "enabled value forces the cache OFF for the whole service, so "
              "a job still runs but nothing is cached or served from cache. "
              "NOTE: a host env var only reaches the container if it is "
              "LISTED in the compose environment: block."),
    ConfigVar("MOCK_TRANSCRIBE_SECONDS", "0", "pipeline/transcribe.py",
              "Test-only hold in the MOCK_PIPELINE path: seconds the mock "
              "transcribe parks before returning (default 0 — instant)."),
    ConfigVar("RUNNING_ETA_MIN_PROGRESS", "0.30", "api/main.py",
              "Fraction of progress a processing job must cross before the "
              "running-phase estimate is published (card t_d64bdf29). Below "
              "it, no honest estimate exists and the field is null."),
    ConfigVar("RUNNING_ETA_REANCHOR_SECONDS", "120.0", "api/main.py",
              "Seconds a published running-phase estimate is held when "
              "progress freezes (a stalled stage) before it is withdrawn "
              "(card t_d64bdf29). Never moves up; only tightens."),
)
