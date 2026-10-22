#!/usr/bin/env bash
#
# scripts/smoke-test.sh — OPTIONAL, user-invoked aliveness/plumbing check for
# a running AutoDiarist deployment (card t_66115e7d).
#
# WHAT IT DOES: five pass/fail checks against a live API endpoint:
#   (a) GET  /docs                      -> HTTP 200 (service is up)
#   (b) POST /v1/jobs  <fixture upload> -> HTTP 202 + token
#   (c) GET  /v1/jobs/{token} polled to a terminal state within --timeout
#   (d) GET  /v1/jobs/{token}/download?format=md   -> non-empty markdown with
#       a "| Duration |" metadata row and a "## Transcript" section
#   (e) GET  /v1/jobs/{token}/download?format=json -> valid result JSON with
#       status done and a non-empty segments array
#
# WHAT IT DOES NOT DO: it is NOT part of container activation (activation
# stays exactly: `docker compose up`). It runs NO docker command, touches no
# host/container paths, and writes nothing inside the deployment. It is NOT a
# transcription-quality or WER test — ALIVENESS AND PLUMBING ONLY. Running it
# against the production deploy is safe: it creates exactly one short job.
#
# FIXTURE lifecycle (product-owner + planner mandate, 2026-10-22):
#   primary  : download https://files.pyannote.ai/marklex1min.wav (a REAL
#              79 s two-speaker conversation, 44.1 kHz stereo) into a
#              `mktemp -d` temp dir, upload it AS PUBLISHED (never
#              pre-converted — if the service mishandles the format that is a
#              genuine finding this script must surface, not paper over).
#   fallback : if the download fails (offline, blocked) OR answers 200 with a
#              body that fails the RIFF/WAVE sanity check, discard any staged
#              bytes first (rm -f; ffmpeg also runs with -y — never upload
#              untrusted leftovers) and generate a throwaway ~5 s 16 kHz mono
#              tone via ffmpeg — the README's own recipe — verifying the
#              generated file is RIFF/WAVE before it is uploaded. Output
#              states truthfully WHICH case occurred and WHICH fixture was
#              used.
#   cleanup  : the temp dir (and the wav inside it) is removed on exit,
#              including on failure and on SIGINT/SIGTERM. Nothing is cached
#              anywhere user-visible; nothing touches /data/jobs or any
#              mounted volume; no audio is ever committed to this repo.
#   override : --audio FILE uses the user's own file and skips download/tone.
#
# Requirements: curl + python3. ffmpeg ONLY when the fixture download fails
# and no --audio was given; missing then -> clear actionable error, never a
# silent pass or skip.
#
set -u
set -o pipefail

readonly DEFAULT_ENDPOINT="http://127.0.0.1:8000"
readonly DEFAULT_FIXTURE_URL="${SMOKE_TEST_FIXTURE_URL:-https://files.pyannote.ai/marklex1min.wav}"
readonly POLL_INTERVAL="${SMOKE_TEST_POLL_INTERVAL:-3}"

ENDPOINT="$DEFAULT_ENDPOINT"
AUDIO=""
TIMEOUT=900
FIXTURE_URL="$DEFAULT_FIXTURE_URL"
TMP_DIR=""

cleanup() {
  [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ] && rm -rf "$TMP_DIR"
  TMP_DIR=""
}
trap cleanup EXIT
trap 'trap - INT TERM; cleanup; exit 130' INT
trap 'trap - INT TERM; cleanup; exit 143' TERM

die() { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-2}"; }

usage() {
  cat <<'EOF'
Usage: smoke-test.sh [--endpoint URL] [--audio FILE] [--timeout SECONDS] [--help]

Optional, user-invoked aliveness/plumbing smoke test for a running
AutoDiarist deployment. NOT part of container activation, NOT a
transcription-quality test. ALIVENESS AND PLUMBING ONLY.

  --endpoint URL     API base URL (default http://127.0.0.1:8000); point it
                     at a REMOTE deploy from an operator laptop — no docker,
                     no SSH, no host paths required.
  --audio FILE       Use your own audio file instead of the fixture
                     lifecycle below (uploaded as-is).
  --timeout SECONDS  Bounded wait for the job to reach a terminal state
                     (default 900). On expiry the last observed
                     status/stage/progress is reported.
  --help             This text.

Fixture (no file required): downloads
https://files.pyannote.ai/marklex1min.wav (real 79 s two-speaker audio,
uploaded as published) into a temp dir; on download failure falls back to a
generated 5 s 16 kHz ffmpeg tone. The temp dir is removed on exit, including
on failure or Ctrl-C. Requirements: curl + python3; ffmpeg only on the
fallback path. Exit codes: 0 all checks pass, 1 a check failed, 2 a
pre-flight error (e.g. missing tool and no fixture could be staged).

The script creates exactly ONE job per run — safe on a production deploy.
EOF
}

# ---- arg parsing -----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint) [ $# -ge 2 ] || die "--endpoint requires a URL"; ENDPOINT="$2"; shift 2 ;;
    --audio)    [ $# -ge 2 ] || die "--audio requires a FILE";   AUDIO="$2";   shift 2 ;;
    --timeout)  [ $# -ge 2 ] || die "--timeout requires SECONDS"; TIMEOUT="$2"; shift 2 ;;
    --help|-h)  usage; exit 0 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done
ENDPOINT="${ENDPOINT%/}"
case "$TIMEOUT" in
  ''|*[!0-9]*) die "--timeout must be a positive integer (seconds), got: $TIMEOUT" ;;
esac
[ "$TIMEOUT" -gt 0 ] || die "--timeout must be > 0"

printf 'AutoDiarist smoke test — ALIVENESS AND PLUMBING ONLY.\n'
printf 'This is not a container-activation step and not a transcription-quality\n'
printf 'test: transcription quality is not measured here.\n\n'

# ---- small helpers ---------------------------------------------------------
json_get() { # json_get <file> <python-expression on dict d> [default]
  python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as fh:
        d = json.load(fh)
except Exception:
    print(sys.argv[3]); sys.exit(0)
try:
    print(eval(sys.argv[2], {"d": d}))
except Exception:
    print(sys.argv[3])
' "$1" "$2" "${3:-}"
}

speaker_count() { # distinct non-empty speaker labels in a result JSON file
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    segs = d.get("segments") or []
    print(len({s.get("speaker") for s in segs
               if isinstance(s, dict) and s.get("speaker")}))
except Exception:
    print(0)
' "$1"
}

now_s() { date +%s 2>/dev/null || python3 -c 'import time; print(int(time.time()))'; }

sleep_s() {
  if [ -n "${SMOKE_TEST_SLEEP_HELPER:-}" ] && [ -x "${SMOKE_TEST_SLEEP_HELPER:-}" ]; then
    "$SMOKE_TEST_SLEEP_HELPER" "$1"
  else
    sleep "$1"
  fi
}

curl_show() { # curl_show <errlog> <curl args...>
  # Echo the exact command to the script's stderr (stdout captures stay
  # clean), run curl, and route curl's own errors into <errlog> so they are
  # preserved without polluting the echoed transcript.
  local errlog="$1"; shift
  printf '    $ curl' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  curl "$@" 2>>"$errlog"
}

FAILURES=0
SKIPPED=0
pass() { printf 'PASS   (%s) %s\n' "$1" "$2"; }
fail() { printf 'FAIL   (%s) %s\n' "$1" "$2" >&2; FAILURES=$((FAILURES + 1)); }
info() { printf 'INFO   %s\n' "$1"; }
skip() { printf 'INFO   %s\n' "$1"; SKIPPED=1; }

# ---- fixture lifecycle (staging happens before any job is created) --------
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/autodiarist-smoke.XXXXXX")" \
  || die "could not create a temp dir"
FIXTURE_STAGED=""
DEST="$TMP_DIR/smoke-fixture.wav"
CURL_ERRLOG="$TMP_DIR/curl.err"

if [ -n "$AUDIO" ]; then
  [ -f "$AUDIO" ] || die "--audio file not found: $AUDIO"
  FIXTURE_STAGED="user-supplied file $AUDIO"
else
  # FALLBACK_CAUSE records why we fall back; provenance must stay truthful: a
  # 200 whose body is not a real WAV is NOT a "download failure" (RR1).
  info "staging fixture: downloading $FIXTURE_URL"
  if curl_show "$CURL_ERRLOG" -fsSL --connect-timeout 10 --max-time 180 \
       -o "$DEST" "$FIXTURE_URL"; then
    # Sanity only: the server could answer 200 with an HTML error page.
    # This does NOT convert the audio — the bytes are uploaded as published.
    magic="$(dd if="$DEST" bs=1 skip=0 count=4 2>/dev/null)"
    size="$(wc -c <"$DEST" 2>/dev/null | tr -d ' ')"
    if [ "$magic" = "RIFF" ] && [ "${size:-0}" -gt 44 ]; then
      FIXTURE_STAGED="downloaded marklex1min.wav ($size bytes) from $FIXTURE_URL — uploaded as published"
    else
      FALLBACK_CAUSE="downloaded content failed the WAV sanity check (got '$magic', ${size:-0} bytes)"
    fi
  else
    CURL_RC=$?
    CURL_HINT="$(tail -n 1 "$CURL_ERRLOG" 2>/dev/null | tr -d '\r\n')"
    FALLBACK_CAUSE="fixture download failed (curl exit $CURL_RC${CURL_HINT:+: $CURL_HINT})"
  fi
  if [ -z "$FIXTURE_STAGED" ]; then
    # RR1 blocker guard: staged bytes are untrusted here (curl-failure
    # leftovers OR a 200-non-WAV body). ffmpeg refuses to overwrite a
    # pre-existing file yet STILL EXITS 0 on refusal, so without `rm -f` and
    # `-y` those junk bytes would be uploaded under a "generated tone" label.
    info "$FALLBACK_CAUSE — generating throwaway tone fallback via ffmpeg"
    rm -f "$DEST"
    command -v ffmpeg >/dev/null 2>&1 || die "fallback tone unavailable: $FALLBACK_CAUSE
  AND ffmpeg is not installed — cannot stage a smoke-test fixture.
  Causes: (1) could not fetch $FIXTURE_URL (check network/proxy) or its 200 body was not a WAV, (2) ffmpeg missing.
  Fix either one, or pass --audio FILE (any .wav) to supply your own throwaway file.
  NOTE: ffmpeg is only needed on this fallback path; the tone fixture is a fallback."
    # Same recipe README.md documents for audio.wav.
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
      -i sine=frequency=440:duration=5 -ac 1 -ar 16000 "$DEST" \
      || die "ffmpeg tone generation failed (see ffmpeg error above; cause: $FALLBACK_CAUSE)"
    # Fail-safe: never upload a suspect fallback file.
    tone_magic="$(dd if="$DEST" bs=1 skip=0 count=4 2>/dev/null)"
    [ -s "$DEST" ] && [ "$tone_magic" = "RIFF" ] \
      || die "ffmpeg produced no valid RIFF/WAVE at $DEST (got '${tone_magic:-<empty>}') — pass --audio FILE"
    FIXTURE_STAGED="generated ffmpeg tone fallback (5 s, 16 kHz mono)"
  fi
fi
info "fixture used: $FIXTURE_STAGED"
[ -n "$AUDIO" ] && FIXTURE_TO_UPLOAD="$AUDIO" || FIXTURE_TO_UPLOAD="$DEST"

# ---- check (a): aliveness --------------------------------------------------
CODE="$(curl_show "$CURL_ERRLOG" -sS -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 15 \
        "$ENDPOINT/docs" || true)"
if [ "$CODE" = "200" ]; then
  pass a "GET $ENDPOINT/docs -> HTTP 200 (service is listening)"
else
  # RR1 nit (c): surface curl's own last error (connection refused, DNS, TLS)
  # so the FAIL line is actionable, not just "HTTP no-response".
  A_ERR="$(tail -n 1 "$CURL_ERRLOG" 2>/dev/null | tr -d '\r\n')"
  fail a "GET $ENDPOINT/docs returned '${CODE:-no-response}'${A_ERR:+ — curl: $A_ERR} — is the service running? (SMOKE_TEST_ENDPOINT=... / --endpoint)"
fi

# ---- check (b): admission --------------------------------------------------
TOKEN=""
if [ "$FAILURES" -eq 0 ]; then
  CODE="$(curl_show "$CURL_ERRLOG" -sS -o "$TMP_DIR/post.json" -w '%{http_code}' --connect-timeout 10 --max-time 300 \
          -F "file=@${FIXTURE_TO_UPLOAD};type=audio/wav" "$ENDPOINT/v1/jobs" || true)"
  if [ "$CODE" = "202" ]; then
    TOKEN="$(json_get "$TMP_DIR/post.json" "d['token']" "")"
  fi
  if [ -n "$TOKEN" ] && [ "$CODE" = "202" ]; then
    pass b "POST $ENDPOINT/v1/jobs -> HTTP 202 + token $TOKEN"
  else
    detail="$(tr '\n' ' ' <"$TMP_DIR/post.json" 2>/dev/null | cut -c1-300)"
    fail b "POST $ENDPOINT/v1/jobs -> HTTP '${CODE:-no-response}' (expected 202 with token). Body: ${detail:-<empty>}"
  fi
else
  skip "(b) skipped — endpoint failed check (a)"
fi

# ---- check (c): bounded poll to terminal state ------------------------------
LAST_STATUS="never-observed"; LAST_STAGE=""; LAST_PROGRESS=""
if [ -n "$TOKEN" ]; then
  start="$(now_s)"; interval="${POLL_INTERVAL}"
  while :; do
    CODE="$(curl_show "$CURL_ERRLOG" -sS -o "$TMP_DIR/status.json" -w '%{http_code}' --connect-timeout 10 --max-time 30 \
            "$ENDPOINT/v1/jobs/$TOKEN" || true)"
    if [ "$CODE" = "200" ]; then
      LAST_STATUS="$(json_get "$TMP_DIR/status.json" "d['status']" "unknown")"
      LAST_STAGE="$(json_get "$TMP_DIR/status.json" "d.get('stage','')")"
      LAST_PROGRESS="$(json_get "$TMP_DIR/status.json" "d.get('progress','')")"
      case "$LAST_STATUS" in
        done|failed|cancelled) break ;;
      esac
    fi
    if [ $(( $(now_s) - start )) -ge "$TIMEOUT" ]; then
      break
    fi
    sleep_s "$interval"
  done
  if [ "$LAST_STATUS" = "done" ]; then
    pass c "job $TOKEN reached terminal status 'done' within ${TIMEOUT}s (never observed 'failed')"
  else
    fail c "job $TOKEN did not reach 'done' within ${TIMEOUT}s — last observed status='$LAST_STATUS' stage='${LAST_STAGE:-?}' progress='${LAST_PROGRESS:-?}' (a FAIL here on the marklex fixture with a real service may indicate the pipeline cannot decode 44.1 kHz stereo input — genuine finding, report it; do NOT pre-convert)"
  fi
else
  skip "(c) skipped — no token"
fi

# ---- checks (d)/(e): downloads ----------------------------------------------
if [ "$LAST_STATUS" = "done" ] && [ -n "$TOKEN" ]; then
  CODE="$(curl_show "$CURL_ERRLOG" -sS -o "$TMP_DIR/report.md" -w '%{http_code}' --max-time 60 \
          "$ENDPOINT/v1/jobs/$TOKEN/download?format=md" || true)"
  md_ok=0
  if [ "$CODE" = "200" ] && [ -s "$TMP_DIR/report.md" ]; then
    if grep -q "Duration" "$TMP_DIR/report.md" && grep -q "^## Transcript" "$TMP_DIR/report.md"; then
      md_ok=1
    fi
  fi
  if [ "$md_ok" = "1" ]; then
    pass d "GET .../download?format=md -> HTTP 200, non-empty markdown with Duration metadata + '## Transcript'"
  else
    fail d "GET .../download?format=md -> HTTP '${CODE:-no-response}' (expected 200 + non-empty markdown containing a Duration row and '## Transcript')"
  fi

  CODE="$(curl_show "$CURL_ERRLOG" -sS -o "$TMP_DIR/result.json" -w '%{http_code}' --max-time 60 \
          "$ENDPOINT/v1/jobs/$TOKEN/download?format=json" || true)"
  if [ "$CODE" = "200" ] \
     && [ "$(json_get "$TMP_DIR/result.json" "d.get('status','')")" = "done" ] \
     && [ "$(json_get "$TMP_DIR/result.json" "len(d.get('segments') or [])")" != "0" ]; then
    segs="$(json_get "$TMP_DIR/result.json" "len(d.get('segments') or [])")"
    pass e "GET .../download?format=json -> HTTP 200, status done, $segs segment(s) (valid JSON)"
    info "(e) observed speaker count: $(speaker_count "$TMP_DIR/result.json") (informational only — NOT a pass/fail gate; the expected real fixture has 2 speakers)"
  else
    fail e "GET .../download?format=json -> HTTP '${CODE:-no-response}' (expected 200, status done, non-empty segments)"
  fi
else
  skip "(d) skipped — job never reached done"
  skip "(e) skipped — job never reached done"
fi

# ---- verdict ------------------------------------------------------------------
cleanup
if [ "$FAILURES" -eq 0 ] && [ "$SKIPPED" -eq 0 ]; then
  printf '\nSMOKE TEST PASSED — 5/5 checks. ALIVENESS AND PLUMBING ONLY: transcription quality is not measured by this test.\n'
  exit 0
else
  printf '\nSMOKE TEST FAILED — %s check(s) failed. ALIVENESS AND PLUMBING ONLY: transcription quality is not measured by this test.\n' "$FAILURES" >&2
  exit 1
fi
