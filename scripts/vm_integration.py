"""VM integration check for the CrispASR adapter (card t_4d5f7d08, AC1/AC2).

Run ON the GPU VM:

    REPO=/root/autodiarist-adapter python3 scripts/vm_integration.py \
        /root/audio.wav /root/crispasr_adapter_run/segments.json

with CRISPASR_BIN / GGUF_MODEL exported. Prints ONLY summary lines (counts,
monotonicity, speaker labels, VRAM stats) — full segment JSON stays on disk
in the given output path, never echoed to the session.
Exit codes: 0 = all checks pass, 1 = a check failed, 2 = backend error.
"""

import json
import logging
import os
import sys
import traceback

sys.path.insert(0, os.environ.get("REPO", "/root/autodiarist-adapter"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from pipeline import transcribe  # noqa: E402


def main() -> int:
    audio = sys.argv[1] if len(sys.argv) > 1 else "/root/audio.wav"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "segments.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        segments = transcribe.transcribe_and_diarize(
            audio,
            progress=lambda stage, done, total: print(
                f"PROGRESS stage={stage} done={done}/{total}", flush=True
            ),
        )
    except transcribe.CrispasrError as exc:
        print(f"BACKEND-ERROR {exc}")
        return 2
    except transcribe.TranscriptionOOM as exc:
        print(f"VRAM-OOM {exc}")
        return 1

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(segments, fh)

    monotonic = all(
        segments[i]["t0"] >= segments[i - 1]["t1"] for i in range(1, len(segments))
    )
    ordered = all(s["t1"] >= s["t0"] for s in segments)
    nonempty = sum(1 for s in segments if s["text"].strip())
    speakers = sorted({s["speaker"] for s in segments})
    span = segments[-1]["t1"] if segments else 0.0
    stats = transcribe.LAST_RUN_STATS
    print(f"RESULT segments={len(segments)} nonempty_text={nonempty}")
    print(f"CHECK monotonic_t0_t1={monotonic and ordered} span_end={span:.1f}s")
    print(f"CHECK speaker_labels={speakers}")
    print(f"STATS engine_seconds={stats.get('engine_seconds')} "
          f"vram_peak_mib={stats.get('vram_peak_mib')} "
          f"vram_samples={stats.get('vram_samples')} "
          f"vram_limit_mib={stats.get('vram_limit_mib')}")
    ok = bool(segments) and nonempty == len(segments) and monotonic and ordered
    if stats.get("vram_samples"):
        ok = ok and stats["vram_peak_mib"] < stats["vram_limit_mib"]
    print(f"VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
