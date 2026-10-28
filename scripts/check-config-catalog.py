#!/usr/bin/env python3
"""Mechanical config-catalogue verifier (card t_1799d07f, AC5).

The catalogue (pipeline/config_vars.py) must name EVERY environment knob
the app itself reads — no more, no less — and every default it records
must match the literal the code actually falls back to.

Scanned surfaces (the app's own interface; test harnesses and one-off
ops scripts are deliberately NOT app surfaces):
  api/*.py, pipeline/*.py         — Python environ reads, direct or via
                                    the env-named helper functions
  scripts/docker-entrypoint.sh    — container entrypoint
  scripts/sherpa-diarization-wrapper.sh

Exit 0 with a summary on success; exit 1 listing every discrepancy
(uncatalogued knob, catalogue entry with no code reference, defaulted
mismatch) otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.config_vars import CATALOG

# name -> [(relpath, default-literal-or-None)]
_PY_PATTERNS = [
    re.compile(r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]*)"\s*(?:,\s*([^)]*?))?\)'),
    re.compile(r'os\.environ\[\s*"([A-Z][A-Z0-9_]*)"\s*\]'),
    re.compile(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"\s*(?:,\s*([^)]*?))?\)'),
    # helper functions whose first argument is the literal knob name —
    # covers _env_str/_env_float/_env_int/_env_enabled/_validated_env_name/
    # _configured_patterns and any future *_env*/*env* reader helper.
    re.compile(r'\b_[a-z_]*env[a-z_]*\(\s*"([A-Z][A-Z0-9_]*)"\s*(?:,\s*([^)]*?))?\)'),
    re.compile(r'\b_configured_patterns\(\s*"([A-Z][A-Z0-9_]*)"\s*(?:,\s*([^)]*?))?\)'),
]
_SHELL_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")

# Shell-side variables that are script-local computations rather than
# operator-set knobs. The wrapper/env idiom `X="${X:-default}"` counts as
# a knob (SHERPA_REAL included below); currently there are no exclusions.
SHELL_LOCAL_VARS = frozenset()

# Python module-level constants used as default expressions; resolved for
# literal comparison when a reference falls back to one (e.g.
# DEFAULT_TIMEOUT_SECONDS).
_CONST = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$")
_QUOTED = re.compile(r'^["\'](.*)["\']$')


def scan_python_files(paths):
    found = {}
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in _PY_PATTERNS:
            for m in pat.finditer(text):
                default = (m.group(2) if m.re.groups >= 2 and m.group(2) else "")
                name = m.group(1)
                found.setdefault(name, []).append((p.name, default.strip() or None))
    return found


def scan_shell_files(paths):
    found = {}
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in _SHELL_PATTERN.finditer(text):
            name, default = m.group(1), m.group(2)
            if name in SHELL_LOCAL_VARS:
                continue
            found.setdefault(name, []).append((p.name, default))
    return found


def _module_constants(paths):
    consts = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _CONST.match(line)
            if m:
                consts.setdefault(m.group(1), m.group(2).strip())
    return consts


def _norm_default(raw):
    """Normalise a code-side default expression to comparable text."""
    if raw is None:
        return None
    raw = raw.strip()
    m = _QUOTED.match(raw)
    if m:
        raw = m.group(1)
    return raw.strip().strip('"').strip("'")


_SCALARISH = re.compile(r'^[\w .+*/()\-"]+$')


def _verifiable_default(raw):
    """Keep only defaults we can compare as text/number.

    Drop: empty shell defaults (${X:-}), the logging placeholder "unset",
    keyword defaults (default=True), list/tuple defaults, and defaults
    wrapped in non-literal calls other than str(CONST)."""
    if raw is None:
        return None
    s = raw.strip()
    if s in ("", "unset") or s.startswith("default="):
        return None
    if s.startswith(("(", "[", "{", "f\"", "f'")):
        return None
    inner = re.fullmatch(r"str\(([A-Z][A-Z0-9_]*)\)", s)
    if inner:
        return inner.group(1) + "__strwrapped__"
    return s


def check(root: Path = REPO, verbose: bool = True) -> int:
    py_files = sorted((root / "api").glob("*.py")) + sorted((root / "pipeline").glob("*.py"))
    sh_files = [root / "scripts" / "docker-entrypoint.sh",
                root / "scripts" / "sherpa-diarization-wrapper.sh"]
    sh_files = [p for p in sh_files if p.exists()]

    refs = scan_python_files(py_files)
    for name, locs in scan_shell_files(sh_files).items():
        refs.setdefault(name, []).extend(locs)
    consts = _module_constants(py_files)

    catalog = {v.name: v for v in CATALOG}
    problems = []

    for name in sorted(set(refs) - set(catalog)):
        problems.append(f"uncatalogued knob: {name} (read in "
                        f"{', '.join(sorted({f for f, _ in refs[name]}))})")
    for name in sorted(set(catalog) - set(refs)):
        problems.append(f"catalogue entry with no code reference: {name}")

    for name, entry in sorted(catalog.items()):
        if name not in refs:
            continue
        resolved = set()
        for _, d in refs[name]:
            d = _verifiable_default(d)
            if d is None:
                continue
            wrapped = d.endswith("__strwrapped__")
            if wrapped:
                d = d[: -len("__strwrapped__")]
            if d in consts:                      # DEFAULT_X constant
                d = consts[d]
                if not _SCALARISH.fullmatch(d) or re.search(r"[({[]", d):
                    continue                     # non-scalar default expr
                if d[:1] == "f" and d[1:2] in ['"', "'"]:
                    continue                     # f-string default, computed
            elif re.search(r"\(|\bTrue\b|\bFalse\b", d):
                continue                         # expression we cannot verify
            elif re.fullmatch(r"[A-Z][A-Z0-9_]*", d):
                continue                         # unresolvable constant name
            d = _norm_default(d)
            if d is None:
                continue
            if re.fullmatch(r"[\d\s.*+()]+", d):
                d = repr(float(eval(d, {"__builtins__": set()}, {})))
            if wrapped:                          # str(CONST): numeric spelling ok
                resolved.add(d)
                try:
                    resolved.add(str(int(float(d))))
                except ValueError:
                    pass
            else:
                resolved.add(d)
        if resolved:
            want = str(entry.default)
            def _as_num(s):
                try:
                    return float(s)
                except ValueError:
                    return None
            ok = any(
                s == want or (_as_num(s) is not None and _as_num(s) == _as_num(want))
                for s in resolved
            )
            if not ok:
                problems.append(
                    f"default mismatch {name}: catalogue says {want!r}, "
                    f"code falls back to {sorted(resolved)!r}")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        print(f"\n{len(problems)} problem(s); {len(catalog)} catalogue entries.")
        return 1
    if verbose:
        print(f"OK: {len(catalog)} env knobs catalogued, all matched to code "
              f"({len(refs)} reference names across "
              f"{len(py_files) + len(sh_files)} files).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
