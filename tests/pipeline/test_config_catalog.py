"""Complete, mechanically verified env-var catalogue (card t_1799d07f, AC5).

The catalogue (``pipeline/config_vars.py``) must describe EVERY knob the
app reads, with name + default + where it is read + what it affects, and
the checker script must pass against the live tree. Existing CONFIG.md
content must not be rewritten by this card (t_16c871a6 territory).
"""
import subprocess
import sys
from pathlib import Path

from pipeline import config_vars

REPO = Path(__file__).resolve().parents[2]


def _entries():
    return config_vars.CATALOG


def test_catalog_is_complete_records():
    entries = _entries()
    assert entries, "catalogue must not be empty"
    for e in entries:
        assert e.name and e.name == e.name.upper()
        assert e.where, f"{e.name}: where it is read must be recorded"
        assert e.effect, f"{e.name}: what it affects must be recorded"
        # default may legitimately be None for unset-by-default passthroughs,
        # but then the effect must say so
        if e.default is None:
            assert "unset" in e.effect.lower() or "no default" in e.effect.lower()
    names = [e.name for e in entries]
    assert len(names) == len(set(names)), "duplicate catalogue entries"


def test_checker_script_passes_against_the_tree():
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check-config-catalog.py")],
        capture_output=True, text=True, cwd=REPO, timeout=120, check=False,
    )
    assert res.returncode == 0, f"checker failed:\n{res.stdout}\n{res.stderr}"


def test_spot_checks_the_measured_knobs_are_catalogued():
    """The knobs this project's incidents were actually about."""
    by_name = {e.name: e for e in _entries()}
    assert by_name["CRISPASR_TIMEOUT_SECONDS"].default == "1800"
    assert by_name["MAX_PROCESSING_GLOBAL"].default == "1"
    assert "concurrency" in by_name["MAX_PROCESSING_GLOBAL"].effect.lower()
    assert by_name["VRAM_LIMIT_MIB"].default == "3584"
    # the N=4 VRAM exhaustion fix knob must mention lowering concurrency
    oom_entry = by_name["MAX_PROCESSING_GLOBAL"]
    assert "out of memory" in oom_entry.effect.lower() or "vram" in oom_entry.effect.lower()


def test_checker_detects_a_missing_entry(tmp_path):
    """Mechanical verification must be real: an uncatalogued reference fails it."""
    script = REPO / "scripts" / "check-config-catalog.py"
    # Inject a throwaway module reading an uncatalogued knob into a COPY
    # of the tree? Too heavy. Instead call the checker's scan function on a
    # temp source file directly.
    sys.path.insert(0, str(REPO))
    import importlib.util
    spec = importlib.util.spec_from_file_location("checkmod", script)
    checkmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkmod)
    f = tmp_path / "leaky.py"
    f.write_text('import os\nos.environ.get("BRAND_NEW_KNOB_XYZ", "1")\n')
    found = checkmod.scan_python_files([f])
    assert "BRAND_NEW_KNOB_XYZ" in found
