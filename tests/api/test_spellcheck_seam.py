"""Card t_72405565 AC1/AC8: the spell-check seam must run in the SHIPPED
service configuration (no SPELL_CHECK set at all) and be opt-OUT, with the
API gate and the module default agreeing on one shared enablement rule.

Before the fix the API hard-gated `os.environ.get("SPELL_CHECK") == "1"`, so
with the shipped compose files (which set no SPELL_CHECK) the pass never ran,
while the module itself defaulted enabled — a two-way mismatch also recorded
in CONFIG.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.spellcheck import spell_check_enabled

API_MAIN = Path(__file__).resolve().parents[2] / "api" / "main.py"


@pytest.fixture(autouse=True)
def _clean_spell_check_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPELL_CHECK", raising=False)


def test_default_shipped_config_enables_the_pass() -> None:
    """AC1/AC8: SPELL_CHECK unset (the shipped compose configuration) means ON."""
    assert spell_check_enabled() is True


@pytest.mark.parametrize("value", ["0", "off", "OFF", "false", "False", "no"])
def test_opt_out_values_disable_the_pass(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """AC8: SPELL_CHECK=0/off disables the pass."""
    monkeypatch.setenv("SPELL_CHECK", value)
    assert spell_check_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_opt_in_values_still_enable_the_pass(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SPELL_CHECK", value)
    assert spell_check_enabled() is True


def test_api_gate_agrees_with_module_default() -> None:
    """AC1: the API must not keep a bespoke `== "1"` gate; it consults the
    shared spell_check_enabled() so both layers use identical semantics."""
    source = API_MAIN.read_text(encoding="utf-8")
    assert 'os.environ.get("SPELL_CHECK") == "1"' not in source
    assert "spell_check_enabled()" in source


def test_vocab_is_tracked_and_baked_into_the_image() -> None:
    """AC3: the vocabulary is in the repo, COPYed into the runtime image
    layer, and the module default is an absolute path beside it."""
    root = API_MAIN.parent.parent
    assert (root / "vocab.txt").exists()
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "vocab.txt"],
        cwd=root, capture_output=True, text=True,
    )
    assert tracked.returncode == 0, "vocab.txt must be tracked in the repo"

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    runtime_section = dockerfile.split("AS runtime", 1)[1]
    assert "COPY vocab.txt" in runtime_section

    from pipeline.spellcheck import DEFAULT_VOCAB_PATH

    assert DEFAULT_VOCAB_PATH.is_absolute()
