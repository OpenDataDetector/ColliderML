"""Tests for :mod:`colliderml.simulate.presets`.

These tests exercise the bundled presets.yaml via importlib.resources and
verify that the dataclass coercion and error paths do the right thing.
They do NOT touch Docker, the filesystem outside the package, or the
network — they should run in the default ``make test-fast`` gate.
"""

from __future__ import annotations

import pytest

from colliderml.simulate import presets as presets_mod
from colliderml.simulate.presets import Preset, load_presets, resolve_preset


def test_load_presets_returns_nonempty_catalogue() -> None:
    catalogue = load_presets()
    assert len(catalogue) > 0, "bundled presets.yaml should define at least one preset"
    for name, preset in catalogue.items():
        assert isinstance(preset, Preset)
        assert preset.name == name
        assert preset.channel, f"{name} must have a non-empty channel"
        assert preset.events > 0, f"{name} must request > 0 events"
        assert preset.pileup >= 0, f"{name} must have non-negative pileup"


def test_bundled_presets_include_ttbar_quick() -> None:
    # ttbar-quick is a contract with the CLI docs and quickstart examples.
    catalogue = load_presets()
    assert "ttbar-quick" in catalogue
    preset = catalogue["ttbar-quick"]
    assert preset.channel == "ttbar"
    assert preset.events == 10
    assert preset.pileup == 0


def test_bundled_presets_include_higgs_portal_quick() -> None:
    catalogue = load_presets()
    assert "higgs-portal-quick" in catalogue
    preset = catalogue["higgs-portal-quick"]
    assert preset.channel == "higgs_portal"


def test_resolve_preset_happy_path() -> None:
    preset = resolve_preset("ttbar-quick")
    assert preset.name == "ttbar-quick"
    assert preset.events == 10


def test_resolve_preset_unknown_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_preset("not-a-real-preset")
    msg = str(excinfo.value)
    assert "not-a-real-preset" in msg
    # The error message should list the available presets so users can self-correct.
    assert "ttbar-quick" in msg


def test_preset_as_dict_roundtrips_fields() -> None:
    preset = Preset(
        name="demo",
        channel="ttbar",
        events=5,
        pileup=2,
        description="demo preset",
    )
    d = preset.as_dict()
    assert d == {
        "name": "demo",
        "channel": "ttbar",
        "events": 5,
        "pileup": 2,
        "description": "demo preset",
    }


def test_load_presets_rejects_malformed_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkey-patch the private reader to return an invalid structure.
    def _bad() -> dict:
        return {"ttbar-quick": "this-should-be-a-dict"}

    monkeypatch.setattr(presets_mod, "_read_presets_file", _bad)
    with pytest.raises(ValueError, match="must be a mapping"):
        load_presets()


def test_load_presets_rejects_missing_required_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_channel() -> dict:
        return {"broken": {"events": 10}}

    monkeypatch.setattr(presets_mod, "_read_presets_file", _missing_channel)
    with pytest.raises(ValueError, match="missing required field 'channel'"):
        load_presets()
