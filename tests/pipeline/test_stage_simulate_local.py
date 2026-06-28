"""Pipeline stage: simulate locally.

Mock tier asserts the dispatch/validation contract; the `sim` smoke runs a
real 2-event pipeline (skips without a container runtime). The RED test pins
the half-removed ``single_muon`` channel.
"""

from __future__ import annotations

import pytest

from colliderml.simulate.pipeline import CHANNEL_STAGES

pytestmark = pytest.mark.pipeline


def test_channel_stages_known():
    """The two fully-supported channels expose pipeline stages."""
    assert "ttbar" in CHANNEL_STAGES
    assert "higgs_portal" in CHANNEL_STAGES
    assert len(CHANNEL_STAGES["ttbar"]) >= 3


def test_simulate_requires_channel_or_preset():
    """simulate() with neither preset nor channel must fail fast (pre-Docker)."""
    from colliderml.simulate.api import simulate

    with pytest.raises((ValueError, TypeError)):
        simulate()


def test_single_muon_not_simulatable():
    """RED: single_muon's preset was removed and its Pythia stage-routing is
    wrong (it needs an ACTS particle-gun stage), so it must NOT be offered as a
    simulatable channel until that exists. Today it is still in CHANNEL_STAGES.
    """
    assert "single_muon" not in CHANNEL_STAGES, (
        "single_muon is still a simulatable channel but its preset was removed "
        "and it is routed through Pythia (wrong); remove it from CHANNEL_STAGES "
        "(simulate/pipeline.py) or add a proper particle-gun stage."
    )


@pytest.mark.sim
@pytest.mark.slow
def test_simulate_two_events_smoke(docker_available, tmp_path):
    """Real 2-event higgs_portal run (needs Docker/Podman + the ODD image)."""
    from colliderml.simulate.api import simulate

    result = simulate(
        channel="higgs_portal",
        events=2,
        pileup=0,
        output_dir=tmp_path,
        runtime=docker_available,
        quiet=True,
    )
    assert result.channel == "higgs_portal"
    assert any(p.suffix == ".parquet" for p in result.list_files())
