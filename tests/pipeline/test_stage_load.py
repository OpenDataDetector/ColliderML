"""Pipeline stage: load data.

Contract: ``colliderml.load("<channel>_pu<N>")`` returns a dict of frames
keyed by table name, requesting the default object tables. The real
public-data load (no token) is a separate `integration` check.
"""

from __future__ import annotations

import pytest

import colliderml

pytestmark = pytest.mark.pipeline


def test_load_contract_returns_named_frames(stub_loader):
    """Mock tier: load() returns frames keyed by table, default tables requested."""
    frames = colliderml.load("ttbar_pu0")
    assert set(frames) == {"tracker_hits", "particles"}
    # default object tables were requested through the cache-ensurance layer
    _, _, requested_tables, dataset_id, _ = stub_loader["ensure"]
    assert requested_tables == list(colliderml.DEFAULT_OBJECT_TABLES)
    assert dataset_id  # a dataset id was resolved


def test_load_parses_channel_and_pileup(stub_loader):
    colliderml.load("higgs_portal_pu200", tables=["particles"], max_events=5)
    channel, pileup, tables, _, _ = stub_loader["ensure"]
    assert channel == "higgs_portal"
    assert pileup == "pu200"
    assert tables == ["particles"]
    assert stub_loader["ensure_bounds"][0] == 5  # max_events threaded through


def test_load_rejects_malformed_dataset_name(stub_loader):
    with pytest.raises(ValueError):
        colliderml.load("not-a-valid-name")


@pytest.mark.integration
def test_load_real_public_data():
    """Real load against the PUBLIC CERN/ColliderML-Release-1 (no token)."""
    frames = colliderml.load("ttbar_pu0", tables=["particles"], max_events=2)
    assert "particles" in frames
    part = frames["particles"]
    # eager polars frame with an event_id column and at least one event
    assert "event_id" in part.columns
    assert part.height >= 1
