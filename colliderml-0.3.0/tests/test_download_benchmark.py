"""Unit tests for benchmarks/download_benchmark.py (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_download_benchmark_writes_json_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: fake HF repo listing and fake downloads.
    from benchmarks.download_benchmark import main

    def fake_list_repo_files(*, repo_id, repo_type, revision=None):
        assert repo_type == "dataset"
        return [
            "README.md",
            "data/ttbar_pu0_particles/train-00000-of-01000.parquet",
            "data/ttbar_pu0_tracker_hits/train-00000-of-01000.parquet",
            "data/ttbar_pu0_calo_hits/train-00000-of-01000.parquet",
            "data/ttbar_pu0_tracks/train-00000-of-01000.parquet",
        ]

    def fake_hf_hub_download(
        *,
        repo_id,
        repo_type,
        revision=None,
        filename,
        local_dir,
        local_dir_use_symlinks=False,
        force_download=False,
    ):
        # Write a tiny file at the expected location.
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PARQUET")
        return str(p)

    monkeypatch.setattr("benchmarks.download_benchmark.list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("benchmarks.download_benchmark.hf_hub_download", fake_hf_hub_download)

    # Make benchmark write results under tmp and use permissive thresholds.
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(
        "configs:\n"
        "  - name: ttbar_pu0_particles\n"
        "    max_initial_seconds: 0.000001\n"
        "    max_cached_seconds: 0.000001\n",
        encoding="utf-8",
    )

    monkeypatch.chdir("/home/murnanedaniel/Research/FoundationModels/ColliderML/colliderml")

    # Act
    rc = main(
        [
            "--configs",
            "ttbar_pu0_particles",
            "--out",
            str(tmp_path / "out"),
            "--thresholds",
            str(thresholds),
            "--warn-only",
        ]
    )

    # Assert: warn-only must never fail
    assert rc == 0

    results_dir = Path("benchmarks/results")
    assert results_dir.exists()
    results = sorted(results_dir.glob("download_benchmark_*.json"))
    assert results, "expected a benchmark JSON to be written"
    payload = json.loads(results[-1].read_text(encoding="utf-8"))
    assert payload["results"][0]["config"] == "ttbar_pu0_particles"


