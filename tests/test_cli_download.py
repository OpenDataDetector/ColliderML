"""Unit tests for the ColliderML CLI download/list-configs commands.

These tests mock all HuggingFace network calls for CI stability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "colliderml_cache"
    d.mkdir()
    monkeypatch.setenv("COLLIDERML_DATA_DIR", str(d))
    return d


def test_cli_download_writes_metadata_and_layout(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Import inside test so env var is set first.
    from colliderml.cli import main

    calls = {"snapshot": [], "list_repo_files": []}

    def fake_list_repo_files(*, repo_id, repo_type, revision=None):
        calls["list_repo_files"].append((repo_id, repo_type, revision))
        # Minimal file list for one config/split.
        return [
            "README.md",
            "data/ttbar_pu0_particles/train-00000-of-00001.parquet",
            "data/ttbar_pu0_particles/train-00001-of-00001.parquet",
        ]

    def fake_snapshot_download(
        *,
        repo_id,
        repo_type,
        revision=None,
        local_dir=None,
        local_dir_use_symlinks=False,
        allow_patterns=None,
        force_download=False,
    ):
        calls["snapshot"].append(
            (repo_id, repo_type, revision, local_dir, tuple(allow_patterns or []), force_download)
        )
        # Simulate that parquet shards exist on disk in the chosen local_dir.
        assert local_dir is not None
        base = Path(local_dir)
        for fp in fake_list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision):
            if fp.endswith(".parquet") and fp.startswith("data/"):
                p = base / fp
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"PARQUET")
        return str(base)

    # Patch in the module under test (core.hf_download uses these names).
    monkeypatch.setattr("colliderml.core.hf_download.list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("colliderml.core.hf_download.snapshot_download", fake_snapshot_download)

    rc = main(
        [
            "download",
            "--dataset-id",
            "CERN/ColliderML-Release-1",
            "--config",
            "ttbar_pu0_particles",
            "--split",
            "train",
        ]
    )
    assert rc == 0

    # Validate metadata written.
    dataset_dir = tmp_data_dir / "CERN__ColliderML-Release-1" / "ttbar_pu0_particles"
    meta_path = dataset_dir / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dataset_id"] == "CERN/ColliderML-Release-1"
    assert meta["config"] == "ttbar_pu0_particles"
    assert meta["split"] == "train"
    assert any(s.endswith(".parquet") for s in meta["shards"])

    # Validate the shard file exists.
    shard0 = dataset_dir / "data/ttbar_pu0_particles/train-00000-of-00001.parquet"
    assert shard0.exists()

    # Validate we called snapshot_download with the right allow_patterns.
    assert calls["snapshot"], "expected snapshot_download call"
    _, _, _, local_dir, allow_patterns, _ = calls["snapshot"][0]
    assert "data/ttbar_pu0_particles/train-*.parquet" in allow_patterns
    assert local_dir == str(dataset_dir)


def test_cli_list_configs_local(tmp_data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from colliderml.cli import main

    # Create a fake downloaded config dir.
    base = tmp_data_dir / "CERN__ColliderML-Release-1"
    (base / "ttbar_pu0_particles").mkdir(parents=True)
    (base / "ggf_pu0_tracks").mkdir(parents=True)

    rc = main(["list-configs", "--dataset-id", "CERN/ColliderML-Release-1"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["ggf_pu0_tracks", "ttbar_pu0_particles"]


