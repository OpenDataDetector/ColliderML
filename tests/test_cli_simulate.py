"""Tests for the new v0.4.0 CLI subcommands.

All side-effecting calls are monkey-patched so the tests never run
Docker, contact the backend, or touch the network.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from colliderml import cli as cli_mod
from colliderml.cli import build_parser, main


def _parse(argv: list[str]) -> Any:
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


class _FakeSimulationResult:
    def __init__(self, run_dir: str, remote_request_id: str | None = None) -> None:
        self.run_dir = run_dir
        self.remote_request_id = remote_request_id


def test_simulate_parser_flags_round_trip() -> None:
    args = _parse(
        [
            "simulate",
            "--preset",
            "ttbar-quick",
            "--output",
            "/tmp/out",
            "--seed",
            "7",
            "--local",
            "--quiet",
        ]
    )
    assert args.command == "simulate"
    assert args.preset == "ttbar-quick"
    assert args.output == "/tmp/out"
    assert args.seed == 7
    assert args.local is True
    assert args.quiet is True


def test_simulate_local_invokes_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_simulate(**kwargs: Any) -> _FakeSimulationResult:
        captured.update(kwargs)
        return _FakeSimulationResult(run_dir="/tmp/out/runs/0")

    # cli._cmd_simulate imports colliderml.simulate lazily — patch the
    # simulate symbol on the module it looks up.
    import colliderml.simulate as simulate_pkg

    monkeypatch.setattr(simulate_pkg, "simulate", fake_simulate)

    rc = main(
        [
            "simulate",
            "--preset",
            "ttbar-quick",
            "--local",
            "--output",
            "/tmp/out",
            "--quiet",
        ]
    )
    assert rc == 0
    assert captured["preset"] == "ttbar-quick"
    assert captured["remote"] is False
    assert captured["output_dir"] == "/tmp/out"
    assert captured["quiet"] is True


def test_simulate_remote_invokes_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_simulate(**kwargs: Any) -> _FakeSimulationResult:
        captured.update(kwargs)
        return _FakeSimulationResult(run_dir="/tmp/remote", remote_request_id="req-42")

    import colliderml.simulate as simulate_pkg

    monkeypatch.setattr(simulate_pkg, "simulate", fake_simulate)

    rc = main(["simulate", "--preset", "ttbar-quick", "--remote"])
    assert rc == 0
    assert captured["remote"] is True


def test_simulate_missing_args_returns_error_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_simulate(**kwargs: Any) -> None:
        raise ValueError("needs a preset")

    import colliderml.simulate as simulate_pkg

    monkeypatch.setattr(simulate_pkg, "simulate", raising_simulate)
    rc = main(["simulate", "--quiet"])
    assert rc == 2


# ---------------------------------------------------------------------------
# list-presets
# ---------------------------------------------------------------------------


def test_list_presets_prints_entries(capsys: pytest.CaptureFixture) -> None:
    rc = main(["list-presets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ttbar-quick" in out
    assert "higgs-portal-quick" in out
    assert "channel=" in out


# ---------------------------------------------------------------------------
# balance
# ---------------------------------------------------------------------------


def test_balance_prints_credits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def fake_get_me(*, backend_url: str | None = None) -> Dict[str, Any]:
        return {"hf_username": "alice", "credits": 42.5}

    # Stub out the deferred import by setting the attribute on the subpackage.
    import colliderml.remote as remote_pkg

    monkeypatch.setattr(remote_pkg, "get_me", fake_get_me, raising=False)
    # Also patch the symbol the balance handler will import via `from colliderml.remote import ...`.
    # Since the handler does `from colliderml.remote import balance, get_me`, patching the package
    # attribute is sufficient for `get_me`, but `balance` isn't used by _cmd_balance.
    rc = main(["balance"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "42.50" in out


def test_balance_without_extra_raises_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the `remote` extra being missing by making the import fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "colliderml.remote" or name.startswith("colliderml.remote."):
            raise ImportError("pretend remote extra missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as excinfo:
        main(["balance"])
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class _FakeSnap:
    request_id = "req-xyz"
    state = "running"
    channel = "ttbar"
    events = 10
    pileup = 0
    output_hf_repo = None
    raw: Dict[str, Any] = {}


def test_status_prints_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def fake_status(request_id: str, *, backend_url: str | None = None) -> _FakeSnap:
        assert request_id == "req-xyz"
        return _FakeSnap()

    import colliderml.remote as remote_pkg

    monkeypatch.setattr(remote_pkg, "status", fake_status, raising=False)
    rc = main(["status", "req-xyz"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "req-xyz" in out
    assert "running" in out
    assert "ttbar" in out
