"""Tests for optional W&B tracking utilities."""

# ruff: noqa: S101

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

import pytest

from src import utils

if TYPE_CHECKING:
    from pathlib import Path


def test_wandb_defaults_are_disabled_and_run_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify W&B defaults are safe and scoped under the PPO run."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = utils.wandb.WandbTrackingSettings()

    assert settings.mode == "disabled"
    assert settings.project == "drone-rl-llm-curriculum"
    assert utils.wandb.default_wandb_dir() == tmp_path / "runs" / "ppo_tracking_smoke" / "wandb"


def test_disabled_wandb_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify disabled mode is a no-op even when wandb cannot be imported."""
    original_import = builtins.__import__

    def fail_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            message = "wandb should not be imported when disabled"
            raise AssertionError(message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_wandb_import)

    run = utils.wandb.start_wandb_run(
        utils.wandb.WandbTrackingSettings(mode="disabled"),
        config={"total_timesteps": 1},
    )

    assert run is None


def test_wandb_tags_parse_comma_separated_values() -> None:
    """Verify CLI tag strings are normalized before W&B init."""
    assert utils.wandb.parse_wandb_tags(" smoke, docker ,,offline ") == ("smoke", "docker", "offline")
    assert utils.wandb.parse_wandb_tags(None) == ()
    assert utils.wandb.parse_wandb_tags([" smoke ", "", "docker"]) == ("smoke", "docker")


def test_online_wandb_without_key_fails_before_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify online mode cannot hang on login when credentials are absent."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "missing-home"))

    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        utils.wandb.start_wandb_run(
            utils.wandb.WandbTrackingSettings(mode="online"),
            config={},
        )


def test_wandb_key_can_be_loaded_from_home_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the optional home key file populates the environment without printing it."""
    key_path = tmp_path / "wandb_key.txt"
    key_path.write_text("secret-test-key\n", encoding="utf-8")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    utils.wandb._load_wandb_api_key_from_home_file()  # noqa: SLF001

    assert "WANDB_API_KEY" in __import__("os").environ
    assert __import__("os").environ["WANDB_API_KEY"] == "secret-test-key"
