from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "config"


@lru_cache
def load_agents_config() -> dict[str, dict[str, Any]]:
    return _load_yaml("agents.yaml")


@lru_cache
def load_tasks_config() -> dict[str, dict[str, Any]]:
    return _load_yaml("tasks.yaml")


@lru_cache
def load_route_agents_config() -> dict[str, dict[str, Any]]:
    return _load_yaml("route_agents.yaml")


@lru_cache
def load_route_tasks_config() -> dict[str, dict[str, Any]]:
    return _load_yaml("route_tasks.yaml")


def agent_config(name: str) -> dict[str, Any]:
    return dict(load_agents_config()[name])


def task_config(name: str) -> dict[str, Any]:
    return dict(load_tasks_config()[name])


def crewai_task_config(name: str) -> dict[str, Any]:
    config = task_config(name)
    config.pop("agent", None)
    return config


def route_agent_config(name: str) -> dict[str, Any]:
    return dict(load_route_agents_config()[name])


def route_task_config(name: str) -> dict[str, Any]:
    return dict(load_route_tasks_config()[name])


def crewai_route_task_config(name: str) -> dict[str, Any]:
    config = route_task_config(name)
    config.pop("agent", None)
    return config


def _load_yaml(filename: str) -> dict[str, dict[str, Any]]:
    path = CONFIG_DIR / filename
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return _normalize_config(loaded)


def _normalize_config(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {key: _normalize_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_config(item) for item in value]
    return value
