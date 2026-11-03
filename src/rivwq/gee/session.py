from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = REPO_ROOT / "config" / "gee.yaml"


def project_variable_name(config_path: Path = CONFIG_PATH) -> str:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["project_environment_variable"]


def resolve_project(env_path: Path = ENV_PATH, config_path: Path = CONFIG_PATH) -> str:
    load_dotenv(env_path, override=False)
    name = project_variable_name(config_path)
    project = os.environ.get(name, "").strip()
    if not project:
        raise RuntimeError(f"{name} is not set in the environment or in {env_path}")
    return project


def initialize() -> str:
    import ee

    project = resolve_project()
    ee.Initialize(project=project)
    return project
