import runpy
from pathlib import Path

import pytest
import yaml

import rivwq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ["study", "sites", "gee", "features", "models", "validation"]
PIPELINE_STEPS = sorted((ROOT / "pipeline").glob("[0-9][0-9]_*.py"))


def test_package_imports():
    assert rivwq.__name__ == "rivwq"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_file_is_a_mapping(name):
    content = yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    assert isinstance(content, dict)


def test_out_of_time_cutoff_is_configured():
    content = yaml.safe_load((ROOT / "config" / "study.yaml").read_text(encoding="utf-8"))
    assert content["dates"]["out_of_time_start"] == "2025-01-01"


def test_pipeline_has_ten_numbered_steps():
    assert [path.name[:2] for path in PIPELINE_STEPS] == [f"{n:02d}" for n in range(1, 11)]


@pytest.mark.parametrize("path", PIPELINE_STEPS, ids=lambda path: path.name)
def test_pipeline_step_is_a_stub(path):
    with pytest.raises(NotImplementedError):
        runpy.run_path(str(path), run_name="__main__")
