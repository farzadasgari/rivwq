import pytest

from rivwq.gee import session

VARIABLE = "EARTHENGINE_PROJECT"
TRACKED_VARIABLES = (VARIABLE, "OTHER")


@pytest.fixture(autouse=True)
def restore_environment(monkeypatch):
    for name in TRACKED_VARIABLES:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "gee.yaml"
    path.write_text(f"project_environment_variable: {VARIABLE}\n", encoding="utf-8")
    return path


def write_env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_project_from_env_file_when_unset(tmp_path, config_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    env_path = write_env(tmp_path, f"{VARIABLE}=from-file\n")
    assert session.resolve_project(env_path, config_path) == "from-file"


def test_existing_environment_variable_wins(tmp_path, config_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, "from-shell")
    env_path = write_env(tmp_path, f"{VARIABLE}=from-file\n")
    assert session.resolve_project(env_path, config_path) == "from-shell"


def test_missing_everywhere_raises(tmp_path, config_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    env_path = write_env(tmp_path, "OTHER=1\n")
    with pytest.raises(RuntimeError, match=VARIABLE):
        session.resolve_project(env_path, config_path)


def test_empty_value_raises(tmp_path, config_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    env_path = write_env(tmp_path, f"{VARIABLE}=\n")
    with pytest.raises(RuntimeError, match=VARIABLE):
        session.resolve_project(env_path, config_path)


def test_missing_env_file_falls_back_to_environment(tmp_path, config_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, "from-shell")
    assert session.resolve_project(tmp_path / "absent.env", config_path) == "from-shell"


def test_initialize_passes_project_to_earth_engine(monkeypatch):
    import ee

    received = {}
    monkeypatch.setattr(ee, "Initialize", lambda **kwargs: received.update(kwargs))
    monkeypatch.setenv(VARIABLE, "offline-project")
    assert session.initialize() == "offline-project"
    assert received == {"project": "offline-project"}
