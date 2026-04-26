import tempfile
from pathlib import Path

import yaml

from src.identification.config import load_config


def test_load_config_from_path():
    data = {
        "models": {"teacher": "test-teacher"},
        "generation": {"temperature": 1.0},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp_path = f.name

    config = load_config(tmp_path)
    assert config["models"]["teacher"] == "test-teacher"
    assert config["generation"]["temperature"] == 1.0
    Path(tmp_path).unlink()


def test_load_default_config():
    config = load_config()
    assert "models" in config
    assert "teacher" in config["models"]
    assert "generation" in config
    assert "identification" in config
