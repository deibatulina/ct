from pathlib import Path

import yaml

import config


def load_yaml_config(path):
    """
    Читает YAML-файл эксперимента и возвращает обычный словарь.
    """
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return payload
