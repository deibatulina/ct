import json


def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
