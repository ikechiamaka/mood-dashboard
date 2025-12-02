import json
import os
from typing import List, Dict, Any


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _path(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def load_json_list(filename: str) -> List[Dict[str, Any]]:
    path = _path(filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_json_list(filename: str, data: List[Dict[str, Any]]) -> None:
    path = _path(filename)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

