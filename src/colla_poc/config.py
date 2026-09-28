from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .models import Config

DEFAULT = Path(__file__).resolve().parents[2] / 'config' / 'default.yaml'

def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('Config must be a YAML mapping')
    return Config.model_validate(data).model_dump()
