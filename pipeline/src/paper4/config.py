from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    project_candidate = (PROJECT_ROOT / p).resolve()
    return project_candidate


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["data"]["root"] = str(_resolve(cfg["data"]["root"]))
    cfg["project"]["output_dir"] = str(_resolve(cfg["project"]["output_dir"]))
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    return out


def dataset_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["root"])
