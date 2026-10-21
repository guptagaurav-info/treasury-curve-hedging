"""Load the YAML config; resolve paths relative to the repo root."""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["raw_dir"] = ROOT / cfg["raw_dir"]
    cfg["out_dir"] = ROOT / cfg["out_dir"]
    return cfg
