from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("results_dir", "figures_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
