"""Small utilities for reproducible grid sweeps."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def grid_product(grid: Mapping[str, Iterable[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.detach().cpu().item())
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    with path.open("a") as f:
        f.write(json.dumps({k: convert(v) for k, v in row.items()}, sort_keys=True) + "\n")
