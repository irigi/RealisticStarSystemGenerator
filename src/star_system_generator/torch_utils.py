"""Torch import helper with a clear dependency error."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training and sampling. Install project dependencies with:\n"
            '  pip install -e ".[dev]"\n'
            "or install torch separately for your platform."
        ) from exc
    return torch

