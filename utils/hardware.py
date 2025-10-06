from __future__ import annotations

from typing import Optional

try:  # Torch is optional; callers fall back gracefully if unavailable.
    import torch
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None


def select_best_device(requested: Optional[str] = None, allow_multi_gpu: bool = True) -> str:
    """Return the optimal device string for Ultralytics/Torch workloads."""
    if requested:
        return requested

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if allow_multi_gpu and device_count > 1:
            return ",".join(str(i) for i in range(device_count))
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    if hasattr(torch.backends, "cpu") and getattr(torch.backends.cpu, "is_available", lambda: False)():
        return "cpu"

    return "cpu"
