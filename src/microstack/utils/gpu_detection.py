"""GPU detection and configuration utilities using PyTorch."""

from typing import Literal, Optional, Dict
import torch


def detect_gpu_capabilities() -> Dict[str, any]:
    """
    Detect available GPU hardware using PyTorch.

    Returns:
        Dictionary with GPU capabilities
    """
    cuda_available = torch.cuda.is_available()
    cuda_devices = 0
    cuda_device_names = []

    if cuda_available:
        cuda_devices = torch.cuda.device_count()
        cuda_device_names = [torch.cuda.get_device_name(i) for i in range(cuda_devices)]

    capabilities = {
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "cuda_device_names": cuda_device_names,
        "recommended_backend": "cuda" if cuda_available else "cpu",
    }

    return capabilities


def get_torch_device(backend: Optional[Literal["cuda", "cpu"]] = None) -> torch.device:
    """
    Get a torch.device object for the specified backend.

    Args:
        backend: Desired backend (cuda, cpu). Auto-detect if None.

    Returns:
        torch.device object
    """
    if backend is None:
        backend = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(backend)


def get_gpu_memory_info(backend: Literal["cuda", "cpu"] = "cuda") -> dict[str, float]:
    """
    Get GPU memory information using PyTorch.

    Args:
        backend: GPU backend to query

    Returns:
        Dictionary with memory info (total_gb, free_gb, used_gb)
    """
    if backend == "cuda" and torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            return {"total_gb": total, "free_gb": free, "used_gb": allocated}
        except Exception:
            return {"total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0}
    else:
        return {"total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0}
