"""Environment capture and CPU pinning.

Every result file embeds `collect()` so each number can be traced to the
software versions, hardware, power plan and git commit that produced it.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              cwd=REPO_ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"])
    # Results and generated files may change during a run; only code counts as "dirty".
    status = _run(["git", "status", "--porcelain", "--", "src", "experiments",
                   "scripts", "requirements.txt"])
    return {"commit": commit or None, "code_dirty": bool(status)}


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    return platform.processor()


def cpu_sets() -> list[dict]:
    """Logical processors with their core index and efficiency class (Windows).

    On hybrid Intel CPUs the P-cores report a higher EfficiencyClass than the
    E-cores. Returns [] when the API is unavailable.
    """
    if sys.platform != "win32":
        return []
    k32 = ctypes.windll.kernel32
    needed = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    buf = (ctypes.c_ubyte * needed.value)()
    if not k32.GetSystemCpuSetInformation(buf, needed, ctypes.byref(needed), None, 0):
        return []
    raw = bytes(buf)
    out, off = [], 0
    while off < len(raw):
        size = int.from_bytes(raw[off:off + 4], "little")
        out.append({
            "logical": raw[off + 14],
            "core": raw[off + 15],
            "efficiency_class": raw[off + 18],
        })
        off += size
    return out


def performance_cpus() -> list[int]:
    """One logical processor per performance core (first SMT sibling)."""
    sets = cpu_sets()
    if not sets:
        return list(range(os.cpu_count() or 1))
    top = max(s["efficiency_class"] for s in sets)
    seen, cpus = set(), []
    for s in sorted(sets, key=lambda s: s["logical"]):
        if s["efficiency_class"] == top and s["core"] not in seen:
            seen.add(s["core"])
            cpus.append(s["logical"])
    return cpus


def pin_process(cpus: list[int]) -> None:
    """Restrict this process (all threads) to the given logical processors."""
    mask = 0
    for c in cpus:
        mask |= 1 << c
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise OSError("SetProcessAffinityMask failed")
    else:
        os.sched_setaffinity(0, set(cpus))


def power_plan() -> str:
    if sys.platform != "win32":
        return ""
    line = _run(["powercfg", "/getactivescheme"])
    return line[line.find("(") + 1:line.rfind(")")] if "(" in line else line


def gpu_info() -> dict:
    q = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
              "--format=csv,noheader"])
    if not q:
        return {}
    name, driver, mem = [p.strip() for p in q.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "memory": mem}


def collect() -> dict:
    import numpy
    import torch
    info = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "quantized_engines": list(torch.backends.quantized.supported_engines),
        "quantized_engine": torch.backends.quantized.engine,
        "torch_threads": torch.get_num_threads(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "performance_cpus": performance_cpus(),
        "gpu": gpu_info(),
        "power_plan": power_plan(),
        "git": git_info(),
    }
    for mod in ("torchvision", "onnx", "onnxruntime", "onnxscript"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    return info
