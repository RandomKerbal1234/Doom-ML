#!/usr/bin/env python3
"""Detect the installed GPU, explain its driver requirements, and pick tuning.

The training policy in this project is tiny, so the CPU simulation is normally
the fastest choice.  This module still gives the optional PyTorch policy
backend a reliable way to find an accelerator and use settings appropriate for
its generation.  Run it directly for a human-readable report::

    python3 training/gpu.py

It has no third-party dependencies.  PyTorch is imported only when it is
installed and a GPU-backed compute plan is requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run(command: List[str], timeout: float = 5.0) -> Optional[str]:
    """Run a read-only probe, returning stdout or None when it is unavailable."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 and not result.stdout.strip():
        return None
    return result.stdout.strip()


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _version_tuple(value: Any) -> Tuple[int, ...]:
    parts = re.findall(r"\d+", str(value or ""))
    return tuple(int(part) for part in parts[:4])


def _version_at_least(installed: Any, minimum: Any) -> Optional[bool]:
    have = _version_tuple(installed)
    need = _version_tuple(minimum)
    if not have or not need:
        return None
    width = max(len(have), len(need))
    return have + (0,) * (width - len(have)) >= need + (0,) * (width - len(need))


def _driver_status(installed: Any, minimum: Any) -> str:
    comparison = _version_at_least(installed, minimum)
    if comparison is None:
        return "unknown"
    return "ok" if comparison else "update required"


def _normalized_dtype(value: str) -> str:
    value = (value or "auto").strip().lower()
    if value in ("auto", ""):
        return "auto"
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp8": "float8",
        "float8": "float8",
    }
    return aliases.get(value, value)


def _normalized_mode(value: str) -> str:
    value = (value or "off").strip().lower()
    return value if value in ("off", "auto", "force") else "off"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Tuning:
    """Generation-appropriate defaults for a GPU-backed policy backend."""

    dtype: str = "float32"
    batch_size: int = 1
    tf32: bool = False
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    matmul_precision: str = "highest"
    note: str = ""


@dataclass
class GpuInfo:
    vendor: str
    name: str
    backend: str
    architecture: str = "unknown"
    generation: str = "unknown"
    compute_capability: Optional[str] = None
    driver: str = "unknown"
    driver_version: str = "unknown"
    minimum_driver: str = "unknown"
    driver_status: str = "unknown"
    memory_mb: Optional[int] = None
    unified_memory: bool = False
    cores: Optional[int] = None
    metal_version: Optional[str] = None
    tuning: Tuning = field(default_factory=Tuning)
    notes: str = ""


@dataclass
class TorchStatus:
    installed: bool
    version: Optional[str] = None
    cuda: bool = False
    rocm: bool = False
    mps: bool = False
    xpu: bool = False
    devices: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AcceleratorPlan:
    mode: str
    requested_backend: str
    gpus: List[GpuInfo]
    selected: Optional[GpuInfo]
    torch: TorchStatus
    use_gpu: bool
    device: str
    dtype: str
    batch_size: int
    reason: str
    call_timeout: float = 30.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_backend": self.requested_backend,
            "gpus": [asdict(gpu) for gpu in self.gpus],
            "selected": asdict(self.selected) if self.selected else None,
            "torch": asdict(self.torch),
            "compute": {
                "use_gpu": self.use_gpu,
                "device": self.device,
                "dtype": self.dtype,
                "batch_size": self.batch_size,
                "call_timeout": self.call_timeout,
                "reason": self.reason,
            },
            "warnings": self.warnings,
            "error": self.error,
        }

    def report(self) -> str:
        lines = ["GPU / accelerator report"]
        if not self.gpus:
            lines.append("  No GPU detected.")
        for gpu in self.gpus:
            lines.append(f"  {gpu.name}")
            lines.append(f"    vendor: {gpu.vendor} | backend: {gpu.backend}")
            lines.append(
                f"    architecture: {gpu.architecture} | generation: {gpu.generation}"
            )
            if gpu.compute_capability:
                lines.append(f"    compute capability: {gpu.compute_capability}")
            memory = "unknown"
            if gpu.memory_mb:
                memory = f"{gpu.memory_mb} MB"
            if gpu.unified_memory:
                memory += " (unified)"
            lines.append(f"    memory: {memory}")
            if gpu.cores:
                lines.append(f"    cores: {gpu.cores}")
            if gpu.metal_version:
                lines.append(f"    Metal: {gpu.metal_version}")
            lines.append(
                f"    driver: {gpu.driver} {gpu.driver_version}".rstrip()
            )
            lines.append(
                f"    requirement: {gpu.minimum_driver} [{gpu.driver_status}]"
            )
            tuning = gpu.tuning
            lines.append(
                "    tuning: "
                f"dtype={tuning.dtype} batch={tuning.batch_size} "
                f"tf32={'yes' if tuning.tf32 else 'no'} "
                f"fp16={'yes' if tuning.fp16 else 'no'} "
                f"bf16={'yes' if tuning.bf16 else 'no'} "
                f"fp8={'yes' if tuning.fp8 else 'no'}"
            )
            if tuning.note:
                lines.append(f"    note: {tuning.note}")
        if self.torch.installed:
            flags = ", ".join(
                f"{name}={'yes' if value else 'no'}"
                for name, value in (
                    ("CUDA", self.torch.cuda),
                    ("ROCm", self.torch.rocm),
                    ("MPS", self.torch.mps),
                    ("XPU", self.torch.xpu),
                )
            )
            lines.append(f"PyTorch: {self.torch.version} ({flags})")
        else:
            lines.append("PyTorch: not installed (GPU compute disabled)")
        lines.append(
            f"Compute: {'GPU' if self.use_gpu else 'CPU'} "
            f"(device={self.device}, dtype={self.dtype}, batch={self.batch_size})"
        )
        lines.append(f"Reason: {self.reason}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tuning tables
# ---------------------------------------------------------------------------

_APPLE_MIN_MACOS = {"M1": "11.0", "M2": "12.3", "M3": "14.0", "M4": "15.0"}
_APPLE_BATCH = {"": 512, "Pro": 1024, "Max": 2048, "Ultra": 4096}


def _apple_generation(model: str) -> Tuple[str, str, str, str]:
    match = re.search(r"\bM([1-4])\b(?:\s+(Pro|Max|Ultra))?", model, re.IGNORECASE)
    if not match:
        return ("unknown", "Apple GPU", "", "15.0")
    number = f"M{match.group(1)}"
    tier = (match.group(2) or "").title()
    generation = f"{number} {tier}".strip()
    architecture = f"Apple GPU ({generation})"
    minimum = _APPLE_MIN_MACOS.get(number, "15.0")
    return (generation, architecture, tier, minimum)


@lru_cache(maxsize=1)
def _apple_display_entries() -> List[Dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entries = data.get("SPDisplaysDataType", [])
    return entries if isinstance(entries, list) else []


def _detect_apple() -> List[GpuInfo]:
    results: List[GpuInfo] = []
    macos = platform.mac_ver()[0] or platform.release()
    ram_text = _run(["sysctl", "-n", "hw.memsize"])
    ram_mb = (_as_int(ram_text) or 0) // (1024 * 1024) or None
    for entry in _apple_display_entries():
        model = str(entry.get("sppci_model") or entry.get("_name") or "")
        vendor = str(entry.get("spdisplays_vendor") or entry.get("sppci_vendor") or "")
        if "apple" not in vendor.lower() and not model.startswith("Apple "):
            continue
        generation, architecture, tier, minimum = _apple_generation(model)
        mps_minimum = minimum if _version_at_least(minimum, "12.3") else "12.3"
        metal_raw = str(entry.get("spdisplays_mtlgpufamilysupport") or "")
        metal = metal_raw.replace("spdisplays_metal", "") or None
        tuning = Tuning(
            dtype="float32",
            batch_size=_APPLE_BATCH.get(tier, 512),
            fp16=True,
            bf16=generation.startswith(("M3", "M4")),
            matmul_precision="high",
            note="Metal Performance Shaders; no separate driver install.",
        )
        results.append(
            GpuInfo(
                vendor="apple",
                name=model or "Apple GPU",
                backend="mps",
                architecture=architecture,
                generation=generation,
                driver="Metal (built into macOS)",
                driver_version=macos,
                minimum_driver=f"macOS {minimum}+ for Metal, {mps_minimum}+ for PyTorch MPS",
                driver_status=_driver_status(macos, mps_minimum),
                memory_mb=ram_mb,
                unified_memory=True,
                cores=_as_int(entry.get("sppci_cores")),
                metal_version=metal,
                tuning=tuning,
                notes="MPS uses unified memory; leave headroom for the OS.",
            )
        )
    return results


_NVIDIA_ARCH = {
    "5": ("Maxwell", "legacy"),
    "6": ("Pascal", "legacy"),
    "7.0": ("Volta", "volta"),
    "7.5": ("Turing", "turing"),
    "8.0": ("Ampere", "ampere"),
    "8.6": ("Ampere", "ampere"),
    "8.7": ("Ampere", "ampere"),
    "8.9": ("Ada Lovelace", "ada"),
    "9.0": ("Hopper", "hopper"),
    "10.0": ("Blackwell", "blackwell"),
    "10.3": ("Blackwell", "blackwell"),
    "12.0": ("Blackwell", "blackwell"),
}


def _nvidia_arch(compute_capability: Optional[str], name: str) -> Tuple[str, str]:
    if compute_capability:
        key = ".".join(compute_capability.split(".")[:2])
        if key in _NVIDIA_ARCH:
            return _NVIDIA_ARCH[key]
        major = compute_capability.split(".")[0]
        if major in _NVIDIA_ARCH:
            return _NVIDIA_ARCH[major]
    lowered = name.lower()
    patterns = (
        (r"b100|b200|blackwell", ("Blackwell", "blackwell")),
        (r"rtx\s*50\d0", ("Blackwell", "blackwell")),
        (r"h100|h200|hopper", ("Hopper", "hopper")),
        (r"rtx\s*40\d0|l40|ada", ("Ada Lovelace", "ada")),
        (r"a100", ("Ampere", "ampere")),
        (r"rtx\s*30\d0|a10|a30|a40|a6000|ampere", ("Ampere", "ampere")),
        (r"v100|volta", ("Volta", "volta")),
        (r"rtx\s*20\d0|gtx\s*16\d0|t4|turing", ("Turing", "turing")),
        (r"gtx\s*10\d0|p100|p4|p40|pascal", ("Pascal", "legacy")),
        (r"gtx\s*9\d0|maxwell", ("Maxwell", "legacy")),
    )
    for pattern, result in patterns:
        if re.search(pattern, lowered):
            return result
    return ("Unknown NVIDIA", "unknown")


def _nvidia_min_driver(generation: str) -> str:
    windows = platform.system() == "Windows"
    if generation in ("Maxwell", "Pascal"):
        return "452.39 (Windows) / 450.80.02 (Linux), CUDA 11.8" if windows else \
            "450.80.02 (Linux) / 452.39 (Windows), CUDA 11.8"
    if generation == "Hopper":
        return "536.23 (Windows) / 535.54.03 (Linux), CUDA 12.x" if windows else \
            "535.54.03 (Linux) / 536.23 (Windows), CUDA 12.x"
    if generation == "Blackwell":
        return "571.96 (Windows) / 570.26 (Linux), CUDA 12.8+" if windows else \
            "570.26 (Linux) / 571.96 (Windows), CUDA 12.8+"
    return "527.41 (Windows) / 525.60.13 (Linux), CUDA 12.x" if windows else \
        "525.60.13 (Linux) / 527.41 (Windows), CUDA 12.x"


def _nvidia_min_version(generation: str) -> str:
    windows = platform.system() == "Windows"
    if generation in ("Maxwell", "Pascal"):
        return "452.39" if windows else "450.80.02"
    if generation == "Hopper":
        return "536.23" if windows else "535.54.03"
    if generation == "Blackwell":
        return "571.96" if windows else "570.26"
    return "527.41" if windows else "525.60.13"


def _nvidia_tuning(generation: str) -> Tuning:
    if generation in ("Maxwell", "Pascal"):
        return Tuning(
            dtype="float32",
            batch_size=256,
            note="Legacy architecture: use FP32 and a current CUDA 11.8 build.",
        )
    if generation == "Volta":
        return Tuning(dtype="float16", batch_size=512, fp16=True)
    if generation == "Turing":
        return Tuning(dtype="float16", batch_size=512, fp16=True)
    if generation == "Ampere":
        return Tuning(
            dtype="bfloat16",
            batch_size=1024,
            tf32=True,
            fp16=True,
            bf16=True,
            matmul_precision="high",
        )
    if generation == "Ada Lovelace":
        return Tuning(
            dtype="bfloat16",
            batch_size=1024,
            tf32=True,
            fp16=True,
            bf16=True,
            matmul_precision="high",
        )
    if generation == "Hopper":
        return Tuning(
            dtype="bfloat16",
            batch_size=2048,
            tf32=True,
            fp16=True,
            bf16=True,
            fp8=True,
            matmul_precision="high",
        )
    if generation == "Blackwell":
        return Tuning(
            dtype="bfloat16",
            batch_size=4096,
            tf32=True,
            fp16=True,
            bf16=True,
            fp8=True,
            matmul_precision="high",
        )
    return Tuning()


def _nvidia_gpu(
    name: str,
    driver_version: str,
    memory_mb: Optional[int],
    compute_capability: Optional[str],
) -> GpuInfo:
    generation, family = _nvidia_arch(compute_capability, name)
    minimum = _nvidia_min_driver(generation)
    minimum_version = _nvidia_min_version(generation)
    notes: List[str] = []
    if family == "legacy":
        notes.append("New PyTorch CUDA wheels may not include this architecture.")
    if family == "blackwell":
        notes.append("Requires a recent CUDA 12.8+ PyTorch build.")
    return GpuInfo(
        vendor="nvidia",
        name=name or "NVIDIA GPU",
        backend="cuda",
        architecture=generation,
        generation=generation,
        compute_capability=compute_capability,
        driver="NVIDIA proprietary driver",
        driver_version=driver_version or "unknown",
        minimum_driver=minimum,
        driver_status=_driver_status(driver_version, minimum_version),
        memory_mb=memory_mb,
        tuning=_nvidia_tuning(generation),
        notes=" ".join(notes),
    )


def _nvidia_smi_rows() -> List[Tuple[str, str, Optional[int], Optional[str]]]:
    fields = "name,driver_version,memory.total,compute_cap"
    output = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if output is None:
        output = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"]
        )
    rows: List[Tuple[str, str, Optional[int], Optional[str]]] = []
    if not output:
        return rows
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name, driver, memory = parts[0], parts[1], parts[2]
        capability = parts[3] if len(parts) > 3 and parts[3] not in ("", "[N/A]") else None
        rows.append((name, driver, _as_int(memory), capability))
    return rows


def _detect_nvidia() -> List[GpuInfo]:
    results = [_nvidia_gpu(*row) for row in _nvidia_smi_rows()]
    if results:
        return results

    if platform.system() == "Darwin":
        for entry in _apple_display_entries():
            model = str(entry.get("sppci_model") or entry.get("_name") or "")
            if "nvidia" in model.lower() or "geforce" in model.lower():
                results.append(_nvidia_gpu(model, "unknown", None, None))
        return results

    if platform.system() == "Linux":
        driver_version = "unknown"
        proc = Path("/proc/driver/nvidia/version")
        if proc.is_file():
            try:
                match = re.search(r"Kernel Module\s+([0-9.]+)", proc.read_text())
                if match:
                    driver_version = match.group(1)
            except OSError:
                pass
        for line in _lspci_lines():
            if re.search(r"nvidia", line, re.IGNORECASE):
                name = _lspci_name(line)
                results.append(_nvidia_gpu(name, driver_version, None, None))
    return results


_AMD_ARCH = {
    "gfx900": ("Vega 10", "Vega", "5.0", False, False),
    "gfx906": ("Vega 20", "Vega", "5.0", False, False),
    "gfx908": ("CDNA 1", "CDNA", "4.5", False, False),
    "gfx90a": ("CDNA 2", "CDNA", "5.0", True, False),
    "gfx942": ("CDNA 3", "CDNA", "6.0", True, True),
    "gfx1030": ("RDNA 2", "RDNA", "5.0", False, False),
    "gfx1031": ("RDNA 2", "RDNA", "5.0", False, False),
    "gfx1032": ("RDNA 2", "RDNA", "5.0", False, False),
    "gfx1100": ("RDNA 3", "RDNA", "5.7", True, False),
    "gfx1101": ("RDNA 3", "RDNA", "5.7", True, False),
    "gfx1102": ("RDNA 3", "RDNA", "5.7", True, False),
    "gfx1200": ("RDNA 4", "RDNA", "6.3", True, True),
    "gfx1201": ("RDNA 4", "RDNA", "6.3", True, True),
}


def _amd_arch(gfx: Optional[str], name: str) -> Tuple[str, str, str, bool, bool]:
    if gfx:
        for key, value in _AMD_ARCH.items():
            if gfx.startswith(key):
                return value
    lowered = name.lower()
    patterns = (
        (r"mi300|gfx942", _AMD_ARCH["gfx942"]),
        (r"mi2\d0|gfx90a", _AMD_ARCH["gfx90a"]),
        (r"mi100|gfx908", _AMD_ARCH["gfx908"]),
        (r"rx\s*9\d{3}|gfx120", _AMD_ARCH["gfx1200"]),
        (r"rx\s*7\d{3}|gfx110", _AMD_ARCH["gfx1100"]),
        (r"rx\s*6\d{3}|gfx103", _AMD_ARCH["gfx1030"]),
        (r"radeon\s*vii|gfx906", _AMD_ARCH["gfx906"]),
        (r"vega|gfx900", _AMD_ARCH["gfx900"]),
    )
    for pattern, value in patterns:
        if re.search(pattern, lowered):
            return value
    return ("Unknown AMD", "unknown", "5.0", False, False)


def _amd_rocm_version() -> str:
    for command in (
        ["rocm-smi", "--version"],
        ["cat", "/opt/rocm/.info/version"],
    ):
        output = _run(command)
        if output:
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
            if match:
                return match.group(1)
    return "unknown"


def _detect_amd() -> List[GpuInfo]:
    results: List[GpuInfo] = []
    if platform.system() == "Darwin":
        macos = platform.mac_ver()[0] or platform.release()
        for entry in _apple_display_entries():
            name = str(entry.get("sppci_model") or entry.get("_name") or "")
            if not re.search(r"amd|radeon", name, re.IGNORECASE):
                continue
            generation, family, minimum_rocm, fp16, bf16 = _amd_arch(None, name)
            results.append(
                GpuInfo(
                    vendor="amd",
                    name=name,
                    backend="rocm",
                    architecture=generation,
                    generation=generation,
                    driver="AMD driver (bundled with macOS)",
                    driver_version=macos,
                    minimum_driver="macOS 12 or newer; ROCm is not available on macOS",
                    driver_status=_driver_status(macos, "12.0"),
                    tuning=Tuning(
                        dtype="float16" if fp16 else "float32",
                        batch_size=512,
                        fp16=fp16,
                        bf16=bf16,
                    ),
                    notes="Mac AMD GPUs do not have a ROCm/PyTorch backend; use CPU.",
                )
            )
        return results

    rocm_version = _amd_rocm_version()
    products: Dict[str, str] = {}
    if shutil.which("rocm-smi"):
        output = _run(["rocm-smi", "--showproductname"])
        if output:
            current = None
            for line in output.splitlines():
                gpu_match = re.search(r"GPU\[(\d+)\]", line)
                if gpu_match:
                    current = gpu_match.group(1)
                    continue
                name_match = re.search(r"(?:Card series|Card SKU|Product Name)\s*:\s*(.+)", line, re.I)
                if current is not None and name_match:
                    products[current] = name_match.group(1).strip()
    gfx_names: List[Optional[str]] = []
    if shutil.which("rocminfo"):
        output = _run(["rocminfo"])
        if output:
            gfx_matches = re.findall(r"Name:\s*(gfx[0-9a-f]+)", output)
            marketing = re.findall(r"Marketing Name:\s*(.+)", output)
            gfx_names.extend(gfx_matches)
            for index, name in enumerate(marketing):
                if index not in products:
                    products[str(index)] = name.strip()
    if products:
        for index in sorted(products, key=lambda value: int(value) if value.isdigit() else 0):
            name = products[index]
            gfx = next((value for value in gfx_names if value), None)
            generation, family, minimum_rocm, fp16, bf16 = _amd_arch(gfx, name)
            tuning = Tuning(
                dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                batch_size=2048 if family == "CDNA" else 1024,
                fp16=fp16,
                bf16=bf16,
            )
            results.append(
                GpuInfo(
                    vendor="amd",
                    name=name,
                    backend="rocm",
                    architecture=generation,
                    generation=generation,
                    driver="amdgpu kernel driver + ROCm",
                    driver_version=rocm_version,
                    minimum_driver=f"ROCm {minimum_rocm} or newer",
                    driver_status=_driver_status(rocm_version, minimum_rocm),
                    tuning=tuning,
                    notes="Install the ROCm PyTorch wheel, not the CUDA wheel.",
                )
            )
    if results:
        return results

    for line in _lspci_lines():
        if re.search(r"amd|ati|radeon", line, re.IGNORECASE):
            name = _lspci_name(line)
            generation, family, minimum_rocm, fp16, bf16 = _amd_arch(None, name)
            results.append(
                GpuInfo(
                    vendor="amd",
                    name=name,
                    backend="rocm",
                    architecture=generation,
                    generation=generation,
                    driver="amdgpu kernel driver + ROCm",
                    driver_version=rocm_version,
                    minimum_driver=f"ROCm {minimum_rocm} or newer",
                    driver_status=_driver_status(rocm_version, minimum_rocm),
                    tuning=Tuning(
                        dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                        batch_size=1024,
                        fp16=fp16,
                        bf16=bf16,
                    ),
                    notes="ROCm runtime not detected; install ROCm before the PyTorch wheel.",
                )
            )
    return results


def _intel_arch(name: str) -> Tuple[str, str, bool, bool, str]:
    lowered = name.lower()
    if re.search(r"arc\s*b|battlemage", lowered):
        return ("Arc Battlemage", "Xe2", True, True, "1024")
    if re.search(r"arc\s*a|alchemist", lowered):
        return ("Arc Alchemist", "Xe", True, False, "512")
    if re.search(r"iris|uhd|hd graphics|xe", lowered):
        return ("Intel integrated GPU", "Xe", True, False, "256")
    return ("Unknown Intel GPU", "unknown", False, False, "256")


def _detect_intel() -> List[GpuInfo]:
    results: List[GpuInfo] = []
    if shutil.which("xpu-smi"):
        output = _run(["xpu-smi", "discovery"])
        if output:
            for line in output.splitlines():
                match = re.search(r"Device Name\s*:\s*(.+)", line, re.I)
                if not match:
                    continue
                name = match.group(1).strip()
                generation, architecture, fp16, bf16, batch = _intel_arch(name)
                results.append(
                    GpuInfo(
                        vendor="intel",
                        name=name,
                        backend="xpu",
                        architecture=generation,
                        generation=architecture,
                        driver="Intel GPU driver + oneAPI Level Zero",
                        driver_version=_run(["xpu-smi", "--version"]) or "unknown",
                        minimum_driver="Latest Intel GPU driver + oneAPI Level Zero",
                        tuning=Tuning(
                            dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                            batch_size=int(batch),
                            fp16=fp16,
                            bf16=bf16,
                        ),
                        notes="Requires a PyTorch build with Intel XPU support.",
                    )
                )
    if results:
        return results

    if platform.system() == "Darwin":
        for entry in _apple_display_entries():
            name = str(entry.get("sppci_model") or entry.get("_name") or "")
            if "intel" in name.lower():
                generation, architecture, fp16, bf16, batch = _intel_arch(name)
                results.append(
                    GpuInfo(
                        vendor="intel",
                        name=name,
                        backend="xpu",
                        architecture=generation,
                        generation=architecture,
                        driver="Intel GPU driver (bundled with macOS on Intel Macs)",
                        driver_version=platform.mac_ver()[0] or "unknown",
                        minimum_driver="macOS 12 or newer",
                        tuning=Tuning(
                            dtype="float16" if fp16 else "float32",
                            batch_size=int(batch),
                            fp16=fp16,
                            bf16=bf16,
                        ),
                        notes="Intel GPU acceleration depends on the PyTorch XPU runtime.",
                    )
                )
        return results

    for line in _lspci_lines():
        if re.search(r"intel", line, re.IGNORECASE):
            name = _lspci_name(line)
            generation, architecture, fp16, bf16, batch = _intel_arch(name)
            results.append(
                GpuInfo(
                    vendor="intel",
                    name=name,
                    backend="xpu",
                    architecture=generation,
                    generation=architecture,
                    driver="Intel GPU driver + oneAPI Level Zero",
                    driver_version="unknown",
                    minimum_driver="Latest Intel GPU driver + oneAPI Level Zero",
                    tuning=Tuning(
                        dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                        batch_size=int(batch),
                        fp16=fp16,
                        bf16=bf16,
                    ),
                    notes="Requires a PyTorch build with Intel XPU support.",
                )
            )
    return results


def _lspci_lines() -> List[str]:
    if platform.system() != "Linux":
        return []
    output = _run(["lspci"], timeout=6)
    if not output:
        return []
    return [
        line
        for line in output.splitlines()
        if re.search(r"VGA|3D|Display", line, re.IGNORECASE)
    ]


def _lspci_name(line: str) -> str:
    match = re.search(r":\s*(.+?)(?:\s+\[[0-9a-f]{4}:[0-9a-f]{4}\])?$", line)
    return match.group(1).strip() if match else line.strip()


def _detect_windows() -> List[GpuInfo]:
    if platform.system() != "Windows":
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"
    )
    output = _run(["powershell", "-NoProfile", "-Command", script], timeout=12)
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    results: List[GpuInfo] = []
    for entry in data:
        name = str(entry.get("Name") or "GPU")
        driver = str(entry.get("DriverVersion") or "unknown")
        memory = _as_int(entry.get("AdapterRAM"))
        memory_mb = memory // (1024 * 1024) if memory and memory > 0 else None
        lowered = name.lower()
        if "nvidia" in lowered:
            results.append(_nvidia_gpu(name, driver, memory_mb, None))
        elif "amd" in lowered or "radeon" in lowered:
            generation, family, minimum_rocm, fp16, bf16 = _amd_arch(None, name)
            results.append(
                GpuInfo(
                    vendor="amd",
                    name=name,
                    backend="rocm",
                    architecture=generation,
                    generation=generation,
                    driver="AMD Adrenalin driver",
                    driver_version=driver,
                    minimum_driver="Latest AMD Adrenalin driver + ROCm PyTorch build",
                    tuning=Tuning(
                        dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                        batch_size=1024,
                        fp16=fp16,
                        bf16=bf16,
                    ),
                    notes="ROCm on Windows is limited; CPU fallback is common.",
                )
            )
        elif "intel" in lowered:
            generation, architecture, fp16, bf16, batch = _intel_arch(name)
            results.append(
                GpuInfo(
                    vendor="intel",
                    name=name,
                    backend="xpu",
                    architecture=generation,
                    generation=architecture,
                    driver="Intel GPU driver",
                    driver_version=driver,
                    minimum_driver="Latest Intel GPU driver + oneAPI Level Zero",
                    tuning=Tuning(
                        dtype="bfloat16" if bf16 else "float16" if fp16 else "float32",
                        batch_size=int(batch),
                        fp16=fp16,
                        bf16=bf16,
                    ),
                )
            )
    return results


def detect_gpus() -> List[GpuInfo]:
    """Return every GPU we can identify, best-effort and cross-platform."""
    detected: List[GpuInfo] = []
    detected.extend(_detect_nvidia())
    detected.extend(_detect_amd())
    detected.extend(_detect_apple())
    detected.extend(_detect_intel())
    if platform.system() == "Windows":
        detected.extend(_detect_windows())
    seen = set()
    unique: List[GpuInfo] = []
    for gpu in detected:
        key = (gpu.vendor, gpu.name, gpu.backend)
        if key in seen:
            continue
        seen.add(key)
        unique.append(gpu)
    return unique


# ---------------------------------------------------------------------------
# PyTorch / backend selection
# ---------------------------------------------------------------------------

def torch_status() -> TorchStatus:
    if importlib.util.find_spec("torch") is None:
        return TorchStatus(installed=False, error="torch is not installed")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the host install
        return TorchStatus(installed=False, error=str(exc))
    cuda = bool(torch.cuda.is_available())
    rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    xpu_api = getattr(torch, "xpu", None)
    xpu = bool(xpu_api is not None and xpu_api.is_available())
    devices: List[str] = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(index))
    if mps:
        devices.append("Apple MPS")
    if xpu:
        try:
            devices.extend(str(name) for name in xpu_api.get_device_name(0).split(","))
        except Exception:
            devices.append("Intel XPU")
    return TorchStatus(
        installed=True,
        version=str(getattr(torch, "__version__", "unknown")),
        cuda=cuda,
        rocm=rocm,
        mps=mps,
        xpu=xpu,
        devices=devices,
    )


def _backend_available(gpu: GpuInfo, torch: TorchStatus) -> bool:
    if gpu.backend == "mps":
        return torch.mps
    if gpu.backend == "cuda":
        return torch.cuda and not torch.rocm
    if gpu.backend == "rocm":
        return torch.cuda and torch.rocm
    if gpu.backend == "xpu":
        return torch.xpu
    return False


def _backend_rank(gpu: GpuInfo) -> int:
    rank = {"nvidia": 400, "amd": 300, "apple": 200, "intel": 100}.get(gpu.vendor, 0)
    if gpu.memory_mb:
        rank += min(99, gpu.memory_mb // 1024)
    return rank


def _select_gpu(gpus: List[GpuInfo], requested: str) -> Optional[GpuInfo]:
    if not gpus:
        return None
    if requested and requested != "auto":
        wanted = {
            "cuda": {"nvidia"},
            "rocm": {"amd"},
            "mps": {"apple"},
            "xpu": {"intel"},
            "cpu": set(),
        }.get(requested, set())
        filtered = [gpu for gpu in gpus if gpu.backend == requested or gpu.vendor in wanted]
        if filtered:
            return max(filtered, key=_backend_rank)
        return None
    return max(gpus, key=_backend_rank)


def _resolve_dtype(
    requested: str,
    gpu: GpuInfo,
    warnings_list: List[str],
) -> str:
    dtype = _normalized_dtype(requested)
    tuning = gpu.tuning
    if dtype == "auto":
        return tuning.dtype
    if dtype == "float16" and not tuning.fp16:
        warnings_list.append(
            f"{gpu.name} may not support float16 well; falling back to float32."
        )
        return "float32"
    if dtype == "bfloat16" and not tuning.bf16:
        warnings_list.append(
            f"{gpu.name} may not support bfloat16; falling back to float32."
        )
        return "float32"
    if dtype == "float8":
        if not tuning.fp8:
            warnings_list.append(
                f"{gpu.name} does not advertise float8 support; falling back to {tuning.dtype}."
            )
            return tuning.dtype
        # The policy is too small for FP8 to help, and PyTorch builds expose
        # several incompatible FP8 dtypes.  Use the next stable format.
        warnings_list.append("float8 is not used for the tiny policy; using float16.")
        return "float16" if tuning.fp16 else tuning.dtype
    return dtype


def _default_device(gpu: GpuInfo) -> str:
    if gpu.backend == "mps":
        return "mps"
    if gpu.backend == "rocm":
        return "cuda:0"
    if gpu.backend == "cuda":
        return "cuda:0"
    if gpu.backend == "xpu":
        return "xpu:0"
    return "cpu"


def resolve(
    config: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    requested_backend: Optional[str] = None,
) -> AcceleratorPlan:
    """Build a compute plan from the ``gpu`` section of ``config.json``.

    ``mode`` is ``off``, ``auto``, or ``force``.  ``auto`` uses the GPU when a
    matching PyTorch runtime is available; ``force`` treats a missing runtime
    as an error unless ``fallback_to_cpu`` is true.
    """
    config = dict(config or {})
    mode = _normalized_mode(mode if mode is not None else config.get("mode", "off"))
    requested = str(
        requested_backend
        if requested_backend is not None
        else config.get("backend", "auto")
    ).strip().lower()
    fallback = bool(config.get("fallback_to_cpu", True))
    gpus = detect_gpus()
    torch = torch_status()
    selected = _select_gpu(gpus, requested)
    warnings_list: List[str] = []

    if mode == "off":
        return AcceleratorPlan(
            mode=mode,
            requested_backend=requested,
            gpus=gpus,
            selected=selected,
            torch=torch,
            use_gpu=False,
            device="cpu",
            dtype="float32",
            batch_size=1,
            reason="gpu.mode=off; using the CPU simulation path.",
            warnings=warnings_list,
        )

    if selected is None:
        error = None
        if mode == "force" and not fallback:
            error = "No matching GPU was detected."
        return AcceleratorPlan(
            mode=mode,
            requested_backend=requested,
            gpus=gpus,
            selected=None,
            torch=torch,
            use_gpu=False,
            device="cpu",
            dtype="float32",
            batch_size=1,
            reason="No matching GPU detected; using CPU.",
            warnings=warnings_list,
            error=error,
        )

    if not torch.installed:
        message = (
            "PyTorch is not installed, so the GPU policy backend is unavailable. "
            "Install a platform wheel, for example: "
            "python3 -m pip install torch"
        )
        warnings_list.append(message)
        error = message if mode == "force" and not fallback else None
        return AcceleratorPlan(
            mode=mode,
            requested_backend=requested,
            gpus=gpus,
            selected=selected,
            torch=torch,
            use_gpu=False,
            device="cpu",
            dtype="float32",
            batch_size=1,
            reason="PyTorch is not installed; using CPU.",
            warnings=warnings_list,
            error=error,
        )

    if not _backend_available(selected, torch):
        message = (
            f"{selected.name} was detected, but this PyTorch build cannot use its "
            f"{selected.backend} backend. Use a matching CUDA/ROCm/MPS/XPU build."
        )
        warnings_list.append(message)
        error = message if mode == "force" and not fallback else None
        return AcceleratorPlan(
            mode=mode,
            requested_backend=requested,
            gpus=gpus,
            selected=selected,
            torch=torch,
            use_gpu=False,
            device="cpu",
            dtype="float32",
            batch_size=1,
            reason=f"The installed PyTorch build has no {selected.backend} backend; using CPU.",
            warnings=warnings_list,
            error=error,
        )

    dtype = _resolve_dtype(str(config.get("precision", "auto")), selected, warnings_list)
    requested_device = str(config.get("device", "auto")).strip()
    device = _default_device(selected) if requested_device in ("", "auto") else requested_device
    batch_size = _as_int(config.get("batch_size")) or 0
    if batch_size <= 0:
        batch_size = selected.tuning.batch_size
    try:
        call_timeout = max(1.0, float(config.get("call_timeout", 30.0)))
    except (TypeError, ValueError):
        call_timeout = 30.0
    return AcceleratorPlan(
        mode=mode,
        requested_backend=requested,
        gpus=gpus,
        selected=selected,
        torch=torch,
        use_gpu=True,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        reason=(
            f"Using {selected.name} via {selected.backend} with {dtype}. "
            "The policy is tiny, so the CPU path may still be faster."
        ),
        call_timeout=call_timeout,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        default = Path(__file__).resolve().parents[1] / "config.json"
        path = default if default.is_file() else None
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("gpu", {}) if isinstance(data, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="config.json to read the gpu section from")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument(
        "--mode",
        choices=("off", "auto", "force"),
        help="override the configured GPU mode",
    )
    parser.add_argument("--backend", help="auto, cuda, rocm, mps, xpu, or cpu")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="compare the CPU policy loop with the GPU backend",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="benchmark batch size")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.benchmark and args.mode is None and config.get("mode", "off") == "off":
        # A benchmark is an explicit request to exercise the accelerator.
        args.mode = "auto"
    plan = resolve(config, mode=args.mode, requested_backend=args.backend)
    if args.benchmark:
        from gpu_backend import benchmark

        result = benchmark(plan, batch_size=args.batch_size)
        print(json.dumps(result, indent=2))
        return 0 if not result.get("error") else 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.report())
    if plan.error:
        print(plan.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
