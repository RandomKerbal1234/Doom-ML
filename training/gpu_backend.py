#!/usr/bin/env python3
"""Optional PyTorch policy backend for GPU-accelerated action selection.

This module is imported only when ``gpu.mode`` is ``auto`` or ``force``.  It
keeps the project's default dependency-free CPU path untouched: if PyTorch or
a suitable accelerator is missing, the trainer falls back to the Rust
``--evaluate`` path.

The current policy is deliberately tiny (430 weights).  A GPU is therefore
unlikely to beat the CPU for this workload; ``benchmark`` reports the real
numbers so the choice is explicit instead of assumed.
"""

from __future__ import annotations

import concurrent.futures
import queue
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env import choose_action_from_inputs as cpu_choose_action

INPUTS = 42
OUTPUTS = 10
WEIGHTS = OUTPUTS * (INPUTS + 1)
DEFAULT_TIMEOUT = 30.0


class GpuBackendError(RuntimeError):
    """Raised when the accelerator backend cannot be used safely."""


class GpuTimeoutError(GpuBackendError):
    """Raised when a device call does not finish within the configured timeout."""


class TorchPolicyBackend:
    """Evaluate the linear policy on a PyTorch device.

    Every device call runs on one dedicated daemon thread.  Multi-threaded
    MPS/CUDA use is a known source of deadlocks, and a stuck device call would
    otherwise block the training process forever.  The queue keeps device
    access serialized and the timeout turns a stall into an exception that the
    trainer can recover from by falling back to the CPU path.
    """

    def __init__(self, plan: Any, timeout: Optional[float] = None):
        if not getattr(plan, "use_gpu", False):
            raise RuntimeError("The accelerator plan does not enable a GPU backend")
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on the host install
            raise RuntimeError(f"PyTorch is not available: {exc}") from exc
        self.torch = torch
        self.plan = plan
        self.device = str(plan.device)
        self.dtype_name = str(plan.dtype)
        self.dtype = getattr(torch, self.dtype_name, None)
        if self.dtype is None:
            raise RuntimeError(f"unsupported PyTorch dtype: {self.dtype_name}")
        self.batch_size = int(getattr(plan, "batch_size", 1) or 1)
        configured = timeout
        if configured is None:
            configured = getattr(plan, "call_timeout", DEFAULT_TIMEOUT)
        try:
            self.timeout = max(1.0, float(configured))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT
        self._policy_cache: Dict[int, Tuple[Any, Any]] = {}
        self._failed: Optional[BaseException] = None
        self._closed = False
        self._queue: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run_worker, name="torch-policy", daemon=True
        )
        self._worker.start()
        # Initialize the device on the worker thread.  If this fails or stalls,
        # the caller sees a normal exception and can fall back to the CPU.
        self._call(self._warmup)

    def _configure(self) -> None:
        torch = self.torch
        selected = getattr(self.plan, "selected", None)
        tuning = getattr(selected, "tuning", None)
        use_tf32 = bool(getattr(tuning, "tf32", False))
        cuda = getattr(torch.backends, "cuda", None)
        if cuda is not None and use_tf32:
            try:
                cuda.matmul.allow_tf32 = True
                cudnn = getattr(torch.backends, "cudnn", None)
                if cudnn is not None:
                    cudnn.allow_tf32 = True
            except Exception:
                pass
        try:
            precision = str(getattr(tuning, "matmul_precision", "highest") or "highest")
            torch.set_float32_matmul_precision(
                precision if precision in ("highest", "high", "medium") else "highest"
            )
        except Exception:
            pass

    def _run_worker(self) -> None:
        self._configure()
        while True:
            item = self._queue.get()
            if item is None:
                return
            function, arguments, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                value = function(*arguments)
            except BaseException as exc:  # propagate to the waiting caller
                future.set_exception(exc)
            else:
                future.set_result(value)

    def _call(self, function, *arguments):
        if self._closed:
            raise GpuBackendError("the GPU policy backend is closed")
        if self._failed is not None:
            raise GpuBackendError(
                f"the GPU policy backend is unavailable after an earlier failure: {self._failed}"
            )
        future: "concurrent.futures.Future" = concurrent.futures.Future()
        self._queue.put((function, arguments, future))
        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            error = GpuTimeoutError(
                f"a {self.device} call did not finish within {self.timeout:g}s; "
                "the GPU backend has been disabled"
            )
            self._failed = error
            raise error from None
        except Exception as exc:
            self._failed = exc
            raise

    def _warmup(self) -> None:
        torch = self.torch
        policy = torch.zeros((OUTPUTS, INPUTS + 1), dtype=self.dtype, device=self.device)
        inputs = torch.zeros((1, INPUTS), dtype=self.dtype, device=self.device)
        outputs = torch.tanh(
            inputs @ policy[:, :INPUTS].transpose(0, 1) + policy[:, INPUTS]
        )
        outputs.tolist()
        self._sync()

    def _sync(self) -> None:
        torch = self.torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        elif self.device.startswith("xpu"):
            try:
                torch.xpu.synchronize()
            except Exception:
                pass

    def _policy_for(self, weights: Sequence[float]):
        if len(weights) != WEIGHTS:
            raise ValueError(f"expected {WEIGHTS} weights, got {len(weights)}")
        key = id(weights)
        cached = self._policy_cache.get(key)
        if cached is not None and cached[0] is weights:
            return cached[1]
        if len(self._policy_cache) >= 256:
            self._policy_cache.clear()
        tensor = (
            self.torch.tensor(weights, dtype=self.dtype, device=self.device)
            .reshape(OUTPUTS, INPUTS + 1)
        )
        self._policy_cache[key] = (weights, tensor)
        return tensor

    def _outputs(self, policy, inputs: Sequence[float]) -> List[float]:
        torch = self.torch
        x = torch.tensor(inputs, dtype=self.dtype, device=self.device).reshape(1, INPUTS)
        out = torch.tanh(x @ policy[:, :INPUTS].transpose(0, 1) + policy[:, INPUTS])
        return out[0].tolist()

    def _single_outputs(self, weights: Sequence[float], inputs: Sequence[float]) -> List[float]:
        return self._outputs(self._policy_for(weights), inputs)

    def _batch_outputs(
        self,
        genomes: Sequence[Sequence[float]],
        inputs_batch: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        torch = self.torch
        w = torch.tensor(genomes, dtype=self.dtype, device=self.device).reshape(
            len(genomes), OUTPUTS, INPUTS + 1
        )
        x = torch.tensor(inputs_batch, dtype=self.dtype, device=self.device).reshape(
            len(genomes), 1, INPUTS
        )
        out = torch.tanh(
            torch.bmm(x, w[:, :, :INPUTS].transpose(1, 2)).squeeze(1) + w[:, :, INPUTS]
        )
        return out.tolist()

    def choose_action(
        self,
        weights: Sequence[float],
        inputs: Sequence[float],
        unlocked: Sequence[bool],
    ) -> List[Any]:
        """Return the same action vector as ``env.choose_action``."""
        outputs = self._call(self._single_outputs, weights, inputs)
        return self._decode(outputs, unlocked)

    @staticmethod
    def _decode(outputs: Sequence[float], unlocked: Sequence[bool]) -> List[Any]:
        action: List[Any] = [
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3] > 0,
            outputs[4] > 0,
            0,
            outputs[9] > 0,
        ]
        available = [index for index, value in enumerate(unlocked) if value]
        if available:
            action[5] = max(available, key=lambda index: outputs[5 + index])
        return action

    def choose_actions_batch(
        self,
        genomes: Sequence[Sequence[float]],
        inputs_batch: Sequence[Sequence[float]],
        unlocked_batch: Sequence[Sequence[bool]],
    ) -> List[List[Any]]:
        """Vectorized action selection for many independent policies."""
        if len(genomes) != len(inputs_batch) or len(genomes) != len(unlocked_batch):
            raise ValueError("genomes, inputs_batch, and unlocked_batch must align")
        if not genomes:
            return []
        rows = self._call(self._batch_outputs, genomes, inputs_batch)
        return [self._decode(row, unlocked) for row, unlocked in zip(rows, unlocked_batch)]

    def synchronize(self) -> None:
        """Wait for queued device work to finish (used by the benchmark)."""
        self._call(self._sync)

    def _release(self) -> None:
        self._policy_cache.clear()
        self._sync()
        try:
            if self.device == "mps":
                self.torch.mps.empty_cache()
            elif self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Flush and release device memory on the owning thread before stopping
        # it.  This keeps interpreter shutdown from racing with pending device
        # work, which can otherwise stall the process after training finishes.
        future: "concurrent.futures.Future" = concurrent.futures.Future()
        self._queue.put((self._release, (), future))
        try:
            future.result(timeout=2.0)
        except Exception:
            pass
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)


def benchmark(
    plan: Any,
    batch_size: int = 256,
    repeats: int = 20,
    seed: int = 7,
) -> Dict[str, Any]:
    """Compare pure-Python, per-item GPU, and batched GPU policy inference."""
    if not getattr(plan, "use_gpu", False):
        return {
            "error": "GPU backend is not enabled",
            "reason": getattr(plan, "reason", ""),
            "compute": {
                "use_gpu": getattr(plan, "use_gpu", False),
                "device": getattr(plan, "device", "cpu"),
            },
        }
    try:
        backend = TorchPolicyBackend(plan)
    except Exception as exc:
        return {"error": str(exc)}

    rng = random.Random(seed)
    count = max(1, int(batch_size))
    genomes = [
        [rng.gauss(0.0, 0.35) for _ in range(WEIGHTS)] for _ in range(count)
    ]
    observations = [
        [rng.random() for _ in range(INPUTS)] for _ in range(count)
    ]
    unlocked = [[True, rng.random() > 0.5, rng.random() > 0.5, rng.random() > 0.5]
                for _ in range(count)]

    # Warm up all paths so lazy device initialization is not timed.
    cpu_choose_action(genomes[0], observations[0], unlocked[0])
    backend.choose_action(genomes[0], observations[0], unlocked[0])
    backend.choose_actions_batch(genomes, observations, unlocked)
    backend.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        for genome, observation, mask in zip(genomes, observations, unlocked):
            cpu_choose_action(genome, observation, mask)
    cpu_seconds = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        for genome, observation, mask in zip(genomes, observations, unlocked):
            backend.choose_action(genome, observation, mask)
    backend.synchronize()
    single_seconds = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        backend.choose_actions_batch(genomes, observations, unlocked)
    backend.synchronize()
    batch_seconds = (time.perf_counter() - start) / repeats
    backend.close()

    def _speedup(baseline: float, candidate: float) -> Optional[float]:
        if candidate <= 0.0:
            return None
        return round(baseline / candidate, 3)

    return {
        "device": plan.device,
        "dtype": plan.dtype,
        "torch_version": plan.torch.version,
        "batch_size": count,
        "repeats": repeats,
        "cpu_ms": round(cpu_seconds * 1000.0, 4),
        "gpu_single_ms": round(single_seconds * 1000.0, 4),
        "gpu_batch_ms": round(batch_seconds * 1000.0, 4),
        "gpu_single_speedup": _speedup(cpu_seconds, single_seconds),
        "gpu_batch_speedup": _speedup(cpu_seconds, batch_seconds),
        "note": (
            "This measures policy inference only. Training also pays the "
            "simulation and IPC cost, which dominates for this tiny model."
        ),
    }
