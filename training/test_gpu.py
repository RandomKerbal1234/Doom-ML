#!/usr/bin/env python3
"""Unit tests for GPU detection and the policy reference implementation."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu  # noqa: E402
from env import HeadlessDoomEnv, choose_action_from_inputs  # noqa: E402


class GenerationTests(unittest.TestCase):
    def test_nvidia_compute_capabilities(self):
        self.assertEqual(gpu._nvidia_arch("6.1", "GTX 1080"), ("Pascal", "legacy"))
        self.assertEqual(gpu._nvidia_arch("7.5", "RTX 2080"), ("Turing", "turing"))
        self.assertEqual(gpu._nvidia_arch("8.6", "RTX 3070"), ("Ampere", "ampere"))
        self.assertEqual(gpu._nvidia_arch("8.9", "RTX 4090"), ("Ada Lovelace", "ada"))
        self.assertEqual(gpu._nvidia_arch("9.0", "H100"), ("Hopper", "hopper"))
        self.assertEqual(gpu._nvidia_arch("12.0", "RTX 5090"), ("Blackwell", "blackwell"))

    def test_nvidia_name_fallback(self):
        self.assertEqual(gpu._nvidia_arch(None, "NVIDIA GeForce RTX 4070"), ("Ada Lovelace", "ada"))
        self.assertEqual(gpu._nvidia_arch(None, "NVIDIA GeForce GTX 1080"), ("Pascal", "legacy"))

    def test_amd_architectures(self):
        self.assertEqual(gpu._amd_arch("gfx1100", "RX 7900")[0], "RDNA 3")
        self.assertEqual(gpu._amd_arch("gfx942", "MI300X")[0], "CDNA 3")
        self.assertEqual(gpu._amd_arch("gfx1030", "RX 6800")[0], "RDNA 2")
        self.assertEqual(gpu._amd_arch(None, "AMD Radeon RX 6800 XT")[0], "RDNA 2")

    def test_apple_families(self):
        self.assertEqual(gpu._apple_generation("Apple M1 Pro")[:3],
                         ("M1 Pro", "Apple GPU (M1 Pro)", "Pro"))
        self.assertEqual(gpu._apple_generation("Apple M4 Max")[0], "M4 Max")

    def test_intel_families(self):
        self.assertEqual(gpu._intel_arch("Intel Arc A770")[0], "Arc Alchemist")
        self.assertEqual(gpu._intel_arch("Intel Arc B580")[0], "Arc Battlemage")


class DriverTests(unittest.TestCase):
    def test_version_comparison(self):
        self.assertTrue(gpu._version_at_least("550.54.14", "525.60.13"))
        self.assertFalse(gpu._version_at_least("470.82", "525.60.13"))
        self.assertIsNone(gpu._version_at_least("unknown", "525.60.13"))

    def test_dtype_aliases(self):
        self.assertEqual(gpu._normalized_dtype("fp32"), "float32")
        self.assertEqual(gpu._normalized_dtype("fp16"), "float16")
        self.assertEqual(gpu._normalized_dtype("bf16"), "bfloat16")
        self.assertEqual(gpu._normalized_dtype("auto"), "auto")


class PolicyTests(unittest.TestCase):
    def test_zero_policy(self):
        weights = [0.0] * 430
        action = choose_action_from_inputs(weights, [0.0] * 42, [True, False, False, False])
        self.assertEqual(action, [0.0, 0.0, 0.0, False, False, 0, False])

    def test_fire_flag_comes_from_output_three(self):
        weights = [0.0] * 430
        weights[3 * 43 + 42] = 1.0
        action = choose_action_from_inputs(weights, [0.0] * 42, [True, False, False, False])
        self.assertTrue(action[3])
        self.assertFalse(action[4])


class HangGuardTests(unittest.TestCase):
    def _stuck_executable(self, directory: Path) -> Path:
        script = directory / "stuck_headless.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, time\n"
            "for _line in sys.stdin:\n"
            "    time.sleep(3600)\n"
        )
        script.chmod(0o755)
        return script

    def test_headless_env_read_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = HeadlessDoomEnv(self._stuck_executable(Path(tmp)), step_timeout=0.3)
            try:
                start = time.monotonic()
                with self.assertRaises(RuntimeError):
                    env.reset(0)
                self.assertLess(time.monotonic() - start, 10.0)
                with self.assertRaises(RuntimeError):
                    env.reset(0)  # fails fast once the environment is broken
            finally:
                env.close()
            self.assertIsNotNone(env.process.poll())

    def test_gpu_call_times_out_and_disables_backend(self):
        try:
            import gpu_backend
        except Exception as exc:  # pragma: no cover - depends on the host install
            self.skipTest(f"gpu backend unavailable: {exc}")
        plan = gpu.resolve({"mode": "auto", "backend": "auto"}, mode="auto")
        if not getattr(plan, "use_gpu", False):
            self.skipTest("no accelerator available")
        backend = gpu_backend.TorchPolicyBackend(plan, timeout=1.0)
        stop = threading.Event()
        try:
            with self.assertRaises(gpu_backend.GpuTimeoutError):
                backend._call(stop.wait)
            with self.assertRaises(gpu_backend.GpuBackendError):
                backend.choose_action([0.0] * 430, [0.0] * 42, [True, False, False, False])
        finally:
            stop.set()
            backend.close()


if __name__ == "__main__":
    unittest.main()
