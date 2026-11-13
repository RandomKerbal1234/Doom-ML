# DOOM machine learning environment

The training loop evolves a small policy from random weights. Each evaluation
starts at a uniformly random level, then attempts 50 exits in campaign order,
wrapping from the final level to the first. Fitness rewards level completion,
shortest-path progress toward each exit, damage dealt, survival, and efficient
ammo and time use. The Rust
headless executable runs the same simulation code as the game, without a
window.

## Build and run

From the repository root:

```sh
./exec_mgmt.sh -b
python3 training/train.py
```

Or build automatically before training with `python3 training/train.py --build`.
Use `--config path/to/config.json` to select another configuration. A short
smoke training can be run with `--generations 1`.

While training is running, open a second terminal and run
`./watch_agent.sh`. It opens the DOOM GUI and follows the policy in
`training/output/best_policy.json`, reloading it when a new generation is saved.
Press **R** to restart with a random starting level or **Q** to close the viewer.
To watch another policy file, pass its path to the script.

## GPU acceleration

The trainer can optionally select policy actions with a PyTorch GPU backend and
can report the installed GPU, the driver/runtime it needs, and generation
specific tuning. The default is still the fast Rust CPU path: the policy has
only 430 weights, so the DOOM simulation dominates and a GPU is usually
**slower** for this particular model. GPU mode is available for experimentation,
larger policies, and future scaling.

Identify the installed accelerator and its recommended settings without
starting a training run:

```sh
python3 training/gpu.py
python3 training/gpu.py --json
python3 training/gpu.py --mode auto
python3 training/gpu.py --benchmark --batch-size 256
```

The report covers:

- **Vendor and backend**: Apple Metal/MPS, NVIDIA CUDA, AMD ROCm, Intel XPU.
- **Generation**: Apple M1/M2/M3/M4 families; NVIDIA Pascal through Blackwell;
  AMD Vega/CDNA/RDNA; Intel Arc Alchemist/Battlemage/Xe.
- **Driver requirements**: Apple's built-in Metal version and minimum macOS,
  NVIDIA's minimum driver and CUDA generation, AMD's ROCm version, and Intel's
  oneAPI Level Zero runtime.
- **Tuning**: preferred dtype, batch size, TF32/FP16/BF16/FP8 availability.

Enable the GPU backend for a run with `--gpu` (use it when available) or
`--force-gpu` (fail instead of silently falling back). Disable it with `--cpu`.
A `gpu` section with `"mode": "auto"` still uses the **CPU** path for this
430-weight policy: `auto` means "use the faster path", and the Rust loop runs
the whole episode in-process while GPU mode drives one simulation step at a
time from Python. That step-by-step path is roughly two orders of magnitude
slower, so it would look hung during a normal run. Use `--gpu` or
`"mode": "force"` to exercise the accelerator anyway; GPU generations print
progress every few episodes. The equivalent configuration in `config.json` is:

```json
"gpu": {
  "mode": "auto",
  "backend": "auto",
  "device": "auto",
  "precision": "auto",
  "batch_size": 0,
  "call_timeout": 30,
  "step_timeout": 60,
  "fallback_to_cpu": true,
  "report": true
}
```

`mode` is `off`, `auto`, or `force`; `backend` is `auto`, `cuda`, `rocm`,
`mps`, `xpu`, or `cpu`; `batch_size: 0` uses the detected generation default.
A `gpu_report.json` is written next to the checkpoints when `report` is true.
`--gpu-benchmark` measures the CPU and GPU policy loops on the current machine
and prints the comparison. The backend is optional: without PyTorch the
trainer prints the detected hardware and continues on the CPU path.

GPU mode is hardened against the two ways an accelerator run can stall: every
device call runs on a single dedicated thread with a `call_timeout` guard, and
each headless simulation request has a `step_timeout` guard. If either guard
fires, the backend is disabled and the current generation is retried on the
Rust CPU path, so a stuck device or simulation cannot hang training. With
`fallback_to_cpu: false` (or `--force-gpu`) the run stops with an error instead.

PyTorch is not a project dependency. Install a wheel that matches the detected
backend (for example the default macOS arm64 wheel for MPS, a CUDA wheel from
the PyTorch index for NVIDIA, or a ROCm wheel for AMD). The game and the
headless trainer continue to build with only the Rust toolchain.

## Configuration

Edit the root `config.json`. `cpu_threads` controls parallel episode evaluations:
`0` uses all detected CPUs, `-1` uses all but one, and a positive integer selects
that many concurrent evaluations. Each episode now runs its policy loop inside
the Rust headless executable, avoiding a Python/Rust JSON exchange on every
simulation step. Episodes are scheduled independently so a population with
multiple evaluation episodes can use more workers than there are candidates.
Other settings control population and mutation,
simulation step size, run length, evaluation episodes, seed, reward weights, checkpointing, and
output paths. `doom_difficulty` selects Easy, Medium, or Hard game settings.
The `gpu` section controls the optional accelerator backend described above;
leave `mode` at `off` to keep the fastest CPU path.
Training begins with random policies when no checkpoint exists;
with `resume` enabled, a saved checkpoint continues at the next generation.

The best policy and population checkpoint are JSON files; generation summaries
are appended to CSV. The evaluation job is self-contained and has a seed and
policy, which leaves a clear seam for distributing evaluation jobs across
machines later. This version only runs testers on the local machine.

## Interface

`training/env.py` exposes `HeadlessDoomEnv`, with `reset(level)` and `step(action)`.
Observations contain player state, map, enemies, pickups, and exit position.
Actions contain normalized turn, forward and strafe values, fire/exit flags,
weapon index, and run flag. The trainer maps observations to actions and
evaluates evolved policies.
