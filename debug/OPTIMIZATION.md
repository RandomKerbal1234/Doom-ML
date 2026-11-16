# Profiling and performance optimization

This document covers the profiling instrumentation added to the `debug/` copy of
DOOM-ML, the 50-generation measurement run that continued from the existing
generation-99 checkpoint, the optimizations derived from that data, and the
measured before/after results.

All work described here is confined to `debug/`. The root project was not
modified.

---

## 1. Summary

The profile showed that the trainer was **not** limited by the DOOM simulation.
It was limited by starting a new `doom_headless` process for every one of the
4,800 episodes:

| 50 generations, 4,800 episodes | Before | After | Change |
| --- | ---: | ---: | ---: |
| Wall time, summed per generation | **91.16 s** | **16.58 s** | **5.50× faster** |
| Episodes per second | 52.7 | 289.6 | 5.50× |
| Simulated steps per second | 202 k | 1.11 M | 5.50× |
| Summed per-episode wall time | 600.14 s | 102.84 s | 5.84× less |
| Process/transport overhead | 469.00 s (78.1%) | 8.49 s (8.3%) | 55.2× less |
| Rust simulation time | 131.13 s (21.9%) | 94.35 s (91.7%) | 1.39× less |
| User CPU time of the whole run | 495.36 s | 86.67 s | 5.72× less |
| System CPU time of the whole run | 122.42 s | 1.23 s | 99.5× less |

The main changes:

1. **Persistent evaluation servers** — one reusable Rust process per worker,
   driven over a JSON-lines protocol, instead of one process per episode.
2. **Persistent Python worker pool** — one thread per worker for the whole run,
   instead of creating a `ProcessPoolExecutor` every generation.
3. **`macroquad` removed from the headless build** — the headless binary no
   longer links AppKit, Metal, MetalKit, Vision, CoreVideo, or ImageIO.
4. **Hot-path simulation work removed** — cached wall grid, allocation-free
   policy encoding, single-pass nearest-enemy search, a kill counter instead of
   per-step alive scans, and no visual-only danger/effect work.

Correctness was verified by replaying the same 50 generations from the same
generation-99 checkpoint: all 50 generation summaries match exactly (best and
mean fitness, levels, kills, health, seconds, start level), the total step count
is identical (18,429,672), and individual genomes replayed against the preserved
baseline binary produced identical scores.

---

## 2. Test environment and workload

| Item | Value |
| --- | --- |
| Machine used for all timings | macOS, 4 physical cores / 8 hardware threads, Python 3.9.6 |
| Trainer workers | 8 (default `cpu_threads: 0` detected 8 CPUs) |
| Population | 48 genomes |
| Evaluation episodes | 2 per genome (96 per generation) |
| Generations measured | 100–149 (50 generations) |
| Total episodes | 4,800 |
| Total simulated steps | 18,429,672 (3,839 per episode) |
| Episode settings | `dt = 0.05`, `max_steps = 15,000`, difficulty `medium` |
| Fitness weights | unchanged from the existing `config.json` |

Both the before and after runs used profiling enabled (`DOOM_PROFILE=1`), so
the comparison is like-for-like. The "before" numbers below come from a clean
replay of the same 50 generations with the preserved baseline binary and
trainer (`debug/bench/baseline_run/`), which makes the whole-run CPU accounting a
single invocation. The original required profiling run in
`debug/training/output/` produced identical metrics and a summed generation
wall of 88.09 s, about 3% faster; the difference is run-to-run machine noise.

The target deployment machines are dual Intel Xeon E5-2643 v2 (24 hardware
threads total) or quad AMD Opteron 6172 (48 real cores). Section 9 discusses how
the changes map onto those machines.

---

## 3. What was measured, and how

### Rust side

`debug/doom/src/game.rs` keeps a `GameProfile` with nanosecond counters around
the simulation phases used by training:

- level loading
- player movement / exit activation
- firing
- `update()` as a whole
- enemy AI (including line-of-sight checks)
- the HUD danger scan
- pickup handling

`debug/doom/src/bin/doom_headless.rs` adds outer counters:

- process start and level-table parsing
- per-episode setup (`Game` construction + `agent_start`)
- policy inference (`agent_policy_action`)
- one simulation step (`agent_step`)
- kill accounting
- the rest of the episode loop

When `DOOM_PROFILE=1` is set, each episode emits a `PROFILE {json}` object on
stderr (`--evaluate`) or a `"profile"` member in the JSON result
(`--evaluate-server`). Without the variable, the timers are not taken.

### Python side

`debug/training/train.py` writes one row per generation to `profile.csv` with:

- wall time for the generation, job setup, evaluation, ranking, evolution, writes
- summed per-episode wall time and the non-simulation remainder
- every Rust counter above, plus episodes, steps, and per-generation totals
- `profile_summary.json` with totals, means, and percentages of episode wall

`debug/training/env.py` attaches each episode's `profile` and its own
`wall_seconds` (send → receive) to the result returned by the evaluator.

The instrumentation itself costs about **9.7%** wall time on this machine
(5.78 s vs 5.27 s for the same 20-generation run), so production runs should
leave `DOOM_PROFILE` unset.

---

## 4. Baseline profile: where the time went

Baseline: 50 generations, 4,800 episodes, 18,429,672 steps, profiling on.
Totals are summed across episodes (they are CPU-time-like, not wall time).

| Component | Total | % of episode time | Per episode | Per step |
| --- | ---: | ---: | ---: | ---: |
| **Episode wall** | **600.14 s** | 100% | 125.03 ms | 32.6 µs |
| Process startup / transport | **469.00 s** | **78.1%** | **97.71 ms** | 25.4 µs |
| Rust `main` (simulation) | 131.13 s | 21.9% | 27.32 ms | 7.12 µs |
| ├─ level parsing | 2.87 s | 0.5% | 0.60 ms | 0.16 µs |
| └─ episode evaluation | 128.18 s | 21.4% | 26.70 ms | 6.96 µs |
| &nbsp;&nbsp;├─ policy inference | 65.27 s | 10.9% | 13.60 ms | 3.54 µs |
| &nbsp;&nbsp;├─ simulation step | 51.33 s | 8.6% | 10.69 ms | 2.79 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;├─ `update()` | 37.50 s | 6.2% | 7.81 ms | 2.04 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;│&nbsp;&nbsp;├─ enemy AI | 15.77 s | 2.6% | 3.28 ms | 0.86 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;│&nbsp;&nbsp;├─ danger scan | 12.91 s | 2.2% | 2.69 ms | 0.70 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;│&nbsp;&nbsp;└─ pickups | 1.78 s | 0.3% | 0.37 ms | 0.10 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;├─ movement | 4.12 s | 0.7% | 0.86 ms | 0.22 µs |
| &nbsp;&nbsp;│&nbsp;&nbsp;└─ firing | 2.71 s | 0.5% | 0.57 ms | 0.15 µs |
| &nbsp;&nbsp;├─ kill accounting | 3.47 s | 0.6% | 0.72 ms | 0.19 µs |
| &nbsp;&nbsp;└─ loop remainder | 7.08 s | 1.2% | 1.47 ms | 0.38 µs |
| Generation wall (sum of 50) | 91.16 s | — | 1.82 s/gen | — |

### Interpretation

- **Process startup dominated at 78.1%.** Each episode was a fresh process:
  `fork`/`exec`, dynamic linking, runtime initialization, level parsing, and
  allocation. On this machine it cost about **98 ms per episode under 8-way
  load** — more than three times the actual simulation work. Even making the
  simulation infinitely fast would have left 78.1% of the cost in place.
- **Policy inference was the largest simulation cost** (49.8% of Rust time).
  The original implementation built a `Vec` for 42 inputs, allocated and sorted
  lists of enemies, and scanned the level through the level registry for every
  ray step.
- **`update()` was next** (28.6% of Rust time). Within it the danger scan alone
  consumed 12.9 s (8.2× its post-optimization cost), and it exists only to draw
  the red HUD direction indicator. Enemy AI was the legitimate cost.
- **Kill accounting scanned the whole enemy list twice per step** (3.47 s) just
  to notice deaths.
- The headless binary linked AppKit/Metal/MetalKit/Vision/CoreVideo/ImageIO
  because `game.rs` imported `macroquad`. That inflated startup and meant the
  trainer could not run on a server with no graphics stack.

---

## 5. Changes

### 5.1 Persistent evaluator protocol (`--evaluate-server`)

`doom_headless` now has a mode that reads one job per line and returns one JSON
result per line:

```text
START_LEVEL,DIFFICULTY,DT,MAX_STEPS,LEVEL_COMPLETION,PARTIAL_LEVEL,
HEALTH_PER_POINT,KILL,SECONDS_PENALTY,w1,...,w430
```

The server creates one `Game` when it starts and reuses its allocations for
every request. `run_episode()` is shared with `--evaluate`, so the scoring code
is identical. `agent_start()` explicitly resets every piece of per-episode
state (time, timers, hit effects, danger flag, kill counter, profile counters,
difficulty settings, level), which makes a reused `Game` equivalent to a fresh
one.

Result: process launch, dynamic linking, level parsing, and the initial
allocation happen once per worker instead of once per episode.

### 5.2 Persistent Python worker pool

`train.py` used to create a `spawn`-based `ProcessPoolExecutor` for every
generation, and every job launched a new simulator process. It now creates one
`ThreadPoolExecutor` for the whole run. Each worker thread owns one
`env.Evaluator` (a persistent `--evaluate-server` process) and runs a warm-up
episode so level parsing is not charged to the first real job.

Threads are sufficient because all simulation state lives in the Rust process;
the Python side only writes a weight vector and parses one short JSON line, so
it releases the GIL while blocked on the pipe. This also avoids pickling the
430-float genome and the config for every job, and avoids tearing down and
recreating worker processes every generation.

`Evaluator` restarts its server and retries once if the process dies; because
each request resets the game, a retry is safe.

### 5.3 Headless build no longer depends on `macroquad`

- `macroquad` is now an optional dependency behind a default `gui` feature.
- Keyboard/mouse handling moved from `game.rs` to a new `src/input.rs`, which
  only the windowed binary includes.
- `render.rs` is no longer compiled into `doom_headless`.
- `add_impact_effect` moved from `render.rs` to `game.rs` because the simulation
  creates hit effects even when it does not draw them.
- `exec_mgmt.sh -b` builds `doom` normally and `doom_headless` with
  `--no-default-features`.

On macOS the headless binary went from ten linked frameworks to only
`libSystem`, and it no longer requires a graphics/display stack on a server.

### 5.4 Wall grid cache and fast line-of-sight

`Game` now owns a `[bool; MAP_WIDTH * MAP_HEIGHT]` wall mask, rebuilt once per
level load (192 tile tests). `is_wall`, enemy movement, and line-of-sight checks
use that array instead of going through the level registry, string rows, and
tile parsing. Each check becomes a bounds test plus one byte load.

This is especially relevant to the Opteron target: four 12-core sockets share a
comparatively small memory bandwidth budget, and removing indirect string/list
access from every ray and LOS sample reduces both memory traffic and cache
pressure per core.

### 5.5 Allocation-free policy encoding

`agent_policy_action` now:

- builds the 42 inputs in a fixed `[f32; 42]` instead of a growing `Vec`
- finds the two nearest living enemies in one pass instead of allocating and
  sorting a list
- finds the nearest pickup in one pass
- writes pickup one-hot values directly instead of comparing strings
- uses the cached wall grid for the 8-ray depth scan
- uses slice-based dot products for the 10×43 weight matrix

The policy math and output selection are unchanged, including the `tanh`
activation and weapon tie-breaking.

### 5.6 Kill counter instead of alive scans

`Game` increments a `kills` counter when an enemy's HP reaches zero. The episode
loop reads the counter before and after a step instead of counting living
enemies twice per step.

### 5.7 Visual-only work disabled in headless mode

`set_headless()` turns off:

- the HUD danger scan and `danger_angle`
- hit-effect timers and impact spawns
- recoil, muzzle-flash, damage-flash, and message timers

These are presentation state only. The simulation-side effects they could have
had (e.g. damage) are handled by their own counters. The danger scan alone was
12.9 s of the baseline profile.

### 5.8 Build profile

The release profile already used `opt-level = 3`, thin LTO, and
`codegen-units = 1`; `panic = "abort"` and `strip = true` were added. The
headless target is built without the GUI feature.

For a target server, `RUSTFLAGS="-C target-cpu=native"` can be used when the
binary is built and run on the same machine. It is not enabled by default
because it makes the binary non-portable between the Xeon and Opteron hosts.
The Opteron 6172 has no AVX, while the E5-2643 v2 does; a native build picks the
right instruction set for each.

---

## 6. Results after optimization

Same workload, same checkpoint, same 50 generations, same profiling setting.

| Component | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Summed generation wall | 91.16 s | **16.58 s** | **5.50×** |
| Episodes per second | 52.7 | 289.6 | 5.50× |
| Summed episode wall | 600.14 s | 102.84 s | 5.84× |
| Non-simulation overhead | 469.00 s (78.1%) | 8.49 s (8.3%) | 55.2× |
| Rust `main` | 131.13 s | 94.35 s | 1.39× |
| Rust episode evaluation | 128.18 s | 93.94 s | 1.36× |
| Policy inference | 65.27 s | 44.17 s | 1.48× |
| Simulation step | 51.33 s | 39.14 s | 1.31× |
| `update()` | 37.50 s | 25.34 s | 1.48× |
| Enemy AI | 15.77 s | 14.24 s | 1.11× |
| Danger scan | 12.91 s | 1.57 s | 8.21× |
| Pickups | 1.78 s | 1.92 s | 0.93× |
| Movement | 4.12 s | 3.92 s | 1.05× |
| Firing | 2.71 s | 2.50 s | 1.09× |
| Kill accounting | 3.47 s | 3.06 s | 1.13× |
| Mean generation wall | 1.823 s | 0.332 s | 5.50× |

Per episode:

| Component | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Episode wall | 125.03 ms | 21.42 ms | 5.84× |
| Startup / transport | 97.71 ms | 1.77 ms | 55.2× |
| Rust simulation | 27.32 ms | 19.66 ms | 1.39× |
| Policy | 13.60 ms | 9.20 ms | 1.48× |
| Simulation step | 10.69 ms | 8.15 ms | 1.31× |

Whole-run process accounting (`/usr/bin/time`, one invocation per side):

| | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Real time | 91.33 s | 16.76 s | 5.45× |
| User CPU | 495.36 s | 86.67 s | 5.72× |
| System CPU | 122.42 s | 1.23 s | 99.5× |
| CPU utilization | 676% | 524% | — |

The system-CPU collapse is the clearest evidence that process setup was the
problem: fork/exec, dynamic loading, page faults, and framework initialization
were the majority of the baseline cost on this machine.

The small regressions in pickups and the loop remainder are noise at the
sub-second level and in code paths whose absolute cost is under 0.1 µs/step.

Two of the "after" counters mostly measure their own `Instant::now()` calls:
the danger scan is disabled in headless mode, and kill accounting is a single
counter read. Their remaining 1.57 s and 3.06 s are therefore timer overhead, so
the speedups in those rows understate the work actually removed.

### 6.1 `startup_s` after optimization

In the after run, the profile column `startup_s` is defined as
`episode_wall − rust_main`. With persistent servers it no longer measures
process startup; it measures pipe round-trip, Python JSON handling, and thread
scheduling. It fell from 97.71 ms to 1.77 ms per episode. Worker startup itself
(process launch, level parsing, first game allocation) happens once per pool
worker outside the generation loop.

---

## 7. Correctness validation

Two independent checks were used.

### 7.1 Same 50 generations, same results

The optimized trainer was run from `debug/training/output/baseline_checkpoint_gen99.json`
for generations 100–149 and compared with the baseline run:

- 50 generations × 96 episodes = 4,800 episodes are represented in the summaries
- 50/50 generation summaries match
- `best_fitness`, `mean_fitness`, `levels`, `kills`, `health`, `seconds`, and
  `start_level` are identical for every generation
- total steps are identical: 18,429,672

### 7.2 Same episodes, same scores

Every genome of the generation-99 population was evaluated once against the
preserved baseline binary (`debug/bench/baseline_doom_headless`) and once
through the optimized server. All 48 scores matched exactly. The same check
passes for generation 100 with both evaluation seeds (96 episodes), which is
the controlled comparison used in section 8.

The reused `Game` is therefore behaviourally equivalent to a fresh process.
The windowed game still builds and uses the original rendering/input path; only
the headless target omits it.

---

## 8. Compute-only comparison

Process startup can hide compute improvements, so the same 96 episodes were
also replayed **sequentially** (no concurrency) on generation 100:

| Metric | Baseline binary | Optimized server | Speedup |
| --- | ---: | ---: | ---: |
| Wall time for 96 episodes | 8.27 s | 1.96 s | 4.23× |
| Rust evaluation time | 2.657 s | 1.930 s | 1.38× |
| Per-episode Rust time | 27.7 ms | 20.1 ms | 1.38× |
| Startup / transport per episode | 58.5 ms | 0.31 ms | 189× |
| Policy | 1.635 µs/step | 1.070 µs/step | 1.53× |
| Simulation step | 1.319 µs/step | 1.011 µs/step | 1.31× |
| `update()` | 0.908 µs/step | 0.618 µs/step | 1.47× |
| Enemy AI | 0.355 µs/step | 0.310 µs/step | 1.14× |
| Danger scan | 0.299 µs/step | 0.044 µs/step | 6.73× |
| Movement | 0.130 µs/step | 0.118 µs/step | 1.10× |
| Firing | 0.0768 µs/step | 0.0702 µs/step | 1.09× |
| Total measured Rust | 3.270 µs/step | 2.376 µs/step | 1.38× |

So the algorithmic simulation work improved by about **1.38×**, while the
end-to-end training run improved by **5.5×**. The remaining gap is the removed
process startup, which was the point of the exercise.

The higher per-step numbers in section 6 (3.54 → 2.40 µs for policy) are the
same code measured under 8-way concurrency, where cores contend for shared
caches and memory. The relative improvement is similar; the absolute values are
not comparable across the sequential and concurrent tables.

---

## 9. Scaling to the target servers

### 9.1 Why the new architecture fits high core counts

- Every episode is independent. After warm-up there is no shared mutable state,
  no locks, and no per-generation process creation.
- The number of OS processes is `cpu_threads` (one evaluator each), not the
  number of episodes. The old design created and destroyed 4,800 processes over
  50 generations (96 per generation); more cores only meant more of those
  processes were alive at the same time, each paying its own startup cost.
- Memory is bounded by the worker count: one small Rust process per worker
  (roughly 12 MB resident in the measurements, plus a 256 KB frame buffer that
  could be removed entirely in a later change). 48 workers is well under 1 GB.
- Profiling is opt-in, so production runs do not pay the ~10% instrumentation
  cost.

### 9.2 Measured worker scaling

20 generations, profiling off, same workload, on the 8-thread development
machine:

| Workers | Wall time | Episodes/s | Scaling |
| ---: | ---: | ---: | ---: |
| 1 | 15.06 s | 127.5 | 1.00× |
| 2 | 8.53 s | 225.1 | 1.77× |
| 4 | 6.22 s | 308.8 | 2.42× |
| 8 | 5.27 s | 364.2 | 2.86× |
| 12 | 5.46 s | 351.8 | 2.76× |

The machine has only 4 physical cores, so the curve flattens at 8 threads and
falls back with 12. On the target machines, `cpu_threads: 0` resolves to:

- dual E5-2643 v2: 24 hardware threads (12 cores × 2 SMT)
- quad Opteron 6172: 48 real cores (4 × 12, no SMT)

Expect the aggregate episode rate to keep increasing with cores until memory
bandwidth or the memory subsystem saturates. The Opteron machine has more cores
but much lower per-core throughput and shared DDR3 bandwidth, so its scaling
ceiling will be lower than a core-count ratio would suggest. The per-core work
reductions in sections 5.4–5.7 (cached wall grid, fewer allocations, no visual
scan) directly reduce that bandwidth pressure.

### 9.3 Recommended server settings

```json
{
  "cpu_threads": 0,
  ...
}
```

- Leave `DOOM_PROFILE` unset for production training (about 10% faster).
- Use `cpu_threads: 0` to use all logical CPUs, or set 24 / 48 explicitly to
  reserve headroom for the OS.
- Build on the machine that will run the job. If moving the binary between the
  Xeon and Opteron hosts, do not use `target-cpu=native`.
- No display, GPU, X11, or OpenGL libraries are required by the headless binary
  after the `macroquad` decoupling.

### 9.4 Expected effect on training wall time

The old pipeline's cost per episode was roughly
`startup + fixed_simulation + steps × per_step`. On the development machine,
startup was 98 ms and simulation was 27 ms. On a server, process startup is
cheaper than macOS framework initialization, but it is still paid per episode
and it is paid on every core simultaneously. If startup is only 5–10 ms on
Linux, the old design still wastes 5–10 ms per episode across 24–48 workers,
and it repeatedly re-parses levels and re-allocates the game.

The new design pays that cost once per worker. The remaining cost is simulation
work, which scales with cores. In practice, expect the largest gains on the
high-core machines precisely because process startup was a per-episode serial
cost that could not be amortized by adding cores.

---

## 10. Reproducing the measurements

From `debug/`:

```sh
# Build the windowed game and the headless evaluator.
./exec_mgmt.sh -b

# The current checkpoint is at generation 149 after the baseline run.
# To replay generations 100-149, start from the preserved generation-99
# checkpoint and use a clean output directory:
mkdir -p bench/repro/training/output
cp training/output/baseline_checkpoint_gen99.json bench/repro/training/output/checkpoint.json
python3 - <<'PY'
import json
cfg = json.load(open("config.json"))
cfg["output_directory"] = "bench/repro/training/output"
json.dump(cfg, open("bench/repro/config.json", "w"), indent=2)
PY
DOOM_PROFILE=1 python3 training/train.py --config bench/repro/config.json
```

To continue the real training run instead, simply run
`DOOM_PROFILE=1 python3 training/train.py`; it resumes from
`training/output/checkpoint.json`.

Single-shot evaluation:

```sh
DOOM_PROFILE=1 doom/target/release/doom_headless --evaluate \
  10 medium 0.05 15000 1000 100 2 20 0.25 < weights.txt
```

Persistent server (one job per line):

```sh
DOOM_PROFILE=1 doom/target/release/doom_headless --evaluate-server < jobs.txt
```

Benchmark artifacts used for this document:

| Path | Contents |
| --- | --- |
| `bench/baseline_run/` | runnable baseline replay: preserved `env.py`/`train.py` (only the binary path changed), baseline config, and the clean 50-generation output |
| `bench/baseline_run/output/profile.csv` | clean 50-generation baseline replay used for the "before" tables |
| `bench/baseline_run/output/metrics.csv` | metrics from the replay; identical to the original run |
| `bench/baseline_clean_run.log` | clean baseline run log and `/usr/bin/time` output |
| `training/output/profile.csv` | original required profiling run (rows 100–149) |
| `training/output/metrics.csv` | full metric history through generation 149 |
| `training/output/baseline_checkpoint_gen99.json` | clean start state |
| `bench/profile_run.log` | log of the original baseline profiling run |
| `bench/baseline_doom_headless` | preserved baseline binary (instrumented, pre-optimization) |
| `bench/baseline_env.py.txt`, `bench/baseline_train.py.txt` | preserved baseline Python sources |
| `bench/scratch_opt50/training/output/profile.csv` | optimized 50-generation profile |
| `bench/optimized_run.log` | optimized run log and `/usr/bin/time` output |
| `bench/baseline_profile_50gen.json` | aggregated clean-baseline totals |
| `bench/optimized_profile_50gen.json` | aggregated optimized totals |
| `bench/scaling/` | worker-scaling runs |
| `bench/scaling_profile/` | profiling-overhead run |

Note: the original `training/output/profile.csv` combines a generation-100 smoke
run with the 49-generation continuation, so its `profile_summary.json` covers 49
generations. The tables in this document use the single-invocation 50-generation
replay in `bench/baseline_run/`, which reproduced the same metrics exactly.

---

## 11. Files changed in `debug/`

| File | Change |
| --- | --- |
| `doom/src/game.rs` | profile timers, wall-grid cache, fast wall/LOS helpers, allocation-free policy, kill counter, headless flags, per-episode reset, impact effects moved in |
| `doom/src/bin/doom_headless.rs` | profiling, shared `run_episode`, `--evaluate`, `--evaluate-server` |
| `doom/src/input.rs` | new: keyboard/mouse handling for the GUI build |
| `doom/src/main.rs` | includes `input` |
| `doom/src/render.rs` | impact effects moved out; still used by the GUI build |
| `doom/src/levels.rs` | reference tile/LOS helpers retained with `allow(dead_code)` |
| `doom/Cargo.toml` | optional `macroquad` behind `gui`, `panic = "abort"`, strip |
| `exec_mgmt.sh` | builds the GUI and headless binaries separately |
| `training/env.py` | profiling wrapper, persistent `Evaluator` client, one-shot `evaluate` kept |
| `training/train.py` | persistent thread pool, profile CSV/summary, stage timings |
| `config.json` | `generations: 150`, `profile_file`, `profile_summary_file` |
| `README.md` | updated headless protocol, worker model, profiling, scripts |

---

## 12. Limitations and possible next steps

- All timings are from a 4-core/8-thread Mac. Absolute server numbers will
  differ; use the included scaling harness on the target hardware before
  committing to a worker count.
- Process startup on Linux is cheaper than on macOS, so the 55.2× startup
  reduction measured here overstates the Linux gain. The persistent-server
  design still removes per-episode fork/exec, level parsing, and allocation on
  every platform, and the compute reductions are platform-independent.
- Policy inference is now the largest single Rust cost. A SIMD dot product or a
  faster `tanh` approximation could reduce it further, but both risk changing
  policy outputs and therefore training trajectories. They were left out to
  preserve exact behavioural equivalence.
- The 320×200 RGBA frame buffer is still allocated in the headless game. It is
  small (256 KB per worker) and never touched, but removing it would slightly
  reduce memory use at 48 workers.
- Episode load balancing is uneven because episode length varies from a few
  hundred to 15,000 steps. With 2 seeds per genome and 96 jobs per generation,
  a 24- or 48-thread server has only 4 or 2 jobs per worker, so tail effects can
  matter. Increasing the worker count beyond the population × episodes count
  cannot help; if needed, change `evaluation_episodes` or split work at a
  finer grain in a future revision.
