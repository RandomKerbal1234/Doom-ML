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

## Configuration

Edit the root `config.json`. `cpu_threads` controls parallel episode evaluations:
`0` uses all detected CPUs, `-1` uses all but one, and a positive integer selects
that many concurrent evaluations. Each worker thread owns a long-lived
`doom_headless --evaluate-server` process, which runs the policy loop and the
whole episode inside Rust; the Python side sends one weight vector and receives
one JSON result per episode. Episodes are scheduled independently so a
population with multiple evaluation episodes can use more workers than there are
candidates.
Other settings control population and mutation,
simulation step size, run length, evaluation episodes, seed, reward weights, checkpointing, and
output paths. `doom_difficulty` selects Easy, Medium, or Hard game settings.
Training begins with random policies when no checkpoint exists;
with `resume` enabled, a saved checkpoint continues at the next generation.

The best policy and population checkpoint are JSON files; generation summaries
are appended to CSV. The evaluation job is self-contained and has a seed and
policy, which leaves a clear seam for distributing evaluation jobs across
machines later. This version only runs testers on the local machine.

## Profiling

Set `DOOM_PROFILE=1` to enable the Rust-side nanosecond timers. The trainer
always writes `profile.csv` (one row per generation) and `profile_summary.json`
(totals, per-generation means, and percentages); without the environment
variable the Rust columns are zero but wall-clock timings are still recorded.
Profiling costs roughly 10% wall time, so leave it unset for production runs.
See `../OPTIMIZATION.md` for the measured breakdown and the before/after
comparison.

## Interface

`training/env.py` exposes `HeadlessDoomEnv`, with `reset(level)` and `step(action)`.
Observations contain player state, map, enemies, pickups, and exit position.
Actions contain normalized turn, forward and strafe values, fire/exit flags,
weapon index, and run flag. The trainer maps observations to actions and
evaluates evolved policies.
