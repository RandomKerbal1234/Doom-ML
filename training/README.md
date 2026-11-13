# DOOM machine learning environment

The training loop evolves a small policy from random weights. Each evaluation
starts at a uniformly random level, then attempts up to `max_levels_per_agent`
exits in campaign order, wrapping from the final level to the first. Fitness
rewards level completion, shortest-path progress toward each exit, damage dealt,
survival, and efficient ammo and time use. The Rust
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
smoke training can be run with `--generations 1`. Use `--fresh` to ignore an
existing checkpoint and start a new run explicitly.

While training is running, open a second terminal and run
`./watch_agent.sh`. It opens the DOOM GUI and follows the policy in
`training/output/best_policy.json`, reloading it when a new generation is saved.
Press **R** to restart with a random starting level or **Q** to close the viewer.
To watch another policy file, pass its path to the script.

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
`max_levels_per_agent` caps how many level exits an episode can complete; it is
passed to the Rust evaluator instead of the old hard-coded ten-level limit.

All thirteen reward weights in `fitness` are active. `level_completion`,
`partial_level`, `health_per_point`, `kill`, `health_gain`, and `armor_gain` add
score; `seconds`, `damage_taken`, `armor_damage_taken`, `ammo_spent`, and
`death` subtract score. `exit_progress` rewards positive reductions in distance
to the current exit each simulation step (it never penalizes moving away), and
`damage_dealt` rewards enemy health removed.

The configuration is schema-validated at startup. Unknown keys, missing required
keys, and out-of-range values stop the run rather than being silently ignored,
and the effective configuration is printed before training begins.

Training begins with random policies when no checkpoint exists. `resume` defaults
to false; when it is enabled, a saved checkpoint continues at the next
generation. Resuming refuses a checkpoint whose seed, behavioral configuration,
or executable fingerprint differs from the current run. `--fresh` overrides
`resume` and starts a new run.

Each run receives a run ID derived from its seed, configuration fingerprint, and
start time. The ID is written to every metrics row and to the checkpoint,
`best_policy.json`, and `run_manifest.json`. The manifest records the seed, full
configuration, Git commit and dirty state, Python and Rust versions, executable
SHA-256, start time, and resume state. A resume also verifies that the metrics
file starts at generation 0, contains only this run's rows, and has the
seed-derived start level for every generation.

The best policy and population checkpoint are JSON files; generation summaries
are appended to a CSV whose first column is the run ID. The evaluation job is
self-contained and has a seed and policy, which leaves a clear seam for
distributing evaluation jobs across machines later. This version only runs
testers on the local machine.

## Interface

`training/env.py` exposes `HeadlessDoomEnv`, with `reset(level)` and `step(action)`.
Its constructor accepts a `max_levels` cap (default 10) for interactive episodes.
Observations contain player state, map, enemies, pickups, and exit position.
Actions contain normalized turn, forward and strafe values, fire/exit flags,
weapon index, and run flag. The trainer maps observations to actions and
evaluates evolved policies.
