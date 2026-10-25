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

Edit the root `config.json`. `cpu_threads` controls parallel testers: `0` uses
all detected CPUs, `-1` uses all but one, and a positive integer selects that
many concurrent tester processes. Other settings control population and mutation,
simulation step size, run length, evaluation episodes, seed, reward weights, checkpointing, and
output paths. `doom_difficulty` selects Easy, Medium, or Hard game settings.
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
