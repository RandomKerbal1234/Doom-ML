# DOOM machine learning environment

The training loop evolves a small policy from random weights. Each evaluation
starts at a level drawn uniformly from the configured `start_level` range, then
attempts up to `max_levels_per_agent`
exits in campaign order, wrapping from the final level to the first. Fitness
rewards level completion, shortest-path progress toward each exit, damage dealt,
survival, and efficient ammo and time use. The Rust
headless executable runs the same simulation code as the game, without a
window.

## Policy

The policy is a fully connected `tanh` multilayer perceptron whose shape is data,
not code: `hidden_layers` in `config.json` lists the hidden widths, and the total
weight count follows from it. The default `[16, 8]` gives
`64 → 16 → 8 → 10` and 1266 parameters; an empty list reproduces the original
single-layer linear policy. Adding, removing, or resizing a hidden layer only
changes `hidden_layers` and needs `--fresh`.

The 64 inputs are wall rays, the two nearest enemies, the nearest pickup, the
exit, health/armor/ammo/unlocked/weapon/time, plus two recurrent blocks: a leaky
average of the wall rays and the previous action (turn, forward, strafe, fire,
exit, run). The recurrent blocks give the policy a short memory of where it has
been; they are reset at the start of every episode.

Weights are layer-major: for each layer, every output neuron stores its `fan_in`
input weights followed by its bias. `initial_weight_sigma` is the target
standard deviation of a neuron's *pre-activation*, so each layer is scaled by
`1/sqrt(fan_in)`; a flat per-weight sigma would saturate the wide input layer.
`mutation_sigma` is normalized the same way. When the best fitness stops
improving for `mutation_adapt_patience` generations the mutation radius grows by
`mutation_adapt_growth` (up to `mutation_adapt_max`) to escape a local optimum,
and resets on any improvement. The radius is a multiplier on every mutation
step size (not on `mutation_rate`); generation lines show it as `| radius 1.23`
only while it is above 1.0, and it applies to the offspring built after that
generation.

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
that many concurrent evaluations. Each episode runs its policy loop inside the
Rust headless executable, avoiding a Python/Rust JSON exchange on every
simulation step. The trainer uses a thread pool in which every worker thread
owns one long-lived `doom_headless --evaluate-server` process, so process
startup, dynamic linking, and level parsing happen once per worker instead of
once per episode. Episodes are scheduled independently so a population with
multiple evaluation episodes can use more workers than there are candidates.
Other settings control population and mutation,
simulation step size, run length, evaluation episodes, seed, reward weights, checkpointing, and
output paths. `doom_difficulty` selects Easy, Medium, or Hard game settings.
`max_levels_per_agent` caps how many level exits an episode can complete; it is
passed to the Rust evaluator instead of the old hard-coded ten-level limit.
`start_level` selects where episodes begin: an inclusive 1-based range
`[low, high]` (for example `[3, 5]` picks levels 3–5), where `[N, N]` pins every
episode to level `N`. `evaluation_episodes` is the number of episodes averaged
per genome per generation; larger values reduce fitness noise at a proportional
cost in wall time.

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
expected start level (drawn from the `start_level` range for that seed) for
every generation.

The best policy and population checkpoint are JSON files; generation summaries
are appended to a CSV whose first column is the run ID. Each generation also
appends a row to `profile.csv` (wall-clock stage timings plus Rust counters when
profiling is enabled) and refreshes `profile_summary.json` with totals and
percentages. Set `DOOM_PROFILE=1` to enable the Rust timers; instrumentation
costs roughly 10% wall time. The evaluation job is self-contained and has a seed
and policy, which leaves a clear seam for distributing evaluation jobs across
machines later. This version only runs testers on the local machine.

### Threads and CPU utilisation

Each evaluation runs in its own persistent `doom_headless --evaluate-server`
process, one per worker thread, so the simulator work is genuinely parallel. The
trainer's own per-generation work (ranking, mutation, checkpoint writing) is
single-threaded, and that is what limits utilisation on a many-core machine:
adding cores shortens the evaluation phase but not the serial one. Each
generation prints `eval busy N%`, and if the serial part exceeds 15% of the
wall time the trainer prints a `note:` suggesting the `evaluation_episodes`
value that would push it under 10%. Generating more episodes per genome is also
the cheapest way to reduce fitness noise, so raising it is usually a win:

```json
"evaluation_episodes": 16,
"population_size": 64
```

With 24 hardware threads, roughly 10x more evaluation work than serial work per
generation is needed to keep every core busy, i.e. a few hundred episodes per
generation. `cpu_threads` sets the worker count (`0` = every available CPU, `-1`
= all but one); the trainer never starts more workers than there are jobs.

Weights reach the evaluator as a packed f32 blob rather than formatted text, and
the checkpoint stores the population as one base64 blob, because that work runs
on the Python interpreter lock while the simulator processes would otherwise be
idle. See `profile.csv` (`generation_s`, `eval_s`, `write_s`) when investigating
a slow run.

## Interface

`training/env.py` exposes `Evaluator`, the persistent `--evaluate-server` client
used by the trainer, and `HeadlessDoomEnv`, with `reset(level)` and
`step(action)`. `HeadlessDoomEnv` accepts a `max_levels` cap (default 10) for
interactive episodes. Observations contain player state, map, enemies, pickups,
and exit position. Actions contain normalized turn, forward and strafe values,
fire/exit flags, weapon index, and run flag. The trainer maps observations to
actions and evaluates evolved policies.

`observe(obs, state)` builds the policy input vector and `forward_policy(weights,
shape, inputs)` runs the network; `choose_action(weights, obs, state, shape)`
combines them. These are Python mirrors of `Game::agent_inputs` and
`Policy::forward` in `doom/src/game.rs` and must stay in sync with them. Pass a
`PolicyState` to `observe`/`choose_action` for the recurrent inputs, and reset it
per episode.

To check the two implementations against each other, compare the Python encoder
with the Rust probe:

```sh
# prints the inputs, outputs and shape for the first step of a level
doom/target/release/doom_headless --probe 0 medium --arch 64,16,8,10 < weights.txt
```
