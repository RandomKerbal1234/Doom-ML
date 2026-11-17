# DOOM-ML

A small, asset-free DOOM-style first-person shooter written in Rust, plus a Python
neuroevolution trainer that learns to play it.

The repository contains two cooperating programs:

- **`doom`** — a playable desktop game built with `macroquad` and a 320×200
  software raycaster. It ships with a 50-level campaign, four weapons, two enemy
  types, pickups, difficulty presets, an in-game settings editor, and a
  "watch the agent" mode.
- **`doom_headless`** — a windowless build of the same game logic used by the
  trainer. It can be driven step-by-step over a JSON-lines protocol, evaluate a
  complete policy episode in a single process (`--evaluate`), or serve many
  episodes from one long-lived process (`--evaluate-server`).

> The `debug/` directory is a historical optimization benchmark snapshot. See
> [`debug/OPTIMIZATION.md`](debug/OPTIMIZATION.md) for the profiling
> methodology and measured before/after performance. The live code is at the
> repository root.

Training is from scratch: no human demonstrations and no neural-network
library. A policy is a flat list of weights for a configurable `tanh` MLP,
evolved with a simple elitist mutation hill-climber in parallel across CPU
cores. The current configuration uses a 64→48→32→16→12→10 network with 5,550
weights.

The **`visualizer`** app displays the network and its learned weights from the
latest `training/output/best_policy.json`. It watches for policy updates while
open; hover over nodes and connections to inspect them. Scroll over the network
to zoom up to 5000%, use `W` / `A` / `S` / `D` or the arrow keys to move the
view, or drag the network directly. Click a neuron to highlight its incoming
and outgoing connections; click blank space to clear the selection.

---

## Requirements

| Component | Requirement |
| --- | --- |
| Game / headless binary | Rust with edition 2024 support (Rust 1.85+); `cargo` |
| Trainer | Python 3.9+ (standard library only — no pip packages) |
| OS | macOS, Linux, or Windows. `DOOM.app` is macOS-only |

The game has no external assets: levels, textures, sprites, and the font are all
generated or embedded at compile time.

---

## Quick start

From the repository root:

```sh
# 1. Build the game and the headless evaluator (release mode)
./exec_mgmt.sh -b

# 2. Train (writes checkpoints and the best policy to training/output/)
python3 training/train.py

# 3. In a second terminal, watch the current best policy play
./watch_agent.sh

# 4. Or play the game yourself
./exec_mgmt.sh -r

# 5. Visualize the latest best-agent network
cd visualizer && cargo run --release
```

The visualizer defaults to `training/output/best_policy.json`; pass a policy
JSON path to inspect another agent. See
[`visualizer/README.md`](visualizer/README.md) for controls and details.

A short smoke run (single generation):

```sh
python3 training/train.py --build --generations 1 --fresh
```

Build automatically as part of training:

```sh
python3 training/train.py --build
```

Start a completely fresh training run (build, then delete all previous
output):

```sh
./ready.sh
```

> Note: `ready.sh` **deletes** `training/output/`. Use it only when you intend to
> discard the existing population and metrics.

---

## Repository layout

```text
doom-ml/
├── config.json                 # Trainer configuration (the main tuning file)
├── ready.sh                    # Build binaries, then clear training/output/
├── clear_training_data.sh      # Delete training/output/
├── exec_mgmt.sh                # Build / run the game (and both binaries)
├── watch_agent.sh              # Open the GUI and follow best_policy.json
├── visualizer/                 # Rust GUI for exploring the best policy network
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
├── debug/                      # Historical optimization benchmark snapshot (not live code)
├── training/
│   ├── train.py                # Evolution loop, checkpointing, metrics, CLI
│   ├── env.py                  # Evaluator client + observation/action encoding
│   ├── README.md               # Detailed trainer notes
│   └── output/                 # Created at runtime (checkpoint, policy, metrics, profile)
└── doom/
    ├── Cargo.toml              # Optional macroquad (default `gui` feature), release profile
    ├── build.rs                # Embeds every src/levels/*.lvl file at compile time
    ├── run.sh                  # Convenience: ./target/release/doom
    ├── DOOM.app/               # macOS app-bundle launcher (macOS only)
    └── src/
        ├── main.rs             # Window, main loop, rendering surface, watch mode
        ├── game.rs             # Simulation: player, enemies, weapons, pickups, AI hooks
        ├── input.rs            # Keyboard/mouse handlers (GUI build only)
        ├── render.rs           # Raycaster, sprites, weapon view, HUD (GUI build only)
        ├── settings.rs         # Tunable stats, difficulty presets, settings UI data
        ├── levels.rs           # .lvl parser, validation, embedded level registry
        ├── ui.rs               # Pause/help, death/victory, settings overlays
        ├── bin/doom_headless.rs # Headless protocol, --evaluate, --evaluate-server
        └── levels/*.lvl        # 50 campaign levels (01_e1m1.lvl … 50_e5m10.lvl)
```

---

## Playing the game

Launch the windowed game with `./exec_mgmt.sh -r`, `cargo run --release` from
`doom/`, or by opening `doom/DOOM.app` on macOS. The app bundle is a thin shell
script that runs `doom/target/release/doom` and shows an alert if it has not been
built yet.

The game opens **paused** on a controls and level-selection screen.

### Controls

| Input | Action |
| --- | --- |
| `W` / `S` | Move forward / backward |
| `A` / `D` | Strafe left / right |
| `←` / `→` | Turn |
| `Shift` | Run (faster movement and turning) |
| `Ctrl`, `Space`, or left mouse | Fire |
| `1`–`4` / mouse wheel | Select a collected weapon |
| `E` | Use the green exit portal when nearby |
| `F1` (or `Fn+F1`) / `Esc` | Pause and show help / level select |
| `Shift+P` | Open the settings menu |
| `↑` / `↓` | Choose a level in the pause screen |
| `Enter` | Start the selected level |
| `Q` | Quickplay from level 1 and advance through exits |
| `Esc` / `Space` / `F1` | Resume from the pause screen |
| `F10` | Quit from the pause / level-select screen |
| `R` | Restart after death or after finishing the campaign |

Exiting the last level shows **MISSION COMPLETE**; losing all health shows
**YOU DIED**. Both screens restart with `R`.

### Gameplay

- The campaign has **50 levels** (`E1M1` … `E5M10`), played in filename order.
  Exits advance to the next level; the final exit wins the game.
- Four weapons: pistol (always available), shotgun, chaingun, and rocket
  launcher. Later weapons are unlocked by picking them up and need their own
  ammo. The rocket launcher also has a blast radius.
- Two enemy types: **imp** and **demon**. They chase the player, damage them
  when in range with line of sight, and have separate health, speed, damage,
  attack delay, and stopping-distance stats.
- Pickups: `shotgun`, `chaingun`, `rocket_launcher`, `medkit`, `armor`, and
  `ammo` caches.
- Armor absorbs a configurable fraction of incoming damage before health.
- A pulsing **DEMON IN RANGE** alert and an edge arrow appear when a demon has
  a clear shot at the player.
- The HUD shows health, armor, current ammo, level code, current weapon, and
  transient messages.

### Settings menu

Open with `Shift+P`. Use `↑`/`↓` to select a value, `←`/`→` to change it,
`Tab` / `Page Up` / `Page Down` to switch sections, and `1` / `2` / `3` to apply
Easy / Medium / Hard. Editing any value switches the preset label to **Custom**
and takes effect immediately (including on enemies already in the level).

The four sections are:

1. **Player** — current/max health, current/max armor, movement and turning speeds, player radius, and armor absorption.
2. **Weapons** — damage, range, spread, fire delay, ammo cost, and rocket blast radius for all four weapons.
3. **Demons** — health, speed, chase/attack range, stopping distance, damage, and attack delay for imps and demons.
4. **Supplies** — current/starting ammo, medkit/armor amounts, pickup radius, and ammo granted by weapon and ammo pickups.

The default preset is **Medium**.

### Watch-the-agent mode

```sh
./watch_agent.sh [path/to/policy.json]
```

This runs `doom --watch-agent <policy>` (default:
`training/output/best_policy.json`). While running:

- The policy starts on a random level and plays by itself.
- The policy file is re-read every 0.5 seconds. When its generation, difficulty,
  or modification time changes, the new weights are picked up automatically.
- The top banner shows the generation and completed-level count.
- `R` restarts with a random level, `Q` quits.
- If the policy file is missing, the viewer waits and displays a message.

---

## Game engine internals

- **Rendering:** `render.rs` raycasts walls into a 320×200 RGBA buffer with a
  per-column z-buffer, draws floor/ceiling, billboard sprites, hit effects, the
  first-person weapon, the danger arrow, and the HUD. `main.rs` uploads the
  buffer to a texture and scales it, letterboxed, into a resizable window.
- **Simulation:** `game.rs` is the single source of truth for movement,
  collision (per-axis sliding), shooting, enemy pursuit/attacks, pickups,
  damage, level transitions, pause/death/win state, and the agent observation
  and policy-action functions. The windowed game and `doom_headless` compile the
  exact same `game.rs`, so the trainer optimizes the real game. `input.rs` and
  `render.rs` are GUI-only; the headless build excludes them.
- **Levels:** `build.rs` scans `doom/src/levels/` at build time, embeds every
  `.lvl` file in filename order, and regenerates the manifest when a level
  changes. `levels.rs` parses and validates them into a `OnceLock` registry.
  `game.rs` caches the solid tiles in a wall grid when a level loads, so
  collision, line of sight, and agent rays avoid re-reading the level strings.
- **Build profile:** release build with `opt-level = 3`, thin LTO,
  `codegen-units = 1`, `panic = "abort"`, and symbols stripped. `macroquad` is
  optional behind the default `gui` feature, so `doom_headless` does not link
  the windowing stack.

### Level format

Each `.lvl` file is a small INI-like text file. The map must be exactly
**16 columns × 12 rows**.

```ini
code=E1M1
title=HANGAR 13
spawn=2.5,2.5,0.0
exit=14.5,10.5

[map]
################
#......#.......#
#......#..##...#
#......#.......#
#..##..#.......#
#......#.......#
#..............#
#...#..........#
#...#..........#
#..............#
#.............X#
################

[enemies]
5.5,3.5,imp
10.5,8.5,demon

[pickups]
8.5,6.5,shotgun
5.5,6.5,ammo
12.5,8.5,medkit
```

- `#`, `B`, `M` are solid walls; `.` and `X` are walkable floor (`X` is purely
  a visual marker — the actual exit coordinate comes from `exit=`).
- `spawn` is `x,y,angle-in-radians`; coordinates are map positions, normally
  tile centers such as `2.5,2.5`.
- Enemy types: `imp`, `demon`. Pickup types: `shotgun`, `chaingun`,
  `rocket_launcher`, `medkit`, `armor`, `ammo`.
- `;` and `//` start comments. Spawn, exit, enemies, and pickups must be on open
  tiles; level codes must be unique.
- Every level has at least one medkit, and levels with five or more enemies have
  at least two (the 50-level campaign currently has 312 enemies and 91 medkits).
- Editing levels requires a rebuild; the game never reads these files at runtime.

See [`doom/src/levels/README.md`](doom/src/levels/README.md) for the full format.

---

## Machine learning trainer

### Policy representation

The policy is a one-layer network with **42 inputs**, **10 outputs**, and a
bias per output:

```text
output[j] = tanh( bias[j] + Σ input[i] * weight[j][i] )
```

That gives `10 × (42 + 1) = 430` weights, stored row-major per output (bias
last). There is no hidden layer and no external ML dependency; evolution tunes
the weights directly.

**42 observation inputs** (`game.rs::agent_policy_action`, mirrored by
`env.py::encode_observation`):

| Count | Feature |
| --- | --- |
| 8 | Wall distance in 8 directions around the player (normalized by 8, capped at 1) |
| 10 | Two nearest enemies × (sin bearing, cos bearing, distance/12, hp/200, kind) |
| 9 | Nearest pickup × (sin bearing, cos bearing, distance/12, one-hot of the 6 pickup types) |
| 3 | Exit × (sin bearing, cos bearing, distance/16) |
| 2 | Health/200, armor/200 |
| 4 | Ammo for each weapon / 100 |
| 4 | Weapon-unlocked flags |
| 2 | Current weapon/3, episode time/120 |

**10 outputs:**

| Output | Meaning |
| --- | --- |
| 0 | Turn rate (clamped to [-1, 1]) |
| 1 | Forward/backward (clamped to [-1, 1]) |
| 2 | Strafe (clamped to [-1, 1]) |
| 3 | Fire when > 0 |
| 4 | Activate exit when > 0 |
| 5–8 | Weapon preference scores (highest unlocked wins) |
| 9 | Run when > 0 |

### How a generation runs

1. The trainer loads `config.json` and validates it.
2. It creates or resumes a population of genomes (initially Gaussian,
   `initial_weight_sigma` around zero).
3. For each generation it builds one evaluation job per
   `(genome, evaluation_seed)` pair. Seeds are deterministic:
   `seed + generation × evaluation_episodes + i`.
4. Jobs run concurrently through a long-lived `ThreadPoolExecutor`. Each worker
   thread owns one `doom_headless --evaluate-server` process and sends it one
   job line per episode; the Rust process runs the full episode internally and
   returns one JSON line (no per-step Python/Rust round-trip). Keeping the
   process alive avoids re-launching the simulator and re-parsing levels for
   every episode.
5. Episode results are averaged per genome; genomes are ranked by mean fitness.
6. The best policy is written to `best_policy.json`, the population and RNG
   state to `checkpoint.json`, and a row is appended to `metrics.csv` and
   `profile.csv`.
7. The next population is formed from the top `elite_count` genomes (copied
   unchanged) plus mutants. Each mutant is a copy of a parent sampled from the
   top `parent_pool_size`, with every weight independently perturbed by
   `N(0, mutation_sigma)` with probability `mutation_rate`.
8. Training continues until `generations` total generations have been run.

Because the RNG state is checkpointed, a resumed run reproduces the same
sequence of mutations and evaluation seeds. The trainer also refuses to resume
if the behavioral config, seed, or executable hash has changed; use `--fresh`
to start over.

### Episode evaluation and fitness

Each episode:

- Starts on a level drawn uniformly from the configured `start_level` range
  `[low, high]` (1-based, inclusive; `[N, N]` pins the level), deterministically
  from the evaluation seed. All genomes in a generation share the same start
  level.
- Applies one of the difficulty presets (`easy`, `medium`, `hard`).
- Steps at `time_step_seconds` (max 0.05) for up to `max_steps` steps
  (default 15,000 steps ≈ 750 simulated seconds).
- Ends on death, after completing `max_levels_per_agent` levels (default 50), or
  at the step limit. Exiting a level wraps to level 1 after level 50.

The score is accumulated in `doom/src/bin/doom_headless.rs::run_episode` with
all 13 weights from `config.json`:

- Every step: `+ max(previous_exit_distance − current_exit_distance, 0)
  × exit_progress` (moving away never penalizes).
- On each completed level: `+ level_completion + health × health_per_point
  + level_kills × kill − time_in_level × seconds`.
- If the episode ends before the level cap: `+ levels_completed × partial_level
  + health × health_per_point + level_kills × kill − time_in_level × seconds`.
- At the end: `+ damage_dealt × damage_dealt − damage_taken × damage_taken
  − armor_damage_taken × armor_damage_taken + health_gain × health_gain
  + armor_gain × armor_gain − ammo_spent × ammo_spent`, and
  `− death` if the player died.

`--evaluate` prints one CSV line:
`score,levels_completed,total_kills,health,total_time,start_level`;
`--evaluate-server` returns the same fields as JSON. The Python side then
reports the mean fitness, levels, kills, health, and seconds across the
configured evaluation episodes.

### Parallelism

`cpu_threads` in `config.json` controls the worker pool:

| Value | Behavior |
| --- | --- |
| `0` | Use every detected CPU (default) |
| `-1` | Use all CPUs except one |
| positive integer | Use exactly that many workers |
| `< -1` | Error |

Episodes are scheduled independently of candidates, so a population of 48 with
2 evaluation episodes produces 96 evaluation jobs per generation (at most
`cpu_threads` run at once). Each worker thread keeps one headless process alive,
so memory use scales with the worker count rather than with the number of jobs.
A crashed evaluator is restarted and its episode retried once.

### Running the trainer

```sh
python3 training/train.py [--config PATH] [--build] [--generations N] [--fresh]
```

- Run it as a script (`python3 training/train.py`), not as a module: `train.py`
  imports `env.py` from its own directory.
- `--config` selects an alternative JSON config (relative paths inside the
  config are resolved against the repository root).
- `--build` invokes `./exec_mgmt.sh -b` first.
- `--generations` overrides the configured generation count for this run.
- `--fresh` ignores an existing checkpoint and starts a new run (it does not
  delete old files).

The headless binary must exist at `doom/target/release/doom_headless`; training
exits with a build hint if it is missing. (`doom` itself is only needed for
playing and for watch mode.)

Progress is printed per generation, for example:

```text
generation 12/100 | testers 8 | best 4310.2 | levels 3 | kills 7 | hp 62 | 71.4s
```

### Outputs and resuming

Everything is written under `output_directory` (default `training/output/`):

| File | Contents |
| --- | --- |
| `checkpoint.json` | Generation index, full population, best genome/result, Python RNG state, provenance |
| `best_policy.json` | Best genome plus architecture metadata (`generation`, `doom_difficulty`, `result`, `weights`) — this is what the game's watch mode loads |
| `metrics.csv` | One row per generation, first column `run_id`: `generation, best_fitness, mean_fitness, levels, kills, health, seconds, start_level` |
| `run_manifest.json` | Run identity, config, environment, resume state, and lifecycle `status` |
| `profile.csv` | Per-generation wall-clock and (optional) Rust timing breakdown |
| `profile_summary.json` | Aggregated timing totals and percentages |

With `"resume": true`, an existing checkpoint continues at the next generation
and metrics are appended. The trainer validates the checkpoint's config
fingerprint, seed, executable hash, manifest, and metrics before continuing, and
refuses to resume if anything behavioral has changed — including `seed`,
`start_level`, the fitness weights, level caps, difficulty, or output paths. With
`"resume": false` (or
`--fresh`), training starts a fresh random population and the metrics file is
truncated.

To train longer, increase `generations` and keep `resume` enabled. To start
over, delete `training/output/` (`./clear_training_data.sh`) or pass `--fresh`.

Set `DOOM_PROFILE=1` to enable the Rust-side timers. This adds a `profile`
object to each evaluator result and populates the Rust columns of `profile.csv`
and `profile_summary.json`; instrumentation costs roughly 10% wall time.

### Configuration reference

`config.json` at the repository root:

| Key | Default | Meaning |
| --- | --- | --- |
| `cpu_threads` | `0` | Worker threads: `0` all, `-1` all-but-one, else count |
| `generations` | `500` | Total generations (resume counts toward this) |
| `population_size` | `128` | Genomes per generation |
| `elite_count` | `12` | Best genomes copied unchanged into the next generation |
| `parent_pool_size` | `32` | Parents are sampled from this many top genomes |
| `initial_weight_sigma` | `0.35` | Std-dev of the initial random weights |
| `mutation_rate` | `0.08` | Probability that any individual weight mutates |
| `mutation_sigma` | `0.18` | Std-dev of a weight mutation |
| `time_step_seconds` | `0.05` | Simulation step; must be in `(0, 0.05]` |
| `max_steps` | `15000` | Step cap per evaluation episode |
| `max_levels_per_agent` | `50` | Completed level transitions allowed per episode (≤ 50) |
| `evaluation_episodes` | `2` | Episodes averaged per genome per generation |
| `doom_difficulty` | `"medium"` | `easy`, `medium`, or `hard` |
| `start_level` | `[1, 50]` | Inclusive 1-based level range `[low, high]`; each episode picks uniformly from it (`[N, N]` pins the level) |
| `campaign_levels` | `50` | Must equal 50 (validated) |
| `seed` | `641231456` | Master seed for the population and episode schedules |
| `resume` | `true` | Continue from `checkpoint.json` when present |
| `output_directory` | `"training/output"` | Output directory |
| `checkpoint_file` | `"checkpoint.json"` | Population checkpoint name |
| `best_policy_file` | `"best_policy.json"` | Best-policy name |
| `metrics_file` | `"metrics.csv"` | Metrics CSV name |
| `manifest_file` | `"run_manifest.json"` | Run manifest name |
| `profile_file` | `"profile.csv"` | Timing CSV name |
| `profile_summary_file` | `"profile_summary.json"` | Timing summary name |
| `fitness.*` | see above | All 13 reward/penalty weights are used |

---

## Headless interface

### Step-by-step protocol (`doom_headless` with no arguments)

The process reads commands from stdin, one per line, and writes one JSON line
per command to stdout:

```text
reset,<level_index>,<easy|medium|hard>
step,turn,forward,strafe,fire,activate_exit,weapon,run,dt
quit
```

- `level_index` is `0`-based and wraps modulo 50.
- `turn`/`forward`/`strafe` are in `[-1, 1]`; `fire`, `activate_exit`, and
  `run` are truthy when non-zero; `dt` is clamped to `[0, 0.05]`.
- Each command replies with the full observation JSON:

```json
{
  "level": 0, "x": 2.5, "y": 2.5, "angle": 0.0,
  "health": 100, "armor": 0, "ammo": [50, 0, 0, 0],
  "unlocked": [true, false, false, false], "weapon": 0, "time": 0.0,
  "dead": false, "exit": [14.5, 10.5],
  "map": ["################", "..."],
  "enemies": [{"x": 5.5, "y": 3.5, "hp": 60.0, "kind": 0}],
  "pickups": [{"x": 8.5, "y": 6.5, "kind": "shotgun"}]
}
```

### Batch evaluation (`--evaluate`)

```sh
doom/target/release/doom_headless --evaluate \
  START_LEVEL DIFFICULTY DT MAX_STEPS MAX_LEVELS_PER_AGENT \
  LEVEL_COMPLETION PARTIAL_LEVEL HEALTH_PER_POINT KILL SECONDS \
  EXIT_PROGRESS DAMAGE_DEALT DAMAGE_TAKEN ARMOR_DAMAGE_TAKEN \
  HEALTH_GAIN ARMOR_GAIN AMMO_SPENT DEATH
```

The 430 comma-separated weights are read from stdin. One CSV result line is
printed; the process exits with code 2 on argument/weight errors.

### Evaluation server (`--evaluate-server`)

```sh
doom/target/release/doom_headless --evaluate-server
```

Reads one job per line and writes one JSON result per line. A job is
`START_LEVEL,DIFFICULTY,DT,MAX_STEPS,MAX_LEVELS_PER_AGENT,` followed by the 13
fitness weights and the 430 policy weights. The result is
`{"score":…,"levels":…,"kills":…,"health":…,"seconds":…,"start_level":…}`,
plus a `profile` object when `DOOM_PROFILE=1`.

One `Game` is reused for the process lifetime; each job calls `agent_start`,
which resets all per-episode state, so results are identical to `--evaluate` for
the same inputs. This is the mode the trainer uses.

### Python environment (`training/env.py`)

- `Evaluator(executable, profile=False)` — persistent `--evaluate-server`
  client used by training.
  - `evaluate(weights, cfg, seed)` → dict with `fitness`, `levels`, `kills`,
    `health`, `seconds`, `start_level`, plus `wall_seconds` and an optional
    `profile`.
  - `warm_up(cfg)` runs a one-step episode at worker startup.
  - `close()` terminates the subprocess; a crashed server is restarted and the
    episode retried once.
- `evaluate(weights, executable, cfg, seed)` — compatibility wrapper that opens
  a temporary `Evaluator` for a single episode.
- `HeadlessDoomEnv(executable, dt=0.05, difficulty="medium", max_levels=10)` —
  context manager wrapping the interactive stdin/stdout protocol.
  - `reset(level)` → observation dict
  - `step(action)` → `(observation, done)`; `action` is
    `(turn, forward, strafe, fire, activate_exit, weapon, run)`. `done` becomes
    true on death or after `max_levels` level transitions.
  - `close()` / context-manager exit terminates the subprocess cleanly.
- `encode_observation(obs)` — produces the 42-value normalized input vector.
- `choose_action(weights, obs)` — runs the linear + `tanh` policy and returns
  the 7-element action tuple.

`env.py` is a clean seam for future distributed evaluation: an evaluation job
is fully described by an executable path, config, seed, and genome.

---

## Scripts

| Script | Purpose |
| --- | --- |
| `./exec_mgmt.sh -b` | Build the windowed game and `doom_headless --no-default-features` |
| `./exec_mgmt.sh -r` | Build if needed, then run `doom/target/release/doom` |
| `./exec_mgmt.sh -h` | Usage help |
| `./watch_agent.sh [policy]` | Build if needed, then run the GUI in watch mode (default policy: `training/output/best_policy.json`) |
| `./clear_training_data.sh` | Remove `training/output/` |
| `./ready.sh` | Build both binaries, then remove `training/output/` for a fresh run |
| `doom/run.sh` | Run `./target/release/doom` from the `doom/` directory |
| `doom/DOOM.app` (macOS) | GUI launcher for the release build; shows an alert if it is missing |

---

## Troubleshooting

- **`doom_headless` is missing** — run `./exec_mgmt.sh -b` or
  `python3 training/train.py --build`.
- **`cpu_threads must be -1, 0, or a positive number`** — fix the value in
  `config.json`.
- **`campaign_levels must be 50`** — the trainer intentionally validates the
  full campaign size.
- **Training appears to restart or ignore progress** — `"resume": true` reads
  `training/output/checkpoint.json`; delete `training/output/` or pass `--fresh`
  to start over.
- **`checkpoint configuration does not match` / `executable fingerprint`** —
  a behavioral config key, the seed, or the binary changed since the checkpoint
  was written. Restore it or use `--fresh`.
- **`cannot resume: metrics ...`** — the metrics file is missing, from another
  run, or does not line up with the checkpoint. Use `--fresh`.
- **Watch mode says it is waiting** — no `best_policy.json` exists yet, or the
  path passed to `watch_agent.sh` is wrong. The viewer reloads automatically
  once the file appears.
- **`expected 430 weights`** — the policy file or evaluator input does not match
  the fixed 42→10 architecture. Regenerate it with the trainer or rebuild the
  binary from the same source.
- **Changing a level has no effect** — `.lvl` files are embedded at compile
  time; rebuild with `./exec_mgmt.sh -b`.
- **Slow training** — reduce `population_size`, `evaluation_episodes`, or
  `max_steps`, or pin `cpu_threads` to a sensible worker count. Each worker
  thread keeps one simulation process alive; `DOOM_PROFILE=1` adds ~10%
  overhead, so use it only when measuring.

---

## Further reading

- [`training/README.md`](training/README.md) — trainer-focused notes and the
  environment interface.
- [`doom/README.md`](doom/README.md) — game controls, settings, and gameplay.
- [`doom/src/levels/README.md`](doom/src/levels/README.md) — complete level-file
  format.
