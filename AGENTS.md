# AGENTS.md — Doom-ML Project Guide

> **Purpose:** onboarding map for AI agents. Read this before exploring the codebase.
> It covers the whole system, data contracts, invariants, and known pitfalls.
> Update it when architecture or behavior changes.
>
> Source snapshot: 2026-11-12, commit `53aa5f8` (detached worktree of
> `RandomKerbal1234/Doom-ML`).

**Contents:** [1. What this is](#1-what-this-project-is) ·
[2. Repo map](#2-repository-map) · [3. Quick start](#3-quick-start) ·
[4. Data flow](#4-end-to-end-data-flow) · [5. Rust game](#5-rust-game) ·
[6. Levels](#6-level-system) · [7. Training](#7-training-system) ·
[8. Scripts](#8-shell-scripts) · [9. Gotchas](#9-invariants-and-gotchas) ·
[10. Change recipes](#10-common-change-recipes) ·
[11. Verification](#11-verification-checklist) · [12. Glossary](#12-glossary)

## 1. What this project is

Doom-ML is a self-contained machine-learning experiment:

- A small, asset-free **DOOM-like first-person shooter** written in Rust
  (320×200 software raycaster, `macroquad` window).
- A **Python evolutionary trainer** that learns a tiny linear policy to play
  the game. There is no neural-network library, no gradients, and no external
  ML dependency. The policy is a flat vector of **430 floats** evolved by
  mutation and selection.

Two Rust binaries share the exact same simulation code:

| Binary | Purpose |
| --- | --- |
| `doom` | GUI game (`macroquad`, window, keyboard/mouse input) |
| `doom_headless` | Windowless simulator: stdin/stdout JSON protocol plus a fast `--evaluate` path used by training |

The project is essentially a research sandbox. It is not a production game or
a general RL framework.

## 2. Repository map

```text
.
├── config.json                 # Training hyperparameters, reward weights, output paths
├── exec_mgmt.sh                # Build/run helper (cd doom; cargo build -r --bins / cargo run -r)
├── ready.sh                    # Build, then DELETE training/output (fresh run)
├── clear_training_data.sh      # rm -rf training/output/
├── watch_agent.sh              # Launch GUI watcher for training/output/best_policy.json
├── training/
│   ├── train.py                # Evolution loop, checkpointing, metrics, policy export
│   ├── env.py                  # Headless adapter, observation encoder, action decoder, evaluate()
│   └── README.md               # Human-facing training docs
└── doom/
    ├── Cargo.toml              # macroquad 0.4.16, Rust edition 2024
    ├── Cargo.lock
    ├── build.rs                # Embeds all src/levels/*.lvl at compile time
    ├── README.md               # Game controls and high-level ML instructions
    ├── run.sh                  # ./target/release/doom
    ├── DOOM.app/               # macOS launcher wrapper (shell script)
    └── src/
        ├── main.rs             # GUI entry point, window, frame upload, --watch-agent mode
        ├── game.rs             # Game state, simulation, input, agent API
        ├── render.rs           # Software renderer (walls, floor, sprites, HUD, effects)
        ├── ui.rs               # Pause/level-select overlay and settings overlay
        ├── settings.rs         # 57 tunable settings, difficulty presets, accessors
        ├── levels.rs           # Level parser/validator, tile helpers, LOS
        ├── levels/*.lvl        # 50 campaign level files
        └── bin/doom_headless.rs# Headless entry point + training evaluator
```

Generated/ignored:

- `doom/target/` — Cargo build output (ignored by `doom/.gitignore`).
- `training/output/` — checkpoints, policies, metrics (generated; **not**
  currently in `.gitignore`, so avoid committing it).
- `training/__pycache__/` — Python bytecode (not ignored).

## 3. Quick start

Run from the repository root unless noted:

```sh
./exec_mgmt.sh -b                 # cargo build -r --bins (builds both binaries)
python3 training/train.py         # train using config.json
./watch_agent.sh                  # watch best_policy.json in the GUI
```

Useful variants:

```sh
python3 training/train.py --build --generations 1   # smoke test, auto-build first
python3 training/train.py --config path/to/config.json
python3 training/train.py --generations 5           # override generation count
./exec_mgmt.sh -r                 # run the GUI game
open doom/DOOM.app                # macOS GUI launcher (requires prior build)
```

**Destructive:** `./ready.sh` builds and then deletes `training/output/`.
`clear_training_data.sh` deletes it unconditionally. Do not run either if a
checkpoint is valuable. `ready.sh` and `clear_training_data.sh` assume the
current directory is the repository root.

## 4. End-to-end data flow

### Training (primary path)

```text
config.json
   │
   ▼
training/train.py
   │  population of 430-float genomes
   │  for each generation:
   │    jobs = genome × evaluation_episode seed
   ▼
concurrent.futures.ProcessPoolExecutor (spawn, N workers)
   │
   ▼
training/env.py :: evaluate(genome, executable, cfg, seed)
   │  picks a random start level from the seed
   │  subprocess: doom_headless --evaluate <9 args>, weights on stdin
   ▼
doom/src/bin/doom_headless.rs :: evaluate_episode()
   │  Game::new() + agent_start(start_level, difficulty)
   │  loop up to max_steps:
   │      agent_policy_action(weights) → agent_step(...)
   │  score per completed level, health, kills, time
   ▼
stdout: score,levels,kills,health,seconds,start_level
   │
   ▼
train.py: average episodes → rank → elites + mutated offspring
   │
   ├── training/output/checkpoint.json   (population + RNG state)
   ├── training/output/best_policy.json  (best genome for the GUI)
   └── training/output/metrics.csv       (one row per generation)
```

### Watching

```text
watch_agent.sh
   │  ensures release binary exists, then:
   ▼
doom --watch-agent training/output/best_policy.json
   │  polls the policy file every 0.5 s
   │  loads weights + difficulty, picks a random level
   │  runs Game::agent_policy_action / agent_step in the GUI loop
   └── restarts on R, death, or 50 level transitions
```

### Interactive headless API (secondary path)

`training/env.py::HeadlessDoomEnv` keeps one `doom_headless` process alive and
exchanges newline-delimited commands/JSON. `train.py` does **not** use this
class; it uses the `--evaluate` fast path. The class is available for
experimentation and shows the intended interactive protocol.

## 5. Rust game

### 5.1 Build and level embedding

- `doom/build.rs` reads `src/levels/`, sorts `.lvl` files by filename, and
  generates `OUT_DIR/embedded_levels.rs` as
  `static EMBEDDED_LEVELS: &[(&str, &str)]`.
- `levels.rs` includes that generated file and parses levels at **runtime**
  during `initialize_levels()`. Malformed levels therefore fail at startup,
  not at compile time.
- Every `.lvl` change requires a rebuild. The game never reads the source
  files at runtime.
- `Cargo.toml`: `macroquad = "0.4.16"`, edition 2024, release profile with
  `opt-level = 3`, thin LTO, one codegen unit, symbols stripped.

### 5.2 GUI entry point (`main.rs`)

- Window: 960×600 default, resizable, `high_dpi = false`, nearest-neighbor
  texture filtering.
- Initializes levels, creates `Game::new()` (paused on level 0), and builds a
  `Texture2D` from `game.pixels` (RGBA, 320×200).
- Frame loop:
  1. Clamp frame `dt` to `[0, 0.05]`.
  2. If `--watch-agent`, poll/reload the policy and drive the agent;
     otherwise call `game.handle_input(dt)` + `game.update(dt)`.
  3. `game.render()` into the software framebuffer.
  4. Upload framebuffer to the texture; draw it letterboxed/scaled.
  5. Draw pause/end/settings overlays and, in watch mode, a status bar.
- `--watch-agent [path]` defaults to `training/output/best_policy.json`
  (relative to the current working directory; `watch_agent.sh` changes to the
  repo root first).
- The policy loader is a **naive string parser**, not a JSON library: it finds
  the first `"generation"`, `"doom_difficulty"`, and `"weights"` keys and
  parses their values. It only supports the exact file shape emitted by
  `train.py`, and requires exactly 430 weights.
- Reload behavior: if only weights/generation change, the current episode
  continues with new weights; if difficulty changes (or no policy was loaded
  before), a new random-level episode starts.
- Watch-mode status bar shows generation, levels completed out of 50, and
  `R: RESTART  Q: QUIT`.

### 5.3 Simulation model (`game.rs`)

World representation:

- Map is a 16×12 grid of tiles. Tile units are world units.
- Player position is continuous `(x, y)`; angle is radians, `0` points along
  `+x`; `sin/cos` used consistently. Map row `y` increases downward.
- Solid tiles: `#`, `B`, `M`; walkable: `.`, `X` (`X` is a visual marker only).
- Out-of-bounds tiles are treated as walls.

Player state (`Game`):

- `level_index`, `selected_level`, `x`, `y`, `angle`
- `health`, `armor`, `ammo: [i32; 4]` (pistol, shotgun, chaingun, rockets),
  `unlocked: [bool; 4]`, `weapon: usize` (0–3)
- `enemies`, `pickups`, `hit_effects`
- UI/flow: `paused`, `settings_open`, `dead`, `won`, `message`, `message_timer`,
  `time`
- Renderer: `pixels: Vec<u8>` (RGBA, 320×200×4), `z_buffer: [f32; 320]`

Movement and collision:

- `handle_input` (GUI) maps W/S/A/D/arrows/Shift to normalized turn/forward/
  strafe. Diagonal movement is normalized to length 1.
- `try_move` resolves axes independently and rejects positions where the
  player's collision circle (four corners at `player_radius`) overlaps a wall.
- `agent_step` implements the same movement for normalized agent actions.

Combat (`fire()`):

- All weapons are hitscan; there are no travelling projectiles.
- Target selection: nearest living enemy within weapon range whose angular
  offset is inside `spread + atan(0.36 / distance)` and with line of sight.
- Pistol/chaingun: full damage. Shotgun: damage falloff
  `max(1 - distance/15, 0.55)`. Rocket: when it hits an enemy, it damages
  **every** living enemy within `rocket_blast_radius` of that enemy's position
  (no LOS check, no self-damage); a rocket that hits no enemy only produces an
  impact effect.
- Each shot consumes ammo; if empty, shows `OUT OF AMMO`, sets a 0.22 s
  cooldown, and does nothing else.
- `find_shot_endpoint` uses DDA to find wall/miss impact points for visuals.
- Weapon stats come from `GameSettings`; medium defaults:

| Weapon | Damage | Range | Delay | Spread | Ammo/shot | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pistol | 30 | 9.5 | 0.32 s | 0.045 rad | 1 | starting weapon |
| Shotgun | 64 | 7.0 | 0.60 s | 0.15 rad | 1 | falloff to 55% |
| Chaingun | 18 | 10.5 | 0.105 s | 0.035 rad | 1 | fast fire |
| Rocket launcher | 95 | 10.5 | 0.82 s | 0.04 rad | 1 | 1.7-tile blast radius |

Enemies (`Enemy`):

- Two kinds: `imp` (`kind == 0`) and `demon` (`kind == 1`).
- Spawn with imp/demon health from settings; `attack_timer = 0.8`.
- If within `chase_range` and farther than `enemy_stop_distance`, move toward
  the player with axis-separated wall collision.
- If within `attack_range`, attack timer expired, and line of sight is clear,
  deal kind-specific damage. Incoming damage from all enemies is summed and
  applied once via `take_damage`.
- Armor absorbs `armor_absorption` fraction (rounded), capped by current
  armor; the remainder hits health. Death sets `health = 0`, `dead = true`.
- `danger`/`danger_angle` track the nearest enemy within attack range with
  line of sight; the HUD shows `DEMON IN RANGE` and a left/right edge arrow.
- Medium defaults: imp 55 HP / 0.72 speed / 7 damage / 1.25 s delay; demon
  80 HP / 0.94 speed / 10 damage / 1.55 s delay; shared chase 8.5, attack 6.5,
  stop 0.82.

Pickups (`PickupKind`):

| Kind | Effect |
| --- | --- |
| `shotgun` | unlock weapon 1, add `shotgun_pickup_ammo`, auto-switch |
| `chaingun` | unlock weapon 2, add `chaingun_pickup_ammo`, auto-switch |
| `rocket_launcher` | unlock weapon 3, add `rocket_pickup_ammo`, auto-switch |
| `medkit` | add `medkit_heal`, clamped to max health |
| `armor` | add `armor_pickup`, clamped to max armor |
| `ammo` | pistol ammo always; other ammo only for unlocked weapons |

Pickups are consumed when the player is within `pickup_radius` (0.52 default).
They are one-shot (removed from play, `active = false`).

Exit and progression:

- The exit is a coordinate, not a tile. Pressing E within 1.35 tiles:
  - GUI manual mode: advance to next level, or set `won = true` on the last
    level.
  - Headless/agent mode (`agent_step`): always advance with
    `(level_index + 1) % 50`, **wrapping forever and never setting `won`**.
- `start_new_run(first_level)` resets health/armor/ammo/unlocks/weapon,
  clears `dead`/`won`, unpauses, and loads the level.
- `restart()` resets to level 0.
- **`time` is not reset** by `start_new_run` or `restart`; it only starts at 0
  in `Game::new()`. This matters for the policy's `time/120` input during
  long-running watch sessions and repeated episodes.

### 5.4 Agent API in `game.rs`

Used by both the headless evaluator and the GUI watcher:

- `agent_start(first_level, preset)` — apply easy/medium/hard settings, start a
  fresh run at `first_level % 50`.
- `agent_step(turn, forward, strafe, fire, activate_exit, weapon, run, dt)` —
  clamp `dt` to 0.05, select weapon if unlocked, turn, move, optionally use the
  exit, optionally fire, then call `update(dt)`. It is a no-op when
  dead/won/paused/settings-open.
- `agent_observation() -> String` — one-line JSON observation (see §7.6).
- `agent_policy_action(weights) -> (turn, forward, strafe, fire,
  activate_exit, weapon, run)` — computes the 42-value input vector, applies
  the linear policy, and decodes 10 tanh outputs.

### 5.5 Renderer (`render.rs`)

Pure software rendering into `game.pixels`; no external art assets.

- Constants in `main.rs`/`doom_headless.rs`: `WIDTH=320`, `HEIGHT=200`,
  `VIEW_HEIGHT=170`, `HUD_TOP=172`.
- `render()` order: floor/ceiling → walls → sprites → hit effects → weapon →
  danger indicator → HUD.
- Floor/ceiling: ceiling gradient plus ray-cast floor with checker/grid
  texture and distance shading.
- Walls: DDA raycasting with camera plane `FOV_PLANE = 0.66`; procedural
  32×32 brick/marble patterns, fog, side-face darkening; writes `z_buffer`.
- Sprites: enemies, pickups, exit portal; depth-sorted, z-buffer clipped,
  16×16 procedural patterns, distance shading. Exit pulses green.
- Weapon: hand-drawn procedural rectangles per weapon; recoil lift; per-weapon
  muzzle flashes.
- Hit effects: per-weapon enemy/wall/miss particles; rocket blast is a
  hand-authored 16×16 two-frame sprite scaled to the impact distance.
- HUD: 3×5 bitmap font, HP/ARMOR/AMMO readouts, bars, level code, weapon name,
  transient messages, damage flash at the top of the screen.
- `ui.rs` overlays (pause, level select, settings, death/victory) are drawn by
  `macroquad` on top of the scaled framebuffer, not into `pixels`.

### 5.6 Settings and UI (`settings.rs`, `ui.rs`)

- `DifficultyPreset`: Easy, Medium, Hard, Custom.
- 57 settings in four categories:
  - **Player** (11): current/max health, current/max armor, walk/run/turn
    speeds, run turn speed, collision radius, armor absorption, starting ammo.
  - **Weapons** (21): damage, range, delay, spread, ammo cost for pistol,
    shotgun, chaingun, rocket launcher, plus rocket blast radius.
  - **Demons** (11): imp/demon health, speed, damage, attack delay; shared
    chase range, attack range, stop distance.
  - **Supplies** (14): current ammo for each weapon, medkit/armor amounts,
    pickup radius, weapon-pickup ammo, ammo-cache amounts.
- `GameSettings::medium()` defines defaults; `preset()` applies Easy/Hard
  overrides. Presets do not change weapon stats.
- Manual edits set `difficulty = Custom` and take effect immediately. Settings
  are **not persisted** across runs.
- UI: Up/Down selects a row, Left/Right edits, Tab/PageUp/PageDown changes
  section, 1/2/3 applies a preset. Eight rows visible per page. Shift+P opens
  settings; Esc closes.
- Pause/help screen lists controls and all 50 levels; Up/Down selects a level,
  Enter starts it, Q quickplays from level 1, Esc/Space/F1 resumes, F10 quits.
- Death/victory overlay: Enter or R restarts at level 0.

### 5.7 GUI controls (manual play)

`W/S` move, `A/D` strafe, Left/Right turn, Shift run, Ctrl/Space/left mouse
fire, `1–4`/mouse wheel weapon, `E` exit, F1/Esc pause, Shift+P settings,
F10 quit from pause, R restart after death/win.

## 6. Level system

### 6.1 File format

Each `.lvl` file is INI-like:

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
11.5,3.5,imp
10.5,8.5,demon

[pickups]
8.5,6.5,shotgun
5.5,6.5,ammo
12.5,8.5,medkit
```

Rules enforced by `levels.rs`:

- Top-level `key=value` fields: `code`, `title`, `spawn=x,y,angle`,
  `exit=x,y`; all required.
- `[map]`: exactly **12 rows × 16 characters**. Valid tiles: `#`, `B`, `M`
  (solid), `.`, `X` (walkable; `X` is cosmetic).
- `[enemies]`: `x,y,type` with type `imp` or `demon`.
- `[pickups]`: `x,y,type` with type `shotgun`, `chaingun`,
  `rocket_launcher`, `medkit`, `armor`, or `ammo`.
- Comments start with `;` or `//`; blank lines ignored.
- Spawn, exit, enemies, and pickups must lie on open tiles.
- Coordinates must be finite; exactly two/three comma-separated values as
  appropriate.
- Level codes must be unique across all files.
- Files are played in lexicographic filename order; numeric prefixes (`01_`,
  `02_`, …) define campaign order.

### 6.2 Campaign contents

- 50 files, `01_e1m1.lvl` … `50_e5m10.lvl`, codes `E1M1` … `E5M10`.
- 50 unique codes and 50 unique map layouts.
- Episode themes/titles repeat across E2–E5 (e.g. `CINDER MAZE` appears as
  E2M3, E3M2, E4M1, E5M10) but the maps and placements differ.
- `train.py` hard-requires `campaign_levels == 50`; adding/removing levels
  requires updating that check (and `config.json`).

## 7. Training system

### 7.1 Policy representation

- Architecture: **42 inputs → 10 outputs**, fully connected, linear + bias,
  `tanh` activation. No hidden layer.
- Weights are output-major: for each output, 42 input weights followed by one
  bias. Total `10 * (42 + 1) = 430`.
- The same 430-float vector is used by Python and Rust. The parameter count is
  hard-coded in several places (see §9).

Output → action mapping (indices 0–9):

| Output | Meaning |
| --- | --- |
| 0 | turn, clamped to `[-1, 1]` |
| 1 | forward, clamped to `[-1, 1]` |
| 2 | strafe, clamped to `[-1, 1]` |
| 3 | fire if `> 0` |
| 4 | activate exit if `> 0` |
| 5–8 | weapon preference scores; choose the highest among unlocked weapons |
| 9 | run if `> 0` |

### 7.2 Input vector (42 values)

Computed by `training/env.py::encode_observation` and independently by
`Game::agent_policy_action` in `game.rs`. **These two implementations must stay
in sync.**

| Index range | Count | Values |
| --- | ---: | --- |
| 0–7 | 8 | Ray distances at `angle + i*2π/8`; step 0.1 up to 8.0 tiles; stop on wall/out-of-bounds; `min(distance/8, 1)` |
| 8–17 | 10 | Two nearest living enemies: `sin(bearing)`, `cos(bearing)`, `min(dist/12,1)`, `min(hp/200,1)`, `kind` (0/1); missing enemy = `0,0,1,0,0` |
| 18–26 | 9 | Nearest active pickup: `sin(bearing)`, `cos(bearing)`, `min(dist/12,1)`, one-hot `[shotgun, chaingun, rocket_launcher, medkit, armor, ammo]`; missing pickup = `0,0,1,0,0,0,0,0,0` |
| 27–29 | 3 | Exit: `sin(bearing)`, `cos(bearing)`, `min(dist/16,1)` |
| 30–31 | 2 | `health/200`, `armor/200` |
| 32–35 | 4 | `clamp(ammo,0,100)/100` for pistol, shotgun, chaingun, rockets |
| 36–39 | 4 | Unlocked flags as 1/0 |
| 40–41 | 2 | `weapon/3`, `min(time/120,1)` |

Rays use the current player angle; bearings are relative to the player angle.
There is no explicit map input beyond the 8 rays and entity/pickup/exit
bearings.

### 7.3 Evolution loop (`training/train.py`)

- Population initialized from `N(0, initial_weight_sigma)` per weight.
- Per generation:
  1. Build `population_size × evaluation_episodes` jobs, each with seed
     `cfg.seed + generation*evaluation_episodes + i`.
  2. Evaluate all jobs in parallel via `ProcessPoolExecutor` (spawn context).
  3. Average each genome's episode results (`fitness`, `levels`, `kills`,
     `health`, `seconds`; `start_level` taken from the first episode).
  4. Rank genomes by mean fitness descending.
  5. Copy the top `elite_count` genomes unchanged (clamped to at least 1 and at
     most population size).
  6. Fill the rest by choosing a parent uniformly from the top
     `min(parent_pool_size, population_size)` and mutating each gene
     independently with probability `mutation_rate`, adding
     `N(0, mutation_sigma)`.
  7. Append a metrics row, overwrite the checkpoint, write the best policy.
- No crossover, no gradient updates, no backpropagation.
- `generations` is a total count; generation numbering is 0-based internally
  and printed 1-based. `--generations N` overrides it.
- On resume, `start_generation = saved.generation + 1` and the Python RNG state
  is restored from the checkpoint. If the checkpoint is already at/above the
  configured generation count, the loop does nothing.
- If `resume` is false or no checkpoint exists, the population is reinitialized
  and metrics start fresh (`metrics.csv` overwritten).

### 7.4 Evaluation and fitness

Each episode (`doom_headless --evaluate`):

- Starts at a uniformly random level chosen by `random.Random(seed).randrange(50)`.
  All genomes in the same generation/episode share the same start level; each
  evaluation episode uses a different seed.
- Runs up to `max_steps` at `time_step_seconds` per step, or until death, or
  until **10 level completions**. (Despite README wording about 50 exits, the
  cap is 10 in both Python and Rust.)
- Level transitions wrap from level 50 back to level 1.

Scoring in `evaluate_episode`:

- On each completed level:
  `+ level_completion`
  `+ health_at_transition * health_per_point`
  `+ kills_on_that_level * kill`
  `- time_spent_on_that_level * seconds`
- If the episode ends before 10 completions:
  `+ completed_levels * partial_level`
  `+ current_health * health_per_point`
  `+ current_level_kills * kill`
  `- current_level_time * seconds`
- Returned CSV: `score,level_count,total_kills,health,total_time,start_level`.
- There is **no explicit death penalty**; death just ends the episode. The
  `death` config key is currently unused.
- The reward uses only five fitness weights:
  `level_completion`, `partial_level`, `health_per_point`, `kill`, `seconds`.
  The other keys in `config.json` (`exit_progress`, `damage_dealt`,
  `damage_taken`, `armor_damage_taken`, `health_gain`, `armor_gain`,
  `ammo_spent`, `death`) are **not read by any code**. Extending them requires
  changing both `training/env.py` and `doom_headless.rs`.

### 7.5 Configuration (`config.json`)

| Key | Default | Meaning |
| --- | ---: | --- |
| `cpu_threads` | 0 | 0 = all available CPUs; -1 = all but one; positive = exact worker count |
| `generations` | 100 | Total generations to run |
| `population_size` | 48 | Genomes per generation |
| `elite_count` | 4 | Top genomes copied unchanged |
| `parent_pool_size` | 12 | Top genomes eligible as parents |
| `initial_weight_sigma` | 0.35 | Std-dev for initial weights |
| `mutation_rate` | 0.08 | Per-gene mutation probability |
| `mutation_sigma` | 0.18 | Std-dev of mutation noise |
| `time_step_seconds` | 0.05 | Simulation step; must be in `(0, 0.05]` |
| `max_steps` | 15000 | Per-episode step cap |
| `evaluation_episodes` | 2 | Episodes averaged per genome per generation |
| `doom_difficulty` | `medium` | `easy`, `medium`, or `hard` |
| `campaign_levels` | 50 | Must equal 50 (hard-coded check) |
| `seed` | 2001901 | Base RNG seed |
| `resume` | true | Resume from checkpoint if present |
| `output_directory` | `training/output` | Resolved relative to repo root |
| `checkpoint_file` | `checkpoint.json` | Population/RNG checkpoint |
| `best_policy_file` | `best_policy.json` | Best genome for GUI watcher |
| `metrics_file` | `metrics.csv` | Per-generation CSV |

Validation in `train.py`:

- Difficulty must be easy/medium/hard.
- `campaign_levels` must be 50.
- `evaluation_episodes`, `population_size`, `generations` must be positive.
- `time_step_seconds` in `(0, 0.05]`; `max_steps` positive.
- `doom/target/release/doom_headless` must exist unless `--build` is passed.

### 7.6 Outputs

`training/output/checkpoint.json`:

```json
{
  "generation": 0,
  "population": [[/* 430 floats */]],
  "best_result": {"fitness": 0.0, "levels": 0.0, "kills": 0.0,
                   "health": 0.0, "seconds": 0.0, "start_level": 0},
  "best_genome": [/* 430 floats */],
  "random_state": [/* Python random state: version, 625 ints, gauss_next */]
}
```

`training/output/best_policy.json`:

```json
{
  "architecture": {"inputs": 42, "outputs": 10, "activation": "tanh-linear"},
  "generation": 0,
  "doom_difficulty": "medium",
  "result": {"fitness": 0.0, "levels": 0.0, "kills": 0.0,
              "health": 0.0, "seconds": 0.0, "start_level": 0},
  "weights": [/* exactly 430 floats */]
}
```

- `generation` is 0-based; the GUI displays `generation + 1`.
- The GUI's naive parser depends on these key names and the weights array
  shape.

`training/output/metrics.csv` columns:

```text
generation,best_fitness,mean_fitness,levels,kills,health,seconds,start_level
```

- One row per generation; `best_*` values are the best genome's episode
  averages, not a single episode.
- Appended when resuming; overwritten when starting fresh.

### 7.7 Headless protocol (`doom_headless.rs`)

**Interactive mode** (stdin lines, stdout one JSON line per command):

```text
reset,<level_index>,<easy|medium|hard>
step,<turn>,<forward>,<strafe>,<fire 0|1>,<exit 0|1>,<weapon>,<run 0|1>,<dt>
quit
```

- Invalid commands return `{"error":"invalid command"}`.
- `reset` calls `agent_start` and returns an observation.
- `step` calls `agent_step` and returns an observation.
- Observation schema (one line):

```json
{
  "level": 0,
  "x": 2.5, "y": 2.5, "angle": 0.0,
  "health": 100, "armor": 0,
  "ammo": [50, 0, 0, 0],
  "unlocked": [true, false, false, false],
  "weapon": 0,
  "time": 0.0,
  "dead": false,
  "exit": [14.5, 10.5],
  "map": ["################", "..."],
  "enemies": [{"x": 5.5, "y": 3.5, "hp": 55.0, "kind": 0}],
  "pickups": [{"x": 8.5, "y": 6.5, "kind": "shotgun"}]
}
```

- `map` is 12 strings of 16 characters; only living enemies and active
  pickups are included. There is no `won` field.

**Evaluation mode:**

```sh
doom_headless --evaluate \
  <start_level> <difficulty> <dt> <max_steps> \
  <level_completion> <partial_level> <health_per_point> <kill> <seconds>
```

- Policy weights are read from stdin as comma-separated floats; exactly 430
  required.
- stdout: `score,level_count,total_kills,health,total_time,start_level`
  (`total_time` formatted to 2 decimals).
- Exit codes: 1 = level initialization failure, 2 = evaluation failure.
- `doom_headless` compiles `game.rs`, `levels.rs`, `render.rs`, and
  `settings.rs` via `#[path]` includes. It does not include `ui.rs` or
  `main.rs`, never opens a window, and never calls `render()`.

### 7.8 Python files

`training/env.py`:

- `HeadlessDoomEnv` — persistent process adapter with `reset(level)`,
  `step(action)`, `close()`, context-manager support. `step` reports `done`
  when dead or after 10 level completions.
- `encode_observation(obs)` — Python version of the 42-value input vector.
- `choose_action(weights, obs)` — Python version of the policy forward pass
  and action decoding.
- `evaluate(weights, executable, cfg, seed)` — the function actually used by
  training; launches `doom_headless --evaluate`, passes weights on stdin, and
  parses the CSV result.
- The interactive class is not used by `train.py`; it is useful for debugging
  or writing new trainers.

`training/train.py`:

- `PARAMETERS = 10 * (42 + 1) = 430`.
- `CAMPAIGN_LEVELS = 50`.
- `tester_count` resolves `cpu_threads` using `os.process_cpu_count()` when
  available, then `sched_getaffinity`, then `os.cpu_count()`.
- `mutate`, `run_job`, `summarize_episodes`, `list_state`, `tuple_state`.
- `main()` handles CLI/config validation, resume, the generation loop,
  checkpointing, and policy export.

Python 3.9+ is sufficient (the local system Python is 3.9.6).

## 8. Shell scripts

| Script | Behavior |
| --- | --- |
| `exec_mgmt.sh` | `cd doom`; `-b` = `cargo build -r --bins`; `-r` = run `target/release/doom` (or `cargo run -r`); `-h`/no args = help. Must be run from repo root. |
| `ready.sh` | `./exec_mgmt.sh -b` then `./clear_training_data.sh`; fresh training setup. Destructive. No shebang. |
| `clear_training_data.sh` | `rm -rf training/output/`. Destructive. |
| `watch_agent.sh` | Resolves repo root from its own path, builds if `doom/target/release/doom` is missing, then `exec doom --watch-agent "${1:-training/output/best_policy.json}"`. |
| `doom/run.sh` | `./target/release/doom` (run from `doom/`). |
| `doom/DOOM.app/.../doom-launcher` | macOS wrapper: finds `doom/target/release/doom`, shows an alert if missing, runs it with output suppressed. |

## 9. Invariants and gotchas

Read these before changing anything.

1. **430 is hard-coded in multiple places.**
   - `training/train.py`: `PARAMETERS = 10 * (42 + 1)`.
   - `doom/src/main.rs`: `if weights.len() != 430`.
   - `doom/src/bin/doom_headless.rs`: `if weights.len() != 430`.
   - `best_policy.json` architecture metadata in `train.py`.
   Changing the policy size requires updating all of them plus the input
   encoder/decoder.

2. **The 42-input encoder is duplicated.**
   - Python: `training/env.py::encode_observation`.
   - Rust: `game.rs::agent_policy_action`.
   Any change to ray count, entity features, ordering, or normalization must
   be mirrored exactly or Python-side and Rust-side policies will diverge.

3. **The 10-level episode cap exists in code but contradicts docs.**
   - `training/env.py`: `self.levels_completed >= 10`.
   - `doom_headless.rs`: `level_count >= 10`.
   - `training/README.md` says 50 exits.
   If you want full-campaign episodes, change both caps (and consider
   `max_steps`).

4. **Most fitness config keys are dead.** Only five reward weights are passed
   to the headless evaluator. The extra keys are silently ignored.

5. **`campaign_levels` must be 50.** `train.py` exits otherwise. Adding a
   51st level requires changing `CAMPAIGN_LEVELS` and this validation.

6. **`time` never resets between runs.** It is initialized only in
   `Game::new()`. In long-lived processes (GUI watcher, interactive env) the
   policy's `time/120` input saturates after two minutes. Add a reset in
   `start_new_run`/`restart` if that matters.

7. **Headless agents wrap the campaign forever; GUI manual play wins.**
   `agent_step` uses `(level_index + 1) % 50` and never sets `won`. The GUI
   watcher instead restarts after 50 level transitions or death. Do not rely on
   `won` in headless mode.

8. **Level changes require a rebuild.** Levels are embedded by `build.rs`;
   the runtime never reads `.lvl` files.

9. **`training/output/` is not gitignored.** The root `.gitignore` only ignores
   `/doom/target`. Be careful not to commit checkpoints/policies/metrics.
   `ready.sh`/`clear_training_data.sh` delete the directory.

10. **The GUI policy parser is not real JSON parsing.** It searches for the
    first `"generation"`, `"doom_difficulty"`, and `"weights"` strings. Keep
    `best_policy.json` shaped as emitted by `train.py`, or update
    `main.rs::load_agent_policy`. A missing or invalid policy is silently
    ignored, and the watcher shows its "waiting" status.

11. **The GUI watcher reloads weights mid-episode.** If only weights/generation
    change, the current episode continues with the new policy; difficulty
    changes force a restart at a random level.

12. **Settings are not persisted.** Every launch starts at Medium. The GUI
    watcher uses the difficulty stored in `best_policy.json`, not the GUI
    preset.

13. **`render.rs` is compiled into the headless binary** even though it is
    never called. Removing it from the `#[path]` module list breaks the
    headless build.

14. **No automated tests, CI, formatter, or linter config.** Verification is
    manual: `cargo check`/`cargo build`, Python compile, a 1-generation smoke
    run, and launching the GUI.

15. **The Python interactive env assumes `reset()` before `step()`.** `step`
    dereferences the previous observation to detect level changes.

16. **All evaluation is local.** The evaluation job is self-contained
    (genome, executable path, config, seed), which is a clean seam for future
    distributed evaluation, but no distribution code exists.

17. **The repo may be checked out as a detached worktree.** Do not assume a
    named branch; inspect `git status`/`git branch` before branch-based work.

18. **macOS/Windows notes.** Windows uses a GUI subsystem attribute to avoid a
    console window. macOS ships a `DOOM.app` wrapper. Shell scripts are LF via
    `.gitattributes`.

## 10. Common change recipes

- **Add a level:** create `doom/src/levels/NN_xxx.lvl` with a unique code and
  exactly 12×16 map; keep spawn/exit/enemies/pickups on open tiles; rebuild.
  If the total leaves 50, update `CAMPAIGN_LEVELS`/`campaign_levels` and the
  `train.py` check.
- **Tune training:** edit `config.json`; no code changes needed for the
  supported keys. Use `--generations` for quick overrides.
- **Change reward shaping:** edit the five active weights in `config.json`.
  For new terms, extend `evaluate()` in `env.py`, the argument list in
  `doom_headless.rs::evaluate_episode`, and the scoring loop.
- **Change policy size/architecture:** update `train.py` constants and
  architecture metadata, `env.py` encoder/decoder, `game.rs`
  `agent_policy_action`, and both 430-weight checks. Update the observation
  docs here.
- **Change game behavior:** edit `game.rs`; both GUI and training change
  together. Rebuild both binaries. Update `AGENTS.md` if contracts change.
- **Change controls/UI:** `game.rs::handle_input`/`handle_settings_input`,
  `ui.rs`, and the HUD in `render.rs`.
- **Change difficulty:** `settings.rs` presets/defaults. Weapon stats are in
  `GameSettings::medium()`.
- **Watch a different policy:** `./watch_agent.sh path/to/policy.json`.

## 11. Verification checklist

```sh
# Rust compile check
cd doom && cargo check --bins

# Release build (what scripts/training use)
./exec_mgmt.sh -b

# Python syntax
python3 -m py_compile training/env.py training/train.py

# Smoke training (builds if needed, writes training/output)
python3 training/train.py --build --generations 1

# GUI
./exec_mgmt.sh -r
# or: open doom/DOOM.app

# Watch a trained policy
./watch_agent.sh
```

A successful smoke run should print a generation line, write
`training/output/checkpoint.json`, `best_policy.json`, and `metrics.csv`, and
leave a policy with 430 weights.

## 12. Glossary

- **Genome / policy / weights:** the flat 430-float vector.
- **Generation:** one full evaluate → rank → select → mutate cycle.
- **Episode:** one headless run from a random start level until death, 10
  level completions, or `max_steps`.
- **Elite:** a top genome copied unchanged into the next generation.
- **Tick / step:** one simulation update of `time_step_seconds` (max 0.05 s).
- **Headless:** the `doom_headless` binary; same simulation as the GUI, no
  window, stdin/stdout protocol.
- **Watch mode:** GUI mode that loads a policy and plays automatically.
