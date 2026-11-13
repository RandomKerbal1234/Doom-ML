# AGENTS.md — Doom-ML Project Guide

> **Purpose:** onboarding map for AI agents. Read this before exploring the codebase.
> It covers the whole system, data contracts, invariants, and known pitfalls.
> Update it when architecture or behavior changes.
>
> Source snapshot: 2026-11-12, commit `e2affad` (`Analysis` branch of
> `RandomKerbal1234/Doom-ML`). The working tree may contain local changes;
> check `git status` before assuming a clean checkout.

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
- A 50-level campaign embedded in the binaries from editable `.lvl` files.
- A 13-term reward function covering level completion, exit progress, combat,
  survival, resource use, and death. See §7.4.

Two Rust binaries share the exact same simulation code:

| Binary | Purpose |
| --- | --- |
| `doom` | GUI game (`macroquad`, window, keyboard/mouse input, `--watch-agent` mode) |
| `doom_headless` | Windowless simulator: stdin/stdout JSON protocol plus a fast `--evaluate` path used by training |

The project is essentially a research sandbox. It is not a production game or
a general RL framework.

## 2. Repository map

```text
.
├── config.json                 # Training hyperparameters, 13 reward weights, output paths
├── exec_mgmt.sh                # Build/run helper (cd doom; cargo build -r --bins / cargo run -r)
├── ready.sh                    # Build, then DELETE training/output (fresh run). Not executable (mode 644)
├── clear_training_data.sh      # rm -rf training/output/ (destructive)
├── watch_agent.sh              # Launch GUI watcher for training/output/best_policy.json
├── stats/                      # Tracked run-analysis artifacts (PDF + spreadsheet), not code
│   ├── Doom-ML_run_analysis.pdf
│   └── compiled_mean_fitness.xlsx
├── training/
│   ├── train.py                # Evolution loop, validation, run manifest, checkpointing, policy export
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
        ├── levels/             # 50 campaign .lvl files + README.md
        └── bin/doom_headless.rs# Headless entry point + training evaluator
```

Generated/ignored:

- `doom/target/` — Cargo build output (ignored by `doom/.gitignore` / root
  `.gitignore` entry `/doom/target`).
- `training/output/` — checkpoints, policies, metrics, manifests (generated;
  **not** ignored by `.gitignore`, so avoid committing it).
- `training/__pycache__/` — Python bytecode. In this snapshot the `.pyc` files
  are tracked in Git; do not delete or regenerate them casually.
- `.DS_Store` — macOS metadata; may appear staged in a developer checkout.

`.gitattributes` forces `*.sh` to LF. There is no formatter or linter config.

## 3. Quick start

Run from the repository root unless noted:

```sh
./exec_mgmt.sh -b                 # cargo build -r --bins (builds both binaries)
python3 training/train.py         # train using config.json (resume=false -> fresh run)
./watch_agent.sh                  # watch best_policy.json in the GUI
```

Useful variants:

```sh
python3 training/train.py --build --generations 1 --fresh   # smoke test, auto-build
python3 training/train.py --config path/to/config.json
python3 training/train.py --generations 5                   # override total generation count
python3 training/train.py --fresh                            # ignore an existing checkpoint
./exec_mgmt.sh -r                                            # run the GUI game
open doom/DOOM.app                                           # macOS GUI launcher (requires prior build)
```

Training artifacts are written to `training/output/` (paths configurable). To
resume, set `"resume": true` in the config and rerun; the trainer verifies the
checkpoint, manifest, metrics, seed, behavioral configuration, and executable
hash before continuing (§7.5–§7.6).

**Destructive:** `./ready.sh` (currently mode 644 and without a shebang; run
`bash ready.sh`) builds and then deletes `training/output/`.
`clear_training_data.sh` deletes it unconditionally. Do not run either if a
checkpoint is valuable. Both assume the current directory is the repository
root. `--fresh` does **not** delete the output directory; it only ignores the
checkpoint and starts a new run (metrics are truncated at the first generation).

## 4. End-to-end data flow

### Training (primary path)

```text
config.json
   │
   ▼
training/train.py
   │  validate config, resolve executable, hash config + executable
   │  population of 430-float genomes (or checkpoint population)
   │  for each generation:
   │    seeds = seed + generation * evaluation_episodes + i
   │    jobs  = genome × seed
   ▼
concurrent.futures.ProcessPoolExecutor (spawn, N workers)
   │
   ▼
training/env.py :: evaluate(genome, executable, cfg, seed)
   │  random.Random(seed).randrange(campaign_levels) -> start_level
   │  subprocess: doom_headless --evaluate
   │      start_level difficulty dt max_steps max_levels_per_agent
   │      + 13 fitness weights
   │  weights on stdin, comma-separated (exactly 430)
   ▼
doom/src/bin/doom_headless.rs :: evaluate_episode()
   │  Game::new() + agent_start(start_level, difficulty)
   │  loop up to max_steps:
   │      agent_policy_action(weights) -> agent_step(...)
   │  score per completed level, exit progress, combat, health, time, death
   ▼
stdout: score,levels,kills,health,seconds,start_level
   │
   ▼
train.py: average episodes -> rank -> elites + mutated offspring
   │
   ├── training/output/checkpoint.json   (population, RNG state, provenance)
   ├── training/output/best_policy.json  (best genome for the GUI)
   ├── training/output/metrics.csv       (one row per generation, first column run_id)
   └── training/output/run_manifest.json (provenance + status, updated each generation)
```

Each genome in a generation is evaluated on the same seed-derived start levels,
which reduces noise when comparing genomes. The trainer also verifies that the
Rust evaluator returned the start level predicted by the Python RNG; a mismatch
aborts the run. The `run_id` (`seed-configHashPrefix-timestamp`) is written to
every metrics row and to the checkpoint, policy, and manifest.

### Watching

```text
watch_agent.sh
   │  ensures release binary exists, then:
   ▼
doom --watch-agent training/output/best_policy.json
   │  polls the policy file every 0.5 s
   │  loads weights + difficulty (naive string parser, not a JSON library)
   │  picks a random level and runs Game::agent_policy_action / agent_step
   └── restarts on R, death, difficulty change, or 50 level transitions
```

### Interactive headless API (secondary path)

`training/env.py::HeadlessDoomEnv` keeps one `doom_headless` process alive and
exchanges newline-delimited commands/JSON. `train.py` does **not** use this
class; it uses the `--evaluate` fast path. The class is useful for debugging or
writing new trainers. Its `max_levels` cap defaults to 10 and stops an episode
after that many level transitions or on death.

## 5. Rust game

### 5.1 Build and level embedding

- `doom/build.rs` reads `src/levels/`, sorts `.lvl` files by filename, and
  generates `OUT_DIR/embedded_levels.rs` as
  `static EMBEDDED_LEVELS: &[(&str, &str)]`.
- `levels.rs` includes that generated file and parses levels at **runtime**
  during `initialize_levels()`. Malformed levels therefore fail at startup,
  not at compile time. `initialize_levels()` uses a `OnceLock` and errors if
  called twice.
- Every `.lvl` change requires a rebuild. The game never reads the source
  files at runtime. `build.rs` re-runs when the levels directory or any level
  file changes.
- `Cargo.toml`: `macroquad = "0.4.16"`, edition 2024, release profile with
  `opt-level = 3`, thin LTO, one codegen unit, symbols stripped.

### 5.2 GUI entry point (`main.rs`)

- Window: 960×600 default, resizable, `high_dpi = false`, nearest-neighbor
  texture filtering, `sample_count = 1`.
- Initializes levels, creates `Game::new()` (paused on level 0, Medium
  settings), and builds a `Texture2D` from `game.pixels` (RGBA, 320×200).
- Frame loop:
  1. Clamp frame `dt` to `[0, 0.05]`.
  2. If `--watch-agent`, poll/reload the policy and drive the agent;
     otherwise call `game.handle_input(dt)` + `game.update(dt)`.
  3. `game.render()` into the software framebuffer.
  4. Upload framebuffer to the texture; draw it letterboxed/scaled.
  5. Draw settings/pause/end overlays and, in watch mode, a status bar.
- `--watch-agent [path]` defaults to `training/output/best_policy.json`
  (relative to the current working directory; `watch_agent.sh` changes to the
  repo root first).
- The policy loader is a **naive string parser**, not a JSON library: it finds
  the first `"generation"`, `"doom_difficulty"`, and `"weights"` keys and
  parses their values. It only supports the exact file shape emitted by
  `train.py`, and requires exactly 430 weights. It ignores all other keys, so
  the extra provenance fields in `best_policy.json` are harmless.
- Reload behavior: the file is polled every 0.5 s. If only weights/generation
  change, the current episode continues with new weights. If the difficulty
  changes (or no policy was loaded before), a new random-level episode starts.
  Missing or invalid policy files are silently ignored; if no valid policy has
  ever loaded, the watcher shows a
  "WAITING FOR training/output/best_policy.json" status.
- Watch episode lifecycle: `R` restarts at a random level; death or 50 level
  transitions also restart; `Q` quits. The status bar shows generation
  (`generation + 1`), completed level transitions out of 50, and
  `R: RESTART  Q: QUIT`.
- Normal GUI play (`no --watch-agent`) uses manual input and the pause/level
  selection and settings overlays.

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
- UI/flow: `paused`, `settings_open`, `settings_return_paused`,
  `settings_category`, `settings_selection`, `difficulty`, `settings`, `dead`,
  `won`, `message`, `message_timer`
- Combat/visual timers: `shot_timer`, `recoil`, `muzzle_timer`,
  `muzzle_weapon`, `damage_flash`, `danger`, `danger_angle`
- Agent statistics (reset per episode, read by the headless evaluator):
  `damage_dealt: f32`, `damage_taken`, `armor_damage_taken`, `health_gain`,
  `armor_gain`, `ammo_spent` (all `i32` except `damage_dealt`)
- Renderer: `pixels: Vec<u8>` (RGBA, 320×200×4), `z_buffer: [f32; 320]`
- `time` accumulates simulation time; it is initialized only in `Game::new()`
  and is **not** reset by `start_new_run`, `restart`, or `agent_start`.

Movement and collision:

- `handle_input` (GUI) maps W/S/A/D/arrows/Shift to normalized turn/forward/
  strafe. Diagonal movement is normalized to length 1.
- `try_move` resolves axes independently and rejects positions where the
  player's collision circle (four corners at `player_radius`) overlaps a wall.
- `agent_step` implements the same movement for normalized agent actions.
- Medium movement defaults: walk 2.7 tiles/s, run 4.3, turn 2.1 rad/s, run turn
  2.8, collision radius 0.19.

Combat (`fire()`):

- All weapons are hitscan; there are no travelling projectiles.
- Target selection: nearest living enemy within weapon range whose angular
  offset is inside `spread + atan(0.36 / distance)` and with line of sight.
  The extra angular allowance models the enemy's visible width.
- Pistol/chaingun: full damage. Shotgun: damage falloff
  `max(1 - distance/15, 0.55)`. Rocket: when it hits an enemy, it damages
  **every** living enemy within `rocket_blast_radius` of that enemy's position
  (no LOS check, no self-damage); a rocket that hits no enemy only produces an
  impact effect.
- `damage_dealt` accumulates damage actually applied, capped at each enemy's
  remaining HP (overkill is not counted). Rocket damage is summed across all
  enemies in the blast.
- Each shot consumes ammo; if empty, shows `OUT OF AMMO`, sets a 0.22 s
  cooldown, and does nothing else. `ammo_spent` counts only ammo actually
  consumed by fired shots.
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
  armor; the remainder hits health. `damage_taken` records health actually
  lost; `armor_damage_taken` records armor lost separately. Death sets
  `health = 0`, `dead = true`.
- `danger`/`danger_angle` track the nearest enemy within attack range with
  line of sight; the HUD shows `DEMON IN RANGE` and a left/right edge arrow.
- Medium defaults: imp 55 HP / 0.72 speed / 7 damage / 1.25 s delay; demon
  80 HP / 0.94 speed / 10 damage / 1.55 s delay; shared chase 8.5, attack 6.5,
  stop 0.82.

Pickups (`PickupKind`):

| Kind | Effect (medium defaults) |
| --- | --- |
| `shotgun` | unlock weapon 1, add 12 shells, auto-switch |
| `chaingun` | unlock weapon 2, add 70 rounds, auto-switch |
| `rocket_launcher` | unlock weapon 3, add 6 rockets, auto-switch |
| `medkit` | add 30 health, clamped to max health; `health_gain` counts only the actual gain |
| `armor` | add 40 armor, clamped to max armor; `armor_gain` counts only the actual gain |
| `ammo` | pistol +25 always; shotgun +6, chaingun +30, rockets +2 only for unlocked weapons |

Pickups are consumed when the player is within `pickup_radius` (0.52 default).
They are one-shot (removed from play, `active = false`).

Exit and progression:

- The exit is a coordinate, not a tile. Pressing E within 1.35 tiles:
  - GUI manual mode: advance to the next level, or set `won = true` on the last
    level.
  - Headless/agent mode (`agent_step`): always advance with
    `(level_index + 1) % 50`, **wrapping forever and never setting `won`**.
- `start_new_run(first_level)` resets health/armor/ammo/unlocks/weapon, clears
  `dead`/`won`, unpauses, resets the agent statistics, and loads the level.
- `restart()` resets to level 0 with the same stat reset.
- **`time` is not reset** by `start_new_run` or `restart`; it only starts at 0
  in `Game::new()`. This matters for the policy's `time/120` input during
  long-running watch sessions and reused interactive environments.
- The agent statistics are per episode: `agent_start`/`start_new_run` resets
  them, but `load_level` (level transition) does not. They therefore cover the
  whole episode across levels.

### 5.4 Agent API in `game.rs`

Used by both the headless evaluator and the GUI watcher:

- `agent_start(first_level, preset)` — apply easy/medium/hard settings, start a
  fresh run at `first_level % 50`.
- `agent_step(turn, forward, strafe, fire, activate_exit, weapon, run, dt)` —
  clamp `dt` to 0.05, select weapon if unlocked, turn, move, optionally use the
  exit, optionally fire, then call `update(dt)`. It is a no-op when
  dead/won/paused/settings-open.
- `agent_observation() -> String` — one-line JSON observation (see §7.7).
- `agent_policy_action(weights) -> (turn, forward, strafe, fire,
  activate_exit, weapon, run)` — computes the 42-value input vector, applies
  the linear policy, and decodes 10 tanh outputs. If `weights.len() != 430`,
  it returns a no-op action `(0, 0, 0, false, false, 0, true)`.

The 42-input encoder and 10-output decoder are duplicated in
`training/env.py`; they must stay in sync. See §7.2.

### 5.5 Renderer (`render.rs`)

Pure software rendering into `game.pixels`; no external art assets.

- Constants are defined in `main.rs`/`doom_headless.rs`: `WIDTH=320`,
  `HEIGHT=200`, `VIEW_HEIGHT=170`, `HUD_TOP=172`; `render.rs` uses
  `FOV_PLANE = 0.66`.
- `render()` order: floor/ceiling → walls → sprites → hit effects → weapon →
  danger indicator → HUD.
- Floor/ceiling: ceiling gradient plus ray-cast floor with checker/grid
  texture and distance shading.
- Walls: DDA raycasting with camera plane; procedural 32×32 brick/marble
  patterns, fog, side-face darkening; writes `z_buffer`.
- Sprites: enemies, pickups, exit portal; depth-sorted, z-buffer clipped,
  16×16 procedural patterns, distance shading. Exit pulses green.
- Weapon: hand-drawn procedural rectangles per weapon; recoil lift; per-weapon
  muzzle flashes.
- Hit effects: per-weapon enemy/wall/miss particles; rocket blast is a
  hand-authored 16×16 two-frame sprite scaled to the impact distance.
- Danger indicator: pulsing `DEMON IN RANGE` banner plus an edge arrow toward
  the nearest threat.
- HUD: 3×5 bitmap font, HP/ARMOR/AMMO readouts, bars, level code, weapon name,
  transient messages, and a damage flash at the top of the screen.
- `ui.rs` overlays (pause/level select, settings, death/victory) are drawn by
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
  overrides. Presets do **not** change weapon stats.
  - **Easy**: max health/armor 125, armor absorption 0.6, imp/demon health
    40/60, speeds 0.56/0.72, chase 7.5, attack 5.0, damage 5/7, delays
    1.6/1.9, medkit 40, armor pickup 50.
  - **Hard**: max health 90, max armor 75, armor absorption 0.4, imp/demon
    health 80/110, speeds 0.9/1.1, chase 10.0, attack 7.8, damage 11/15,
    delays 0.95/1.15, medkit 20, armor pickup 30, ammo caches 15/4/20/1.
- Manual edits set `difficulty = Custom` and take effect immediately.
  Applying a preset rescales living enemies' HP so their health matches the
  new preset. Settings are **not** persisted across runs; every launch starts
  at Medium.
- UI: Up/Down selects a row, Left/Right edits, Tab/PageUp/PageDown changes
  section, 1/2/3 applies a preset. Eight rows visible per page. Shift+P opens
  settings; Esc closes.
- Pause/help screen lists controls and all 50 levels. The overlay enumerates
  every level, but the panel is sized for a 540 px-high area and rows are
  31 px tall, so only about the first 15 levels fit on the default 600 px
  window. Up/Down can still select off-screen rows; there is no scrolling.
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
  `exit=x,y`; all required. Put them before the first section: after a
  `[section]` header, the parser treats every non-section line according to
  that section until another header appears.
- `[map]`: exactly **12 rows × 16 characters**. Valid tiles: `#`, `B`, `M`
  (solid), `.`, `X` (walkable; `X` is cosmetic).
- `[enemies]`: `x,y,type` with type `imp` or `demon`.
- `[pickups]`: `x,y,type` with type `shotgun`, `chaingun`,
  `rocket_launcher`, `medkit`, `armor`, or `ammo`.
- Comments start with `;` or `//`; blank lines ignored.
- Spawn, exit, enemies, and pickups must lie on open tiles.
- Coordinates must be finite; exactly two/three comma-separated values as
  appropriate.
- Unknown sections or top-level fields are errors.
- Level codes must be unique across all files.
- Files are played in lexicographic filename order; numeric prefixes (`01_`,
  `02_`, …) define campaign order.
- There is no reachability validation: a level can parse successfully while
  containing an unreachable exit or pickup.

### 6.2 Campaign contents

- 50 `.lvl` files, `01_e1m1.lvl` … `50_e5m10.lvl`, codes `E1M1` … `E5M10`.
- 50 unique codes and 50 map layouts.
- Episode themes/titles repeat across E2–E5 (e.g. `CINDER MAZE` appears as
  E2M3, E3M2, E4M1, E5M10) but the maps and placements differ.
- `doom/src/levels/README.md` documents the format and says the campaign is
  ten original levels followed by 40 maze-style levels.
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
- `best_policy.json` architecture metadata is
  `{"inputs": 42, "outputs": 10, "activation": "tanh-linear"}`.

Output → action mapping (indices 0–9):

| Output | Meaning |
| --- | --- |
| 0 | turn, clamped to `[-1, 1]` |
| 1 | forward, clamped to `[-1, 1]` |
| 2 | strafe, clamped to `[-1, 1]` |
| 3 | fire if `> 0` |
| 4 | activate exit if `> 0` |
| 5–8 | weapon preference scores; choose the highest among unlocked weapons (ties choose the lowest index) |
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
bearings. Ties preserve encounter order (both implementations sort stably by
distance).

### 7.3 Evolution loop (`training/train.py`)

- Population initialized from `N(0, initial_weight_sigma)` per weight.
- `PARAMETERS = 10 * (42 + 1) = 430`; `CAMPAIGN_LEVELS = 50`.
- Per generation:
  1. Compute `seeds = seed + generation*evaluation_episodes + i` and the
     expected start level for each seed.
  2. Build `population_size × evaluation_episodes` jobs, genome-major; each
     job is self-contained `(genome, executable, cfg, seed)`.
  3. Evaluate all jobs in parallel via `ProcessPoolExecutor` with the **spawn**
     context. `pool.map` preserves job order.
  4. Verify every result's `start_level` matches the seed-derived expectation.
  5. Average each genome's episode results (`fitness`, `levels`, `kills`,
     `health`, `seconds`; `start_level` taken from the first episode).
  6. Rank genomes by mean fitness descending.
  7. Copy the top `elite_count` genomes unchanged (clamped to at least 1 and at
     most population size).
  8. Fill the rest by choosing a parent uniformly from the top
     `min(parent_pool_size, population_size)` and mutating each gene
     independently with probability `mutation_rate`, adding
     `N(0, mutation_sigma)`.
  9. Append a metrics row and flush; overwrite the checkpoint, manifest, and
     best policy.
- No crossover, no gradient updates, no backpropagation.
- `generations` is a total count; generation numbering is 0-based internally
  and printed/displayed 1-based. `--generations N` overrides it.
- On resume, `start_generation = saved.generation + 1` and the Python RNG state
  is restored from the checkpoint. If the checkpoint is already at/above the
  configured generation count, the loop does nothing and the manifest is
  marked completed.
- If `resume` is false or no checkpoint exists, the population is reinitialized
  and metrics start fresh (`metrics.csv` truncated). `--fresh` forces this; a
  fresh run still leaves old artifacts in place until they are overwritten.
- Determinism: the population, mutation stream, and evaluation seeds all derive
  from `seed`; two fresh runs with the same config and binary produce identical
  results (subject to parallel scheduling being irrelevant because jobs are
  independent).

### 7.4 Evaluation and fitness

Each episode (`doom_headless --evaluate`):

- Starts at `random.Random(seed).randrange(50)`, passed explicitly by Python.
  All genomes in the same generation/episode share the same start level; each
  evaluation episode uses a different seed.
- Runs up to `max_steps` at `time_step_seconds` per step, or until death, or
  until **`max_levels_per_agent` completed level transitions**.
- `max_levels_per_agent` counts exits/transitions, not levels visited. The
  policy may wrap from level 50 back to level 1.
- The level in which the cap is reached is entered but not scored by the
  partial-level block (the loop breaks immediately after the transition).

Scoring in `evaluate_episode` (13 active weights, in this exact order):

1. Every step within a level:
   `score += max(last_exit_distance - current_exit_distance, 0) * exit_progress`
   where distance is straight-line distance to the current exit. Moving away
   adds nothing; it never penalizes retreat. (Because the baseline is updated
   on retreat, oscillating toward and away from the exit can farm this term.)
2. On each completed level:
   `+ level_completion`
   `+ health * health_per_point`
   `- time_spent_on_that_level * seconds`
   `+ kills_on_that_level * kill`
3. If the episode ends before `max_levels_per_agent` completions:
   `+ completed_levels * partial_level`
   `+ current_health * health_per_point`
   `- current_level_time * seconds`
   `+ current_level_kills * kill`
4. End-of-episode counters:
   `+ game.damage_dealt * damage_dealt`
   `- game.damage_taken * damage_taken`
   `- game.armor_damage_taken * armor_damage_taken`
   `+ game.health_gain * health_gain`
   `+ game.armor_gain * armor_gain`
   `- game.ammo_spent * ammo_spent`
   `- death` if the player died.
- Returned CSV: `score,level_count,total_kills,health,total_time,start_level`.
  `total_kills` counts kills on all levels; `total_time` is simulated seconds
  (formatted to 2 decimals), capped by `max_steps * time_step_seconds`.
- There is no separate survival bonus or explicit time bonus; survival is
  implicit through completion and health terms and the per-level time penalty.

Stat definitions:

- `damage_dealt`: damage actually applied (overkill excluded; rocket splash
  summed).
- `damage_taken`: health lost after armor absorption.
- `armor_damage_taken`: armor lost (separate from health loss).
- `health_gain` / `armor_gain`: only gains from medkits/armor pickups, capped
  by max values.
- `ammo_spent`: rounds actually consumed by fired shots.
- All counters are reset at episode start by `agent_start` and accumulate
  across level transitions within that episode.

### 7.5 Configuration (`config.json`)

| Key | Default | Meaning |
| --- | ---: | --- |
| `cpu_threads` | 0 | 0 = all available CPUs; -1 = all but one; positive = exact worker count |
| `generations` | 50 | Total generations to run |
| `population_size` | 48 | Genomes per generation |
| `elite_count` | 4 | Top genomes copied unchanged |
| `parent_pool_size` | 12 | Top genomes eligible as parents |
| `initial_weight_sigma` | 0.35 | Std-dev for initial weights |
| `mutation_rate` | 0.08 | Per-gene mutation probability |
| `mutation_sigma` | 0.18 | Std-dev of mutation noise |
| `time_step_seconds` | 0.05 | Simulation step; must be in `(0, 0.05]` |
| `max_steps` | 15000 | Per-episode step cap (15000 × 0.05 = 750 s) |
| `max_levels_per_agent` | 10 | Maximum completed level transitions per episode; 1 ≤ value ≤ campaign_levels |
| `evaluation_episodes` | 2 | Episodes averaged per genome per generation |
| `doom_difficulty` | `medium` | `easy`, `medium`, or `hard` |
| `campaign_levels` | 50 | Must equal 50 (hard-coded check) |
| `seed` | 2001901 | Base RNG seed |
| `resume` | false | Resume from checkpoint if present |
| `output_directory` | `training/output` | Resolved relative to repo root |
| `checkpoint_file` | `checkpoint.json` | Population/RNG/provenance checkpoint |
| `best_policy_file` | `best_policy.json` | Best genome for the GUI watcher |
| `metrics_file` | `metrics.csv` | Per-generation CSV (first column `run_id`) |
| `manifest_file` | `run_manifest.json` | Run provenance/status |

`fitness` defaults (all 13 are active):

| Key | Default | Applied as |
| --- | ---: | --- |
| `level_completion` | 1000.0 | add per completed level |
| `partial_level` | 100.0 | add per completed level when the episode ends early |
| `health_per_point` | 2.0 | add current/transition health |
| `kill` | 20.0 | add kills |
| `seconds` | 0.25 | subtract per-level elapsed time |
| `exit_progress` | 8.0 | add positive per-step distance reduction |
| `damage_dealt` | 0.25 | add applied damage |
| `damage_taken` | 1.0 | subtract health lost |
| `armor_damage_taken` | 0.5 | subtract armor lost |
| `health_gain` | 0.5 | add actual medkit healing |
| `armor_gain` | 0.25 | add actual armor gained |
| `ammo_spent` | 0.1 | subtract consumed ammo |
| `death` | 250.0 | subtract if the player died |

Validation in `train.py` (strict schema):

- Unknown top-level keys are rejected. Missing keys fail except `resume`,
  `max_levels_per_agent`, and `manifest_file`, which have defaults filled by
  `DEFAULT_CONFIG`.
- `cpu_threads` must be -1, 0, or positive; `generations`, `population_size`,
  `elite_count`, `parent_pool_size`, `max_steps`, `evaluation_episodes`, and
  `max_levels_per_agent` must be positive integers.
- `elite_count` and `parent_pool_size` must not exceed `population_size`.
- `campaign_levels` must equal 50; `max_levels_per_agent` must not exceed it.
- `doom_difficulty` must be easy/medium/hard (case-insensitive).
- `time_step_seconds` in `(0, 0.05]`; all numeric values finite.
- `fitness` must be an object containing exactly the 13 keys above, all finite
  numbers; unknown or missing fitness keys fail.
- Output file names must be non-empty strings.
- The release executable `doom/target/release/doom_headless` must exist unless
  `--build` is passed; `--build` runs `exec_mgmt.sh -b` first.
- The effective configuration is printed before training.

Provenance and resume fingerprints:

- `config_hash` = SHA-256 of the full validated config (sorted compact JSON),
  including scheduling keys (`cpu_threads`, `generations`, `resume` as set for
  the run).
- `compatibility_hash` = SHA-256 of the same config excluding only
  `cpu_threads`, `generations`, and `resume`. This is the resume fingerprint:
  changing the output directory, seed, reward weights, level cap, difficulty,
  or other behavioral settings requires `--fresh`.
- `run_id` = `{seed}-{config_hash[:12]}-{UTC timestamp with microseconds}`.
- The executable's SHA-256 is recorded; resuming requires the current binary to
  match exactly (rebuild or use `--fresh`).
- A checkpoint must contain provenance metadata (`run_id`, `seed`,
  `config_hash`, `config`, `executable_sha256`, `generation`, `population`,
  `random_state`); older checkpoints are rejected with a `--fresh` hint.

### 7.6 Outputs

`training/output/checkpoint.json` (compact JSON, overwritten each generation):

```json
{
  "run_id": "2001901-123b64e24671-20261113T023952050906Z",
  "seed": 2001901,
  "config_hash": "...",
  "compatibility_hash": "...",
  "config": { /* full validated config */ },
  "executable": "/abs/path/doom/target/release/doom_headless",
  "executable_sha256": "...",
  "git_commit": "...",
  "git_dirty": true,
  "python_version": "3.9.6",
  "rust_version": "rustc ...",
  "start_time": "...",
  "resume_state": {"enabled": false, "loaded_checkpoint": false, "checkpoint_generation": null},
  "generation": 0,
  "evaluation_seed_offset": 0,
  "next_evaluation_seed": 2001903,
  "population": [[/* 430 floats */]],
  "best_result": {"fitness": 0.0, "levels": 0.0, "kills": 0.0,
                   "health": 0.0, "seconds": 0.0, "start_level": 0},
  "best_genome": [/* 430 floats */],
  "random_state": [/* Python random state: version, 625 ints, gauss_next */],
  "created_at": "...",
  "updated_at": "..."
}
```

`training/output/best_policy.json` (pretty JSON, sorted keys, overwritten each
generation):

```json
{
  "architecture": {"inputs": 42, "outputs": 10, "activation": "tanh-linear"},
  "run_id": "...",
  "seed": 2001901,
  "config_hash": "...",
  "compatibility_hash": "...",
  "generation": 0,
  "doom_difficulty": "medium",
  "result": {"fitness": 0.0, "levels": 0.0, "kills": 0.0,
              "health": 0.0, "seconds": 0.0, "start_level": 0},
  "weights": [/* exactly 430 floats */]
}
```

- `generation` is 0-based; the GUI displays `generation + 1`.
- The GUI's naive parser depends on the key names and weights array shape. It
  ignores the extra `run_id`, `seed`, `config_hash`, and `compatibility_hash`
  fields.

`training/output/metrics.csv` columns:

```text
run_id,generation,best_fitness,mean_fitness,levels,kills,health,seconds,start_level
```

- One row per generation; `best_*` values are the best genome's episode
  averages, not a single episode. `levels`, `kills`, `health`, `seconds`, and
  `start_level` are the best genome's summary values.
- Appended when resuming; overwritten when starting fresh.
- On resume, the trainer verifies the header/format, that rows start at
  generation 0 and are contiguous, that every row belongs to the checkpoint's
  `run_id`, that each row's `start_level` matches the seed-derived value, and
  that the last row's generation equals the checkpoint generation.

`training/output/run_manifest.json` (pretty JSON, updated each generation and
marked `"status": "completed"` at the end):

- Run identity: `run_id`, `seed`, `config_hash`, `compatibility_hash`, full
  validated `config`.
- Environment: `git_commit`, `git_dirty`, `python_version`, `rust_version`,
  `executable`, `executable_sha256`.
- Lifecycle: `created_at`, `updated_at`, `resume` (`enabled`,
  `loaded_checkpoint`, `checkpoint_generation`), `checkpoint_generation`,
  `status` (`running` or `completed`).
- When resuming with an existing manifest, the manifest's `run_id`, `seed`,
  compatibility hash, and executable hash must match; otherwise resumption is
  refused. If the manifest is missing, it is recreated.

### 7.7 Headless protocol (`doom_headless.rs`)

**Interactive mode** (stdin lines, stdout one JSON line per command):

```text
reset,<level_index>,<easy|medium|hard>
step,<turn>,<forward>,<strafe>,<fire 0|1>,<exit 0|1>,<weapon>,<run 0|1>,<dt>
quit
```

- `step` requires at least 9 comma-separated fields (the first is `step`); the
  action values are parsed as `f32`, booleans use `!= 0.0`, weapon uses
  `max(0) as usize`.
- Invalid commands return `{"error":"invalid command"}`.
- `reset` calls `agent_start` and returns an observation. It does not reset
  `game.time`.
- `step` calls `agent_step` and returns an observation.
- Observation schema (one compact line, no spaces):

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
  <start_level> <difficulty> <dt> <max_steps> <max_levels_per_agent> \
  <level_completion> <partial_level> <health_per_point> <kill> <seconds> \
  <exit_progress> <damage_dealt> <damage_taken> <armor_damage_taken> \
  <health_gain> <armor_gain> <ammo_spent> <death>
```

- Exactly 18 arguments after `--evaluate`; the last 13 are fitness weights in
  the order listed above.
- Policy weights are read from stdin as comma-separated floats; exactly 430
  required.
- stdout: `score,level_count,total_kills,health,total_time,start_level`
  (`total_time` formatted to 2 decimals).
- Exit codes: 1 = level initialization failure, 2 = evaluation failure
  (including bad argument count, invalid numbers, or wrong weight count).
- `doom_headless` compiles `game.rs`, `levels.rs`, `render.rs`, and
  `settings.rs` via `#[path]` includes. It does not include `ui.rs` or
  `main.rs`, never opens a window, and never calls `render()`.

### 7.8 Python files

`training/env.py`:

- `FITNESS_WEIGHT_KEYS` — the canonical 13-key order shared with
  `doom_headless.rs` and `train.py`.
- `HeadlessDoomEnv` — persistent process adapter with
  `reset(level)`, `step(action)`, `close()`, context-manager support, and a
  `max_levels` cap (constructor default 10). `step` reports `done` when dead or
  after `max_levels` level transitions. It requires `reset()` before `step()`
  because it compares against the previous observation.
- `encode_observation(obs)` — Python version of the 42-value input vector.
- `choose_action(weights, obs)` — Python version of the policy forward pass
  and action decoding.
- `evaluate(weights, executable, cfg, seed)` — the function actually used by
  training; computes the start level, launches `doom_headless --evaluate` with
  the five run parameters and 13 weights, passes weights on stdin, and parses
  the CSV result.
- The interactive class is not used by `train.py`; it is useful for debugging
  or writing new trainers.

`training/train.py`:

- Constants: `PARAMETERS = 430`, `CAMPAIGN_LEVELS = 50`.
- `tester_count` resolves `cpu_threads` using `os.process_cpu_count()` when
  available, then `sched_getaffinity`, then `os.cpu_count()`.
- Configuration validation/defaults, `canonical_config`, `compute_config_hash`,
  `compute_compatibility_hash`, `sha256_file`, run-id/manifest helpers.
- `mutate`, `run_job`, `summarize_episodes`, `list_state`, `tuple_state`.
- `main()` handles CLI/config validation, build/executable resolution,
  checkpoint/manifest/metrics validation, resume, the generation loop,
  checkpointing, and policy export.

Python 3.9+ is sufficient (the local system Python is 3.9.6).

## 8. Shell scripts

| Script | Behavior |
| --- | --- |
| `exec_mgmt.sh` | `cd doom`; `-b` = `cargo build -r --bins`; `-r` = run `target/release/doom` if present, else `cargo run -r`; `-h`/no args = help. Must be run from repo root. |
| `ready.sh` | `./exec_mgmt.sh -b` then `./clear_training_data.sh`; fresh training setup. Destructive. No shebang and currently mode 644: run it as `bash ready.sh` or `chmod +x ready.sh` first. |
| `clear_training_data.sh` | `rm -rf training/output/`. Destructive. Executable, no shebang. |
| `watch_agent.sh` | Resolves repo root from its own path, builds if `doom/target/release/doom` is missing, then `exec doom --watch-agent "${1:-training/output/best_policy.json}"` from the repo root. |
| `doom/run.sh` | `./target/release/doom` (run from `doom/`). |
| `doom/DOOM.app/.../doom-launcher` | macOS wrapper: finds `doom/target/release/doom`, shows an alert if missing, runs it with output suppressed. |

## 9. Invariants and gotchas

Read these before changing anything.

1. **430 is hard-coded in multiple places.**
   - `training/train.py`: `PARAMETERS = 10 * (42 + 1)`.
   - `doom/src/main.rs`: `if weights.len() != 430`.
   - `doom/src/bin/doom_headless.rs`: `if weights.len() != 430`.
   - `best_policy.json` architecture metadata in `train.py`.
   - The GUI policy parser requires exactly 430 weights.
   Changing the policy size requires updating all of them plus the input
   encoder/decoder.

2. **The 42-input encoder is duplicated.**
   - Python: `training/env.py::encode_observation`.
   - Rust: `game.rs::agent_policy_action`.
   Any change to ray count, entity features, ordering, or normalization must
   be mirrored exactly or Python-side and Rust-side policies will diverge.

3. **The episode cap is configurable but still duplicated.**
   - `max_levels_per_agent` is passed from Python to Rust and validated
     (`1 ..= campaign_levels`).
   - `HeadlessDoomEnv` has a separate constructor cap (default 10).
   - Rust's evaluator only checks `max_levels > 0`; Python's config validation
     is what enforces the campaign bound.
   `max_levels_per_agent` counts completed level transitions, not levels
   visited; the level entered when the cap is hit is not scored by the
   partial-level block.

4. **All 13 fitness weights are active and order-sensitive.**
   - The order is defined once in `training/env.py::FITNESS_WEIGHT_KEYS` and
     must match the argument order in `doom_headless.rs::evaluate_episode`.
   - `exit_progress` rewards only positive distance reductions; it never
     penalizes retreat and can be farmed by oscillating toward/away from the
     exit.
   - `damage_taken` counts health lost after armor; armor loss is a separate
     term. `health_gain`/`armor_gain` count only pickup gains.

5. **`campaign_levels` must be 50.** `train.py` exits otherwise. Adding a
   51st level requires changing `CAMPAIGN_LEVELS`, the validation check, and
   `config.json`; `max_levels_per_agent` must stay ≤ the campaign size.

6. **`time` never resets between runs.** It is initialized only in
   `Game::new()`. In long-lived processes (GUI watcher, interactive env) the
   policy's `time/120` input saturates after two minutes. The evaluator starts
   a fresh process per episode, so this does not affect training. Add a reset
   in `start_new_run`/`restart` if it matters.

7. **Headless agents wrap the campaign forever; GUI manual play wins.**
   `agent_step` uses `(level_index + 1) % 50` and never sets `won`. The GUI
   watcher instead restarts after 50 level transitions or death. Do not rely
   on `won` in headless mode.

8. **Level changes require a rebuild.** Levels are embedded by `build.rs`;
   the runtime never reads `.lvl` files.

9. **`training/output/` is not gitignored.** Avoid committing checkpoints,
   policies, metrics, or manifests. `stats/` is tracked analysis data.
   `training/__pycache__/` is tracked in this snapshot. `ready.sh` and
   `clear_training_data.sh` delete the output directory.

10. **The GUI policy parser is not real JSON parsing.** It searches for the
    first `"generation"`, `"doom_difficulty"`, and `"weights"` strings. Keep
    `best_policy.json` shaped as emitted by `train.py`, or update
    `main.rs::load_agent_policy`. A missing or invalid policy is silently
    ignored (the previous valid policy, if any, remains active), and the
    watcher shows its waiting status.

11. **The GUI watcher reloads weights mid-episode.** If only weights/generation
    change, the current episode continues with the new policy; difficulty
    changes force a new random-level episode.

12. **Settings are not persisted.** Every launch starts at Medium. The GUI
    watcher uses the difficulty stored in `best_policy.json`, not the GUI
    preset. Manual setting edits are immediate and set the preset to Custom.

13. **`render.rs` is compiled into the headless binary** even though it is
    never called. Removing it from the `#[path]` module list breaks the
    headless build.

14. **Strict configuration schema.** Unknown top-level keys and unknown or
    missing fitness keys are fatal. Adding a config key requires updating
    `REQUIRED_KEYS`/`ALLOWED_KEYS` (and `DEFAULT_CONFIG` if optional), then
    validation and any consumers.

15. **Resume is fingerprint-checked.** The compatibility hash excludes only
    `cpu_threads`, `generations`, and `resume`; it includes `output_directory`,
    `seed`, fitness, level cap, and other behavioral settings. The executable
    SHA-256 must also match. Checkpoints require provenance metadata. Use
    `--fresh` after changing any of these.

16. **Resume also validates metrics and the manifest.** The metrics file must
    start at generation 0, be contiguous, belong to the same `run_id`, have
    the seed-derived start level for every row, and end at the checkpoint
    generation. A missing manifest is recreated; a mismatched manifest is
    fatal.

17. **Evaluation start levels are deterministic per seed.** The trainer
    computes the expected level in Python and aborts if Rust returns a
    different one. Changing `seed`, `campaign_levels`, or
    `evaluation_episodes` invalidates old metrics/resume state.

18. **Rockets damage every living enemy within the blast radius** of the
    target, with no line-of-sight check and no self-damage. A rocket that hits
    no enemy only creates an impact effect. `damage_dealt` is capped at each
    enemy's remaining HP per hit.

19. **The Python interactive env assumes `reset()` before `step()`.** `step`
    dereferences the previous observation to detect level changes. Reusing a
    process across episodes also carries over `game.time` (training does not
    use this path).

20. **Evaluation is local.** The job is self-contained (genome, executable
    path, config, seed), which is a clean seam for future distributed
    evaluation, but no distribution code exists.

21. **The checkout may have local changes or be on a non-default branch.**
    Inspect `git status`/`git branch` before branch-based work. The snapshot
    used for this guide is `e2affad` on `Analysis`.

22. **macOS/Windows notes.** Windows uses a GUI subsystem attribute to avoid a
    console window. macOS ships a `DOOM.app` wrapper. Shell scripts are LF via
    `.gitattributes`. `ready.sh` is currently not executable and has no
    shebang.

23. **The pause level list overflows the default window.** `ui.rs` draws all
    50 level rows at 31 px spacing inside a 540 px panel, so only about 15 are
    visible at the default 960×600 window. Selection via Up/Down still works
    for off-screen rows, but there is no scrolling.

24. **The evaluator's final-level scoring is asymmetric.** At the level cap,
    the transition awards completion/health/time/kill terms for the level just
    finished, but the newly entered level gets no partial-level terms. At
    `max_steps` exhaustion (no death, fewer than the cap), the partial block
    applies to the current level.

## 10. Common change recipes

- **Add a level:** create `doom/src/levels/NN_xxx.lvl` with a unique code and
  exactly 12×16 map; put top-level fields before sections; keep
  spawn/exit/enemies/pickups on open tiles; rebuild. If the total leaves 50,
  update `CAMPAIGN_LEVELS`/`campaign_levels` and the `train.py` check.
- **Tune training:** edit `config.json`; no code changes are needed for the
  supported keys. Use `--generations` for a quick override and `--fresh` to
  abandon a checkpoint. The effective config is printed at startup.
- **Change reward shaping:** edit the 13 active weights in `config.json`.
  For new terms, update `FITNESS_WEIGHT_KEYS` in `env.py`, the argument list
  and struct in `doom_headless.rs::evaluate_episode`, the scoring loop, the
  config schema/validation, and this guide.
- **Change policy size/architecture:** update `train.py` constants and
  architecture metadata, `env.py` encoder/decoder, `game.rs`
  `agent_policy_action`, both 430-weight checks, and the GUI parser's 430
  check. Update the observation docs here.
- **Change game behavior:** edit `game.rs`; both GUI and training change
  together. Rebuild both binaries. Update `AGENTS.md` if contracts change.
- **Change controls/UI:** `game.rs::handle_input`/`handle_settings_input`,
  `ui.rs`, and the HUD in `render.rs`.
- **Change difficulty:** `settings.rs` presets/defaults. Headless episodes use
  the preset string from `config.json`; the GUI watcher uses the policy file's
  difficulty. Weapon stats come from `GameSettings::medium()` and are not
  changed by presets.
- **Watch a different policy:** `./watch_agent.sh path/to/policy.json`, or
  `./doom/target/release/doom --watch-agent path/to/policy.json`.
- **Manage runs:** set `resume` in `config.json` to continue; use `--fresh` to
  start over; use `clear_training_data.sh` (destructive) or a different
  `output_directory` for a completely clean directory. Changing
  `output_directory` changes the compatibility hash.
- **Debug an episode:** call `training/env.py::evaluate` directly, or run
  `doom_headless --evaluate` with the 18 arguments and 430 weights on stdin.
  For step-by-step control, use `HeadlessDoomEnv` (reset first).

## 11. Verification checklist

There is no automated test suite or CI. Verify manually:

```sh
# Rust compile check
cd doom && cargo check --bins

# Release build (what scripts/training use)
./exec_mgmt.sh -b

# Python syntax
python3 -m py_compile training/env.py training/train.py

# Smoke training (builds if needed, writes training/output; back up any valuable run)
python3 training/train.py --build --generations 1 --fresh

# Inspect artifacts
python3 - <<'PY'
import json
p = json.load(open("training/output/best_policy.json"))
print(len(p["weights"]), p["generation"], p["doom_difficulty"])
print(json.load(open("training/output/run_manifest.json"))["status"])
PY
```

Expected smoke result: a `generation 1/1` line, a checkpoint with a
`population_size × 430` population, `best_policy.json` with exactly 430
weights, and a manifest whose `status` becomes `completed`. Run a second
generation with `resume: true` to exercise resume; the metrics file should
append generation 1 and keep the same `run_id`.

Other checks:

```sh
# Interactive headless protocol
./doom/target/release/doom_headless
# then: reset,0,medium / step,1,1,0,1,0,0,0,0.05 / quit

# Direct evaluator (430 comma-separated weights on stdin)
./doom/target/release/doom_headless --evaluate 0 medium 0.05 15000 10 \
  1000 100 2 20 0.25 8 0.25 1 0.5 0.5 0.25 0.1 250 < weights.txt

# GUI
./exec_mgmt.sh -r
# or: open doom/DOOM.app

# Watch a trained policy
./watch_agent.sh
```

A successful smoke run should print a generation line and leave all four
artifacts in `training/output/`. The GUI and watcher should run without a
panic; the watcher prints/waits for a policy if none is present.

## 12. Glossary

- **Genome / policy / weights:** the flat 430-float vector.
- **Generation:** one full evaluate → rank → select → mutate cycle.
- **Episode:** one headless run from a seed-derived start level until death,
  `max_levels_per_agent` level transitions, or `max_steps`.
- **Start level:** `random.Random(seed).randrange(50)`, shared by all genomes
  in the same generation/episode.
- **Elite:** a top genome copied unchanged into the next generation.
- **Tick / step:** one simulation update of `time_step_seconds` (max 0.05 s).
- **Headless:** the `doom_headless` binary; same simulation as the GUI, no
  window, stdin/stdout protocol.
- **Watch mode:** GUI mode that loads a policy and plays automatically.
- **Run ID:** `seed-configHashPrefix-UTC timestamp`, written to all run
  artifacts and used to reject mixed metrics.
- **Config hash:** SHA-256 of the full validated config, used in the run ID
  and records.
- **Compatibility hash / resume fingerprint:** config hash excluding
  `cpu_threads`, `generations`, and `resume`; resume must match it.
- **Manifest:** `run_manifest.json`; run provenance, environment, resume state,
  and lifecycle status.
- **`max_levels_per_agent`:** per-episode cap on completed level transitions.