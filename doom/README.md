# DOOM: Hell on Mars

A small, asset-free first-person shooter written in Rust. It uses a 320×200 software raycaster and scales the image to a resizable window, so the game keeps a small rendering workload on older machines.

On macOS, open [`DOOM.app`](DOOM.app) to launch the game without opening a Terminal window. The app runs the existing release build in the background. If the release executable is missing, build it from the project folder with:

```sh
cargo build --release
```

For development, you can also run it with `cargo run --release`.

The game opens on a paused controls and level-selection screen. Use **Up / Down** to choose a level and **Enter** to start there. Press **Q** to quickplay from the first level and progress through the campaign. Press **F1** (or **Fn+F1** on keyboards that send F1 through the Fn key) during play to pause and reopen the screen.

## Machine learning trainer

From the repository root, build the game and its headless training executable
with `./exec_mgmt.sh -b`, then start training with
`python3 training/train.py`. Adjust tester count, evolution settings, fitness,
and output paths in the root `config.json`. See [`training/README.md`](../training/README.md)
for the environment interface and configuration details.

## Controls

- **W / S**: move forward and backward
- **A / D**: strafe
- **Left / Right arrows**: turn
- **Shift**: run
- **Ctrl**, **Space**, or **left mouse button**: fire
- **1–4** or **mouse wheel**: switch weapons you have collected
- **E**: activate the green exit portal when nearby
- **F1 / Fn+F1 / Esc**: pause and show help
- **Shift + P**: open the settings menu
- **Up / Down**: choose a level in the pause screen
- **Enter**: start the selected level
- **Q**: quickplay from level one, then advance through exits
- **Esc / Space / F1**: resume from help
- **F10**: quit from the help / level-selection screen
- **R**: restart after death or completing the mission

In settings, use **Up / Down** to select a value, **Left / Right** to change it, **Tab** (or **Page Up / Page Down**) to change sections, and **1 / 2 / 3** to apply Easy / Medium / Hard. Manual edits switch the preset label to Custom. The default preset is Medium.

The four settings sections cover player health, armor, movement and ammo; each weapon's damage, range, spread, fire delay and ammo cost; demon health, movement, detection and attack stats; and pickup amounts and reach. Edits take effect immediately.

Find the shotgun, chaingun, and rocket launcher pickups across the ten levels. Walk over health, armor, and ammo supplies to collect them. A pulsing **DEMON IN RANGE** alert appears when a demon has a clear shot at you, with an edge marker pointing toward the nearest threat.

Level sources are the editable `.lvl` files in [`src/levels/`](src/levels/README.md). They are embedded into the executable during compilation; the game does not read or modify them at runtime. See the folder's README for the level format and ordering rules.
