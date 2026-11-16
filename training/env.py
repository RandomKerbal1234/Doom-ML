"""Headless adapter for the Rust DOOM simulation."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path


FITNESS_WEIGHT_KEYS = (
    "level_completion",
    "partial_level",
    "health_per_point",
    "kill",
    "seconds",
    "exit_progress",
    "damage_dealt",
    "damage_taken",
    "armor_damage_taken",
    "health_gain",
    "armor_gain",
    "ammo_spent",
    "death",
)

# Encoder shape. These constants are duplicated in doom/src/game.rs and must
# stay in sync with it (`Game::agent_inputs`).
RAY_COUNT = 12
ACTION_MEMORY = 6
OUTPUT_COUNT = 10
RAY_MEMORY_DECAY = 0.75
# rays + enemies(2x5) + pickup(9) + exit(3) + health/armor(2) + ammo(4) +
# unlocked(4) + weapon/time(2) + ray memory + action memory
INPUT_COUNT = RAY_COUNT + 10 + 9 + 3 + 2 + 4 + 4 + 2 + RAY_COUNT + ACTION_MEMORY


def start_level_bounds(cfg):
    """Return the inclusive 1-based `(low, high)` campaign-level range.

    `start_level` is either a two-element array `[low, high]` or the integer
    shorthand: `0` means the whole campaign and `N` means `[N, N]`.
    """
    value = cfg.get("start_level", 0)
    levels = int(cfg["campaign_levels"])
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("start_level must be a two-element [low, high] array")
        low, high = value
        if any(
            isinstance(bound, bool) or not isinstance(bound, int) for bound in (low, high)
        ):
            raise ValueError("start_level bounds must be integers")
    elif isinstance(value, int) and not isinstance(value, bool):
        low, high = (1, levels) if value == 0 else (value, value)
    else:
        raise ValueError("start_level must be a [low, high] array or an integer")
    if not 1 <= low <= high <= levels:
        raise ValueError(
            f"start_level must satisfy 1 <= low <= high <= {levels}"
        )
    return low, high


def resolve_start_level(cfg, seed: int) -> int:
    """Return the 0-based level an episode should start on.

    The configured range is inclusive and 1-based, so `[3, 5]` can start on
    campaign levels 3, 4, or 5 (indices 2, 3, 4), chosen uniformly from the
    evaluation seed.
    """
    low, high = start_level_bounds(cfg)
    return random.Random(seed).randrange(low - 1, high)


class HeadlessDoomEnv:
    def __init__(self, executable: Path, dt: float = 0.05, difficulty: str = "medium",
                 max_levels: int = 10):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.dt = dt
        self.difficulty = difficulty
        self.max_levels = max(1, int(max_levels))
        self.observation = None
        self.levels_completed = 0

    def _request(self, line: str):
        if self.process.poll() is not None:
            raise RuntimeError("doom_headless exited unexpectedly")
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        self.observation = result
        return result

    def reset(self, level: int):
        self.levels_completed = 0
        return self._request(f"reset,{level},{self.difficulty}")

    def step(self, action):
        old = self.observation
        command = "step," + ",".join(
            "1" if x is True else "0" if x is False else str(x)
            for x in (*action, self.dt)
        )
        new = self._request(command)
        if new["level"] != old["level"]:
            self.levels_completed += 1
        done = new["dead"] or self.levels_completed >= self.max_levels
        return new, done

    def close(self):
        if self.process.poll() is None:
            try:
                assert self.process.stdin
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def encode_observation(obs):
    """Fixed-size normalized state vector suitable for a small policy.

    Kept as a stateless convenience wrapper; recurrent inputs are zero.
    """
    return observe(obs)


def _ray_distance(x, y, angle, rows):
    """Distance to the first solid tile along `angle`, normalized to [0, 1].

    Grid traversal (Amanatides-Woo) matching `Game::ray_distance` in
    doom/src/game.rs. Both sides run in f64 so the values agree closely.
    """
    maximum = 8.0
    direction_x, direction_y = math.cos(angle), math.sin(angle)
    cell_x, cell_y = math.floor(x), math.floor(y)
    delta_x = math.inf if direction_x == 0.0 else abs(1.0 / direction_x)
    delta_y = math.inf if direction_y == 0.0 else abs(1.0 / direction_y)
    if direction_x < 0.0:
        step_x = -1
        side_x = (x - cell_x) * delta_x
    else:
        step_x = 1
        side_x = (cell_x + 1 - x) * delta_x
    if direction_y < 0.0:
        step_y = -1
        side_y = (y - cell_y) * delta_y
    else:
        step_y = 1
        side_y = (cell_y + 1 - y) * delta_y
    height = len(rows)
    width = len(rows[0]) if rows else 0
    distance = 0.0
    while distance <= maximum:
        if side_x < side_y:
            distance, side_x = side_x, side_x + delta_x
            cell_x += step_x
        else:
            distance, side_y = side_y, side_y + delta_y
            cell_y += step_y
        if cell_x < 0 or cell_y < 0 or cell_x >= width or cell_y >= height:
            break
        if rows[cell_y][cell_x] in "#BM":
            break
    return min(min(distance, maximum) / maximum, 1.0)


def _ray_values(x, y, angle, rows):
    """Wall distance for each ray of the fan."""
    step = math.tau / RAY_COUNT
    return [_ray_distance(x, y, angle + ray * step, rows) for ray in range(RAY_COUNT)]


class PolicyState:
    """Leaky recurrent inputs fed back into the policy.

    Mirrors `PolicyMemory` in doom/src/game.rs and is reset by `agent_start`.
    """

    __slots__ = ("rays", "action")

    def __init__(self):
        self.rays = [0.0] * RAY_COUNT
        self.action = [0.0] * ACTION_MEMORY

    def reset(self):
        self.rays = [0.0] * RAY_COUNT
        self.action = [0.0] * ACTION_MEMORY
        return self


def observe(obs, state=None):
    """Policy input vector. Must match `Game::agent_inputs` in game.rs."""
    if state is None:
        state = PolicyState()
    x, y, angle = obs["x"], obs["y"], obs["angle"]
    rows = obs["map"]
    values = _ray_values(x, y, angle, rows)

    enemies = sorted(obs["enemies"], key=lambda e: math.hypot(e["x"] - x, e["y"] - y))[:2]
    for i in range(2):
        if i < len(enemies):
            enemy = enemies[i]
            dx, dy = enemy["x"] - x, enemy["y"] - y
            bearing = math.atan2(dy, dx) - angle
            values.extend((math.sin(bearing), math.cos(bearing), min(math.hypot(dx, dy) / 12, 1),
                           min(enemy["hp"] / 200, 1), float(enemy["kind"])))
        else:
            values.extend((0, 0, 1, 0, 0))

    if obs["pickups"]:
        pickup = min(obs["pickups"], key=lambda p: math.hypot(p["x"] - x, p["y"] - y))
        dx, dy = pickup["x"] - x, pickup["y"] - y
        bearing = math.atan2(dy, dx) - angle
        kinds = ("shotgun", "chaingun", "rocket_launcher", "medkit", "armor", "ammo")
        values.extend((math.sin(bearing), math.cos(bearing), min(math.hypot(dx, dy) / 12, 1)))
        values.extend(float(pickup["kind"] == kind) for kind in kinds)
    else:
        values.extend((0, 0, 1, 0, 0, 0, 0, 0, 0))

    dx, dy = obs["exit"][0] - x, obs["exit"][1] - y
    bearing = math.atan2(dy, dx) - angle
    values.extend((math.sin(bearing), math.cos(bearing), min(math.hypot(dx, dy) / 16, 1)))
    values.extend((obs["health"] / 200, obs["armor"] / 200))
    values.extend(min(max(a, 0) / 100, 1) for a in obs["ammo"])
    values.extend(float(v) for v in obs["unlocked"])
    values.extend((obs["weapon"] / 3, min(obs["time"] / 120, 1)))

    values.extend(state.rays)
    values.extend(state.action)
    if len(values) != INPUT_COUNT:
        raise RuntimeError(f"encoder produced {len(values)} inputs, expected {INPUT_COUNT}")
    return values


def policy_shape(cfg):
    """Input/hidden/output widths for the configured policy."""
    return (INPUT_COUNT, *cfg.get("hidden_layers", ()), OUTPUT_COUNT)


def parameter_count(shape):
    """Flat weight count for a shape, including one bias per neuron."""
    return sum(fan_out * (fan_in + 1) for fan_in, fan_out in zip(shape, shape[1:]))


def forward_policy(weights, shape, inputs):
    """Layer-major tanh MLP. Must match `Policy::forward` in game.rs."""
    expected = parameter_count(shape)
    if len(weights) != expected:
        raise ValueError(f"expected {expected} weights for shape {shape}, got {len(weights)}")
    current = list(inputs)
    cursor = 0
    for fan_in, fan_out in zip(shape, shape[1:]):
        width = fan_in + 1
        next_layer = []
        for out in range(fan_out):
            start = cursor + out * width
            total = weights[start + fan_in]
            for index in range(fan_in):
                total += weights[start + index] * current[index]
            next_layer.append(math.tanh(total))
        cursor += fan_out * width
        current = next_layer
    return current


def choose_action(weights, obs, state=None, shape=None):
    shape = (INPUT_COUNT, OUTPUT_COUNT) if shape is None else tuple(shape)
    inputs = observe(obs, state)
    outputs = forward_policy(weights, shape, inputs)
    action = [outputs[0], outputs[1], outputs[2], outputs[3] > 0, outputs[4] > 0,
              0, outputs[9] > 0]
    available = [i for i, unlocked in enumerate(obs["unlocked"]) if unlocked]
    action[5] = max(available, key=lambda i: outputs[5 + i])
    if state is not None:
        state.action = [action[0], action[1], action[2],
                        float(action[3]), float(action[4]), float(action[6])]
        state.rays = [old * RAY_MEMORY_DECAY + new * (1.0 - RAY_MEMORY_DECAY)
                      for old, new in zip(state.rays, inputs[:RAY_COUNT])]
    return action


def evaluate(weights, executable: Path, cfg, seed: int, profile: bool = False):
    """Evaluate one episode in a one-shot subprocess.

    Kept for compatibility and single-shot benchmarking. Training uses
    `Evaluator` below, which keeps a server process alive between episodes.
    """
    with Evaluator(executable, shape=policy_shape(cfg), profile=profile) as evaluator:
        return evaluator.evaluate(weights, cfg, seed)


class Evaluator:
    """A persistent `doom_headless --evaluate-server` process.

    Each request re-seeds the reusable `Game` and runs a complete episode. The
    expensive parts that used to run per episode -- process launch, dynamic
    linking, level parsing, and the initial game allocation -- happen once per
    worker instead.
    """

    def __init__(self, executable: Path, shape=None, profile: bool = False):
        self.executable = Path(executable)
        self.shape = (INPUT_COUNT, OUTPUT_COUNT) if shape is None else tuple(shape)
        self.parameters = parameter_count(self.shape)
        self.profile = profile
        self.process = None
        self.starts = 0
        self._start()

    def _start(self):
        environment = os.environ.copy()
        if self.profile:
            environment["DOOM_PROFILE"] = "1"
        self.process = subprocess.Popen(
            [str(self.executable), "--evaluate-server",
             "--arch", ",".join(map(str, self.shape))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=environment,
        )
        self.starts += 1

    def _request(self, line: str):
        if self.process.poll() is not None:
            raise RuntimeError("doom_headless evaluate-server exited")
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline()
        if not reply:
            raise RuntimeError("doom_headless evaluate-server closed its output")
        result = json.loads(reply)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def evaluate(self, weights, cfg, seed: int, max_steps: int | None = None):
        fitness = cfg["fitness"]
        missing = [key for key in FITNESS_WEIGHT_KEYS if key not in fitness]
        if missing:
            raise ValueError(f"fitness configuration is missing: {', '.join(missing)}")
        if len(weights) != self.parameters:
            raise ValueError(
                f"expected {self.parameters} weights for shape {self.shape}, "
                f"got {len(weights)}"
            )
        max_levels = int(cfg.get("max_levels_per_agent", 10))
        if max_levels < 1:
            raise ValueError("max_levels_per_agent must be positive")
        start_level = resolve_start_level(cfg, seed)
        steps = cfg["max_steps"] if max_steps is None else max_steps
        header = ",".join((
            str(start_level), cfg["doom_difficulty"], str(cfg["time_step_seconds"]),
            str(steps), str(max_levels),
            *(str(fitness[key]) for key in FITNESS_WEIGHT_KEYS),
        ))
        line = header + "," + ",".join(map(str, weights))
        started = time.perf_counter()
        try:
            result = self._request(line)
        except (BrokenPipeError, RuntimeError, json.JSONDecodeError):
            # A crashed server is restarted and the episode retried once. This
            # is safe because every job starts from a freshly reset game.
            self.close()
            self._start()
            result = self._request(line)
        wall_seconds = time.perf_counter() - started
        return {
            "fitness": float(result["score"]),
            "levels": int(result["levels"]),
            "kills": int(result["kills"]),
            "health": int(result["health"]),
            "seconds": float(result["seconds"]),
            "start_level": int(result["start_level"]),
            "wall_seconds": wall_seconds,
            "profile": result.get("profile") or {},
        }

    def warm_up(self, cfg):
        """Run one trivial episode so level parsing is done before timing."""
        return self.evaluate([0.0] * self.parameters, cfg, seed=0, max_steps=1)

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            try:
                assert process.stdin
                process.stdin.close()
                process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired, ValueError):
                process.kill()
                process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

