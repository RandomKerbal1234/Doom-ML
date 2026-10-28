"""Headless adapter for the Rust DOOM simulation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path


class HeadlessDoomEnv:
    def __init__(self, executable: Path, dt: float = 0.05, difficulty: str = "medium"):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.dt = dt
        self.difficulty = difficulty
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
        done = new["dead"] or self.levels_completed >= 10
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
    """Fixed-size normalized state vector suitable for a small policy."""
    x, y, angle = obs["x"], obs["y"], obs["angle"]
    rows = obs["map"]
    values = []
    for ray in range(8):
        direction = angle + ray * math.tau / 8
        distance = 0.0
        for k in range(1, 81):
            distance = k * 0.1
            sx = int(x + math.cos(direction) * distance)
            sy = int(y + math.sin(direction) * distance)
            if sy < 0 or sy >= len(rows) or sx < 0 or sx >= len(rows[sy]) or rows[sy][sx] in "#BM":
                break
        values.append(min(distance / 8.0, 1.0))

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
    return values


def choose_action(weights, obs):
    inputs = encode_observation(obs)
    outputs = []
    width = len(inputs) + 1
    for output in range(10):
        start = output * width
        value = weights[start + len(inputs)]
        for i, item in enumerate(inputs):
            value += weights[start + i] * item
        outputs.append(math.tanh(value))
    action = [outputs[0], outputs[1], outputs[2], outputs[3] > 0, outputs[4] > 0,
              0, outputs[9] > 0]
    available = [i for i, unlocked in enumerate(obs["unlocked"]) if unlocked]
    action[5] = max(available, key=lambda i: outputs[5 + i])
    return action


def evaluate(weights, executable: Path, cfg, seed: int, profile: bool = False):
    import random
    rng = random.Random(seed)
    start_level = rng.randrange(cfg["campaign_levels"])
    fitness = cfg["fitness"]
    command = [
        str(executable), "--evaluate", str(start_level), cfg["doom_difficulty"],
        str(cfg["time_step_seconds"]), str(cfg["max_steps"]),
        str(fitness["level_completion"]), str(fitness["partial_level"]),
        str(fitness["health_per_point"]), str(fitness["kill"]), str(fitness["seconds"]),
    ]
    environment = None
    if profile:
        environment = os.environ.copy()
        environment["DOOM_PROFILE"] = "1"
    started = time.perf_counter()
    result = subprocess.run(command, input=",".join(map(str, weights)), text=True,
                            capture_output=True, check=True, env=environment)
    wall_seconds = time.perf_counter() - started
    score, levels, kills, health, seconds, returned_start_level = result.stdout.strip().split(",")
    stats = {}
    for line in result.stderr.splitlines():
        if line.startswith("PROFILE "):
            stats = json.loads(line[len("PROFILE "):])
            break
    return {"fitness": float(score), "levels": int(levels), "kills": int(kills),
            "health": int(health), "seconds": float(seconds),
            "start_level": int(returned_start_level), "wall_seconds": wall_seconds,
            "profile": stats}
