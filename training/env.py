"""Headless adapter for the Rust DOOM simulation."""

from __future__ import annotations

import json
import math
import subprocess
from collections import deque
from functools import lru_cache
from pathlib import Path


class HeadlessDoomEnv:
    def __init__(self, executable: Path, dt: float = 0.05, difficulty: str = "medium",
                 campaign_levels: int = 50):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.dt = dt
        self.difficulty = difficulty
        self.campaign_levels = campaign_levels
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
        done = new["dead"] or self.levels_completed >= self.campaign_levels
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


@lru_cache(maxsize=128)
def _exit_distance_field(rows, exit_x, exit_y):
    """Four-way shortest-path distance from each open tile to the exit."""
    height = len(rows)
    width = len(rows[0]) if rows else 0
    distances = [[-1] * width for _ in range(height)]
    if not (0 <= exit_x < width and 0 <= exit_y < height):
        return tuple(tuple(row) for row in distances)
    if rows[exit_y][exit_x] in "#BM":
        return tuple(tuple(row) for row in distances)

    distances[exit_y][exit_x] = 0
    queue = deque([(exit_x, exit_y)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nx < width and 0 <= ny < height and distances[ny][nx] < 0
                    and rows[ny][nx] not in "#BM"):
                distances[ny][nx] = distances[y][x] + 1
                queue.append((nx, ny))
    return tuple(tuple(row) for row in distances)


def exit_path_distance(obs):
    """Return shortest-path distance in tiles from the player to this level's exit."""
    rows = tuple(obs["map"])
    exit_x, exit_y = (int(value) for value in obs["exit"])
    field = _exit_distance_field(rows, exit_x, exit_y)
    x, y = int(obs["x"]), int(obs["y"])
    distance = field[y][x]
    if distance >= 0:
        return float(distance)
    # Malformed or disconnected maps should not crash an evaluation.
    return math.hypot(obs["exit"][0] - obs["x"], obs["exit"][1] - obs["y"])


def total_enemy_health(obs):
    return sum(enemy["hp"] for enemy in obs["enemies"])


def evaluate(weights, executable: Path, cfg, seed: int):
    import random
    rng = random.Random(seed)
    start_level = rng.randrange(cfg["campaign_levels"])
    level_count = 0
    total_kills = 0
    total_time = 0.0
    level_start_time = 0.0
    score = 0.0
    steps = 0
    with HeadlessDoomEnv(executable, cfg["time_step_seconds"], cfg["doom_difficulty"],
                         cfg["campaign_levels"]) as env:
        obs = env.reset(start_level)
        while steps < cfg["max_steps"]:
            previous = obs
            previous_distance = exit_path_distance(previous)
            previous_enemy_health = total_enemy_health(previous)
            action = choose_action(weights, obs)
            previous_level = obs["level"]
            obs, done = env.step(action)
            same_level = obs["level"] == previous_level
            killed = max(0, len(previous["enemies"]) - len(obs["enemies"])) if same_level else 0
            total_kills += killed
            steps += 1
            total_time += cfg["time_step_seconds"]
            if same_level:
                cfg_fitness = cfg["fitness"]
                score += (previous_distance - exit_path_distance(obs)) * cfg_fitness["exit_progress"]
                damage_dealt = max(0.0, previous_enemy_health - total_enemy_health(obs))
                score += damage_dealt * cfg_fitness["damage_dealt"]
                score += killed * cfg_fitness["kill"]

                health_loss = max(0, previous["health"] - obs["health"])
                armor_loss = max(0, previous["armor"] - obs["armor"])
                health_gain = max(0, obs["health"] - previous["health"])
                armor_gain = max(0, obs["armor"] - previous["armor"])
                score -= health_loss * cfg_fitness["damage_taken"]
                score -= armor_loss * cfg_fitness["armor_damage_taken"]
                score += health_gain * cfg_fitness["health_gain"]
                score += armor_gain * cfg_fitness["armor_gain"]

                ammo_spent = max(0, sum(previous["ammo"]) - sum(obs["ammo"]))
                score -= ammo_spent * cfg_fitness["ammo_spent"]
                if obs["dead"] and not previous["dead"]:
                    score -= cfg_fitness["death"]

            if not same_level:
                level_count += 1
                # Level completion remains the strongest signal, with health and time bonuses.
                score += cfg["fitness"]["level_completion"]
                score += obs["health"] * cfg["fitness"]["health_per_point"]
                score -= (total_time - level_start_time) * cfg["fitness"]["seconds"]
                level_start_time = total_time
            if done:
                break
    # Shape incomplete runs by progress and the score from their current level.
    if level_count < cfg["campaign_levels"]:
        score += level_count * cfg["fitness"]["partial_level"]
        score += obs["health"] * cfg["fitness"]["health_per_point"]
        score -= (total_time - level_start_time) * cfg["fitness"]["seconds"]
    return {"fitness": score, "levels": level_count, "kills": total_kills,
            "health": obs["health"], "seconds": round(total_time, 2), "start_level": start_level}
