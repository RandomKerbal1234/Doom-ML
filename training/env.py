"""Headless adapter for the Rust DOOM simulation."""

from __future__ import annotations

import collections
import json
import math
import queue
import subprocess
import threading
from pathlib import Path

DEFAULT_STEP_TIMEOUT = 60.0


class HeadlessDoomEnv:
    def __init__(
        self,
        executable: Path,
        dt: float = 0.05,
        difficulty: str = "medium",
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
    ):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.dt = dt
        self.difficulty = difficulty
        self.step_timeout = step_timeout
        self.observation = None
        self.levels_completed = 0
        # A blocked readline() is indistinguishable from a hung simulation, so
        # a daemon reader feeds lines into a queue and _request applies a
        # timeout.  stderr is drained continuously: an undrained pipe can fill
        # up and deadlock the child, which then never answers on stdout.
        self._lines: "queue.Queue" = queue.Queue()
        self._errors = collections.deque(maxlen=20)
        self._broken = False
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_stdout, name="doom-stdout", daemon=True
        )
        self._stderr = threading.Thread(
            target=self._read_stderr, name="doom-stderr", daemon=True
        )
        self._reader.start()
        self._stderr.start()

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            self._lines.put(None)
            return
        try:
            for line in iter(stream.readline, ""):
                self._lines.put(line)
        except (OSError, ValueError) as exc:
            self._errors.append(f"stdout reader failed: {exc}")
        finally:
            self._lines.put(None)

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if line.strip():
                    self._errors.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _diagnostics(self) -> str:
        if not self._errors:
            return ""
        return " Last stderr: " + " | ".join(list(self._errors)[-5:])

    def _request(self, line: str):
        if self._closed:
            raise RuntimeError("the headless environment is closed")
        if self._broken:
            raise RuntimeError("the headless environment is unusable after an earlier failure")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"doom_headless exited unexpectedly ({self.process.returncode})."
                f"{self._diagnostics()}"
            )
        stream = self.process.stdin
        if stream is None:
            raise RuntimeError("doom_headless stdin is closed")
        try:
            stream.write(line + "\n")
            stream.flush()
        except (BrokenPipeError, OSError) as exc:
            self._broken = True
            raise RuntimeError(
                f"could not send a command to doom_headless: {exc}.{self._diagnostics()}"
            ) from exc
        try:
            response = self._lines.get(timeout=self.step_timeout)
        except queue.Empty:
            self._broken = True
            raise RuntimeError(
                f"doom_headless did not answer within {self.step_timeout:g}s "
                f"(command: {line.split(',')[0]}).{self._diagnostics()}"
            ) from None
        if response is None:
            self._broken = True
            raise RuntimeError(
                "doom_headless closed its stdout unexpectedly."
                f"{self._diagnostics()}"
            )
        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"doom_headless returned invalid JSON: {response.strip()!r}"
            ) from exc
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
        if self._closed:
            return
        self._closed = True
        self._broken = True
        process = self.process
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, ValueError):
                pass
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass

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


def choose_action_from_inputs(weights, inputs, unlocked):
    """Linear policy over an already-encoded input vector."""
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
    available = [i for i, flag in enumerate(unlocked) if flag]
    if available:
        action[5] = max(available, key=lambda i: outputs[5 + i])
    return action


def choose_action(weights, obs):
    return choose_action_from_inputs(weights, encode_observation(obs), obs["unlocked"])


def evaluate(weights, executable: Path, cfg, seed: int):
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
    result = subprocess.run(command, input=",".join(map(str, weights)), text=True,
                            capture_output=True, check=True)
    score, levels, kills, health, seconds, returned_start_level = result.stdout.strip().split(",")
    return {"fitness": float(score), "levels": int(levels), "kills": int(kills),
            "health": int(health), "seconds": float(seconds),
            "start_level": int(returned_start_level)}


def evaluate_with_policy_backend(weights, executable: Path, cfg, seed: int, backend):
    """Evaluate one policy while choosing actions with a Python policy backend.

    This mirrors the fitness calculation in the Rust ``--evaluate`` mode.  It
    exists so the optional GPU backend can drive the simulation; it is slower
    than the in-process Rust loop because each step crosses the process
    boundary, which is why the CPU path remains the default.
    """
    import random
    rng = random.Random(seed)
    start_level = rng.randrange(cfg["campaign_levels"])
    fitness = cfg["fitness"]
    gpu_cfg = cfg.get("gpu") if isinstance(cfg.get("gpu"), dict) else {}
    step_timeout = float(
        (gpu_cfg or {}).get("step_timeout", DEFAULT_STEP_TIMEOUT) or DEFAULT_STEP_TIMEOUT
    )
    env = HeadlessDoomEnv(
        Path(executable),
        cfg["time_step_seconds"],
        cfg["doom_difficulty"],
        step_timeout=step_timeout,
    )
    try:
        observation = env.reset(start_level)
        levels = 0
        level_kills = 0
        total_kills = 0
        total_time = 0.0
        level_start_time = 0.0
        score = 0.0
        steps = 0
        while steps < cfg["max_steps"]:
            enemies_before = len(observation["enemies"])
            previous_level = observation["level"]
            action = backend.choose_action(
                weights, encode_observation(observation), observation["unlocked"]
            )
            observation, _ = env.step(action)
            enemies_after = len(observation["enemies"])
            if observation["level"] == previous_level:
                killed = max(0, enemies_before - enemies_after)
            else:
                killed = 0
            level_kills += killed
            total_kills += killed
            steps += 1
            total_time += cfg["time_step_seconds"]
            if observation["level"] != previous_level:
                levels += 1
                score += fitness["level_completion"]
                score += observation["health"] * fitness["health_per_point"]
                score += level_kills * fitness["kill"]
                score -= (total_time - level_start_time) * fitness["seconds"]
                level_kills = 0
                level_start_time = total_time
            if observation["dead"] or levels >= 10:
                break
        if levels < 10:
            score += levels * fitness["partial_level"]
            score += observation["health"] * fitness["health_per_point"]
            score += level_kills * fitness["kill"]
            score -= (total_time - level_start_time) * fitness["seconds"]
        return {
            "fitness": score,
            "levels": levels,
            "kills": total_kills,
            "health": observation["health"],
            "seconds": total_time,
            "start_level": start_level,
        }
    finally:
        env.close()
