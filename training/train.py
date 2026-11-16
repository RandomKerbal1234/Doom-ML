#!/usr/bin/env python3
"""Train a compact DOOM policy with parallel, from-scratch evolution."""

from __future__ import annotations

import argparse
import array
import base64
import concurrent.futures
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from env import (
    INPUT_COUNT,
    OUTPUT_COUNT,
    Evaluator,
    FITNESS_WEIGHT_KEYS,
    parameter_count,
    policy_shape,
    resolve_start_level,
    weight_blob,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_LEVELS = 50
PROFILE_ENABLED = os.environ.get("DOOM_PROFILE", "") not in ("", "0")
PROFILE_KEYS = (
    "steps", "levels", "main_ns", "levels_ns", "setup_ns", "policy_ns",
    "sim_ns", "killcount_ns", "loop_other_ns", "total_ns", "level_load_ns",
    "movement_ns", "fire_ns", "update_ns", "enemy_ns", "danger_ns", "pickup_ns",
)
PROFILE_COLUMNS = (
    "generation", "episodes", "steps", "generation_s", "jobs_s", "eval_s",
    "rank_s", "evolve_s", "write_s", "generation_other_s",
    "episode_wall_s", "startup_s", "rust_main_s", "rust_eval_s",
    "rust_levels_s", "setup_s", "policy_s", "sim_s", "killcount_s",
    "loop_other_s", "movement_s", "fire_s", "update_s", "enemy_s",
    "danger_s", "pickup_s", "level_load_s",
)

# These keys control scheduling or bookkeeping, not the experiment itself.
# They are deliberately left out of the resume fingerprint so a run can be
# extended or resumed with a different worker count.
NON_BEHAVIOR_KEYS = frozenset({
    "cpu_threads",
    "generations",
    "resume",
})

DEFAULT_CONFIG = {
    "max_levels_per_agent": 10,
    "manifest_file": "run_manifest.json",
    "profile_file": "profile.csv",
    "profile_summary_file": "profile_summary.json",
    "resume": False,
    "start_level": 0,
    # No hidden layer reproduces the original single-layer linear policy.
    "hidden_layers": [],
    # Stagnation-triggered mutation schedule: widen the search radius when the
    # best fitness has not improved for this many generations.
    "mutation_adapt_patience": 25,
    "mutation_adapt_growth": 1.1,
    "mutation_adapt_max": 3.0,
}

REQUIRED_KEYS = (
    "cpu_threads",
    "generations",
    "population_size",
    "elite_count",
    "parent_pool_size",
    "initial_weight_sigma",
    "mutation_rate",
    "mutation_sigma",
    "time_step_seconds",
    "max_steps",
    "evaluation_episodes",
    "doom_difficulty",
    "campaign_levels",
    "seed",
    "resume",
    "output_directory",
    "checkpoint_file",
    "best_policy_file",
    "metrics_file",
    "manifest_file",
    "max_levels_per_agent",
    "fitness",
)

ALLOWED_KEYS = frozenset(REQUIRED_KEYS) | frozenset(DEFAULT_CONFIG)
METRICS_FIELDS = (
    "run_id",
    "generation",
    "best_fitness",
    "mean_fitness",
    "levels",
    "kills",
    "health",
    "seconds",
    "start_level",
)


def tester_count(value):
    if hasattr(os, "process_cpu_count"):
        available = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0)) or 1
    else:
        available = os.cpu_count() or 1
    if value == 0:
        return available
    if value == -1:
        return max(1, available - 1)
    if value < -1:
        raise ValueError("cpu_threads must be -1, 0, or a positive number")
    return value


def validate_config(cfg):
    """Validate the effective configuration and fill in safe defaults."""
    cfg = dict(cfg)
    for key, value in DEFAULT_CONFIG.items():
        cfg.setdefault(key, value)

    unknown = sorted(set(cfg) - ALLOWED_KEYS)
    if unknown:
        raise SystemExit(f"unknown configuration keys: {', '.join(unknown)}")
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise SystemExit(f"missing configuration keys: {', '.join(missing)}")

    def require_int(key, minimum=None, maximum=None):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"{key} must be an integer")
        if minimum is not None and value < minimum:
            raise SystemExit(f"{key} must be at least {minimum}")
        if maximum is not None and value > maximum:
            raise SystemExit(f"{key} must be at most {maximum}")

    def require_number(key, *, minimum=None, maximum=None, exclusive_minimum=None):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"{key} must be a number")
        if not math.isfinite(value):
            raise SystemExit(f"{key} must be finite")
        if minimum is not None and value < minimum:
            raise SystemExit(f"{key} must be at least {minimum}")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            raise SystemExit(f"{key} must be greater than {exclusive_minimum}")
        if maximum is not None and value > maximum:
            raise SystemExit(f"{key} must be at most {maximum}")

    def require_string(key):
        value = cfg[key]
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{key} must be a non-empty string")

    require_int("cpu_threads", minimum=-1)
    require_int("generations", minimum=1)
    require_int("population_size", minimum=1)
    require_int("elite_count", minimum=1)
    require_int("parent_pool_size", minimum=1)
    require_number("initial_weight_sigma", minimum=0.0)
    require_number("mutation_rate", minimum=0.0, maximum=1.0)
    require_number("mutation_sigma", minimum=0.0)
    require_number("time_step_seconds", exclusive_minimum=0.0, maximum=0.05)
    require_int("max_steps", minimum=1)
    require_int("evaluation_episodes", minimum=1)
    require_int("campaign_levels", minimum=1)
    require_int("seed")
    require_int("max_levels_per_agent", minimum=1)

    def require_hidden_layers():
        value = cfg["hidden_layers"]
        if not isinstance(value, (list, tuple)):
            raise SystemExit("hidden_layers must be an array of layer widths")
        layers = list(value)
        if len(layers) > 8:
            raise SystemExit("hidden_layers supports at most 8 hidden layers")
        for width in layers:
            if isinstance(width, bool) or not isinstance(width, int):
                raise SystemExit("hidden_layers must contain integers")
            if not 1 <= width <= 128:
                raise SystemExit("hidden_layers widths must be in 1..128")
        cfg["hidden_layers"] = layers

    require_hidden_layers()
    require_int("mutation_adapt_patience", minimum=0)
    require_number("mutation_adapt_growth", minimum=1.0)
    require_number("mutation_adapt_max", minimum=1.0)

    def require_start_level(levels):
        value = cfg["start_level"]
        if isinstance(value, (list, tuple)):
            if len(value) != 2 or any(
                isinstance(bound, bool) or not isinstance(bound, int) for bound in value
            ):
                raise SystemExit(
                    "start_level must be a two-element integer array [low, high]"
                )
            low, high = value
        elif isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit("start_level must be a [low, high] array or an integer")
        elif value == 0:
            low, high = 1, levels
        else:
            low = high = value
        if not 1 <= low <= high <= levels:
            raise SystemExit(
                f"start_level must satisfy 1 <= low <= high <= {levels}"
            )

    require_start_level(cfg["campaign_levels"])

    if cfg["elite_count"] > cfg["population_size"]:
        raise SystemExit("elite_count must not exceed population_size")
    if cfg["parent_pool_size"] > cfg["population_size"]:
        raise SystemExit("parent_pool_size must not exceed population_size")
    if cfg["campaign_levels"] != CAMPAIGN_LEVELS:
        raise SystemExit(f"campaign_levels must be {CAMPAIGN_LEVELS} for the full DOOM campaign")
    if cfg["max_levels_per_agent"] > cfg["campaign_levels"]:
        raise SystemExit("max_levels_per_agent must not exceed campaign_levels")
    if not isinstance(cfg["doom_difficulty"], str):
        raise SystemExit("doom_difficulty must be easy, medium, or hard")
    cfg["doom_difficulty"] = cfg["doom_difficulty"].lower()
    if cfg["doom_difficulty"] not in ("easy", "medium", "hard"):
        raise SystemExit("doom_difficulty must be easy, medium, or hard")
    if not isinstance(cfg["resume"], bool):
        raise SystemExit("resume must be true or false")
    for key in ("output_directory", "checkpoint_file", "best_policy_file",
                "metrics_file", "manifest_file", "profile_file",
                "profile_summary_file"):
        require_string(key)

    if not isinstance(cfg["fitness"], dict):
        raise SystemExit("fitness must be an object")
    unknown_fitness = sorted(set(cfg["fitness"]) - set(FITNESS_WEIGHT_KEYS))
    if unknown_fitness:
        raise SystemExit(f"unknown fitness keys: {', '.join(unknown_fitness)}")
    missing_fitness = [key for key in FITNESS_WEIGHT_KEYS if key not in cfg["fitness"]]
    if missing_fitness:
        raise SystemExit(f"missing fitness keys: {', '.join(missing_fitness)}")
    for key in FITNESS_WEIGHT_KEYS:
        value = cfg["fitness"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"fitness.{key} must be a number")
        if not math.isfinite(value):
            raise SystemExit(f"fitness.{key} must be finite")
    return cfg


def canonical_config(cfg):
    return {key: value for key, value in cfg.items() if key not in NON_BEHAVIOR_KEYS}


def compute_config_hash(cfg):
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_compatibility_hash(cfg):
    payload = json.dumps(canonical_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_run_id(seed, config_hash):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{seed}-{config_hash[:12]}-{stamp}"


def write_json(path, value, compact=False, raw=None):
    """Write JSON, optionally splicing pre-encoded values in afterwards.

    `raw` maps a key to an already-serialized JSON string. It exists for the
    checkpoint's packed population blob: pushing that one ~900 KB base64 string
    through `json.dumps` only makes the encoder scan it for escapes, which
    costs more than serializing the other two hundred keys combined.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, separators=(",", ":"))
    else:
        text = json.dumps(value, indent=2, sort_keys=True)
    if raw:
        for key, encoded in raw.items():
            placeholder = json.dumps(key) + (':' if compact else ': ')
            empty = placeholder + '""'
            if empty not in text:
                raise SystemExit(f"placeholder for {key} is missing from {path.name}")
            text = text.replace(empty, placeholder + encoded, 1)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_run_artifacts(checkpoint_path, checkpoint, population, manifest_path, manifest,
                        policy_path, policy, stats):
    """Persist one generation's checkpoint, manifest, and policy.

    Serializing that state costs ~7 ms of interpreter time and used to sit
    between two evaluation phases while every simulator process was idle. It
    only depends on data that is final once the next population exists, so the
    trainer hands it to a thread and starts the next generation's evaluation
    immediately. The file write itself releases the GIL, so the overlap costs
    the workers almost nothing.
    """
    started = time.perf_counter()
    blob = pack_floats(population)
    write_json(checkpoint_path, checkpoint, compact=True,
               raw={"population_blob": '"' + blob + '"'})
    write_json(manifest_path, manifest)
    write_json(policy_path, policy)
    stats["write_s"] = time.perf_counter() - started


def utilisation_note(cfg, row, workers):
    """Diagnose a generation whose single-threaded phases dominate the clock.

    Returns a one-off hint when the serial floor is large enough to matter.
    Nothing here can be parallelised without changing the RNG stream or the
    synchronous generation model, so the fix is to give the evaluators more
    episodes per generation.
    """
    total = row["generation_s"]
    parallel = row["eval_s"]
    if total <= 0 or parallel <= 0:
        return None
    serial = max(0.0, total - parallel)
    if serial <= 0.15 * total:
        return None
    episodes = max(1, cfg["evaluation_episodes"])
    needed = math.ceil(9.0 * serial * episodes / parallel)
    ceiling = 100.0 * parallel / total
    note = (
        f"note: {100 * serial / total:.0f}% of each generation runs on one thread "
        f"(~{serial * 1e3:.1f} ms of ranking, mutation, and artifact writes) while the "
        f"{workers} evaluators\n"
        f"      wait, so worker utilisation tops out around {ceiling:.0f}%. "
    )
    if needed > episodes:
        note += (
            f"Raising evaluation_episodes to {needed} would put that\n"
            f"      floor under 10% of the generation; growing population_size has "
            f"the same effect."
        )
    else:
        note += ("Giving the evaluators more episodes per generation would "
                 "push it further down.")
    return note


_UTILISATION_ROWS_SEEN = 0
_UTILISATION_NOTE_SHOWN = False


def emit_profile_row(profiler, profile_file, profile_rows, row, cfg, workers):
    """Record one generation's timing row (plus the opt-in human summary)."""
    global _UTILISATION_ROWS_SEEN, _UTILISATION_NOTE_SHOWN
    profile_rows.append(row)
    profiler.writerow(row)
    profile_file.flush()
    _UTILISATION_ROWS_SEEN += 1
    # Skip the first row: it includes one-off startup and warm-up effects.
    if _UTILISATION_ROWS_SEEN > 1 and not _UTILISATION_NOTE_SHOWN:
        note = utilisation_note(cfg, row, workers)
        if note is not None:
            _UTILISATION_NOTE_SHOWN = True
            print(note, flush=True)
    if PROFILE_ENABLED:
        measured = row["episode_wall_s"] or 1.0
        print(
            f"  profile | gen {row['generation_s']:.2f}s "
            f"(eval {row['eval_s']:.2f}s) | episode wall {measured:.2f}s: "
            f"startup {row['startup_s']:.2f}s "
            f"({100 * row['startup_s'] / measured:.1f}%), "
            f"rust {row['rust_main_s']:.2f}s "
            f"(policy {row['policy_s']:.2f}s, sim {row['sim_s']:.2f}s, "
            f"killscan {row['killcount_s']:.2f}s)",
            flush=True,
        )
        rust_measured = row["rust_main_s"] or 1.0
        print(
            f"          sim | update {row['update_s']:.2f}s "
            f"({100 * row['update_s'] / rust_measured:.1f}% of rust: "
            f"enemy {row['enemy_s']:.2f}s, danger {row['danger_s']:.2f}s, "
            f"pickup {row['pickup_s']:.2f}s) | movement "
            f"{row['movement_s']:.2f}s | fire {row['fire_s']:.2f}s",
            flush=True,
        )


def git_metadata():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def rust_version():
    try:
        return subprocess.run(
            ["rustc", "--version"], text=True, capture_output=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_executable():
    release = ROOT / "doom" / "target" / "release"
    names = ("doom_headless.exe", "doom_headless") if os.name == "nt" else (
        "doom_headless", "doom_headless.exe")
    for name in names:
        candidate = release / name
        if candidate.is_file():
            return candidate
    return release / names[0]


def expected_start_level(cfg, generation):
    seed = cfg["seed"] + generation * cfg["evaluation_episodes"]
    return resolve_start_level(cfg, seed)


def pack_floats(rows):
    """Pack a matrix of floats as base64 little-endian f64.

    The checkpoint rewrites the whole population every generation. Encoded as a
    JSON array of 64x1266 floats that is ~1.75 MB and costs ~30 ms of encoding,
    all of it while every evaluator process sits idle. The same values packed
    with `struct` take ~1.5 ms, are half the size, and stay bit-exact, so an
    interrupted run resumes on identical values.
    """
    flat = struct.pack("<%dd" % sum(len(row) for row in rows),
                       *itertools.chain.from_iterable(rows))
    return base64.b64encode(flat).decode("ascii")


def unpack_floats(text, rows, columns):
    data = array.array("d")
    data.frombytes(base64.b64decode(text))
    if sys.byteorder != "little":
        data.byteswap()
    if len(data) != rows * columns:
        raise SystemExit(
            f"checkpoint holds {len(data)} weights, expected {rows}x{columns}"
        )
    return [data[index * columns:(index + 1) * columns].tolist()
            for index in range(rows)]


def checkpoint_population(saved):
    """Population from either checkpoint layout: packed blob or float lists."""
    if "population_blob" in saved:
        rows, columns = saved["population_shape"]
        return unpack_floats(saved["population_blob"], int(rows), int(columns))
    population = saved.get("population")
    if not isinstance(population, list) or not population:
        raise SystemExit("checkpoint population is missing")
    return population


def checkpoint_manifest(saved):
    required = ("run_id", "seed", "config_hash", "config", "executable_sha256",
                "generation", "random_state")
    missing = [key for key in required if key not in saved]
    if missing:
        raise SystemExit(
            "checkpoint is missing provenance metadata "
            f"({', '.join(missing)}); start a new run with --fresh"
        )
    if "population" not in saved and "population_blob" not in saved:
        raise SystemExit("checkpoint population is missing")
    return {
        "run_id": saved["run_id"],
        "seed": saved["seed"],
        "config_hash": saved["config_hash"],
        "compatibility_hash": saved.get("compatibility_hash", saved["config_hash"]),
        "config": saved["config"],
        "executable": saved.get("executable"),
        "executable_sha256": saved["executable_sha256"],
        "created_at": saved.get("created_at"),
    }


def validate_checkpoint(saved, cfg, compatibility_hash, executable_hash):
    metadata = checkpoint_manifest(saved)
    if not isinstance(metadata["run_id"], str) or not metadata["run_id"]:
        raise SystemExit("checkpoint has an invalid run_id")
    if metadata["seed"] != cfg["seed"]:
        raise SystemExit(
            f"checkpoint seed {metadata['seed']} does not match configured seed {cfg['seed']}"
        )
    if metadata["compatibility_hash"] != compatibility_hash:
        raise SystemExit(
            "checkpoint configuration does not match the current configuration; "
            "use --fresh to start a new run"
        )
    if metadata["executable_sha256"] != executable_hash:
        raise SystemExit(
            "checkpoint executable fingerprint does not match the current binary; "
            "rebuild or use --fresh"
        )
    if not isinstance(saved["generation"], int) or saved["generation"] < 0:
        raise SystemExit("checkpoint generation is invalid")
    if "population" not in saved and "population_blob" not in saved:
        raise SystemExit("checkpoint population is missing")
    return metadata


def validate_manifest(manifest, run_id, cfg, compatibility_hash, executable_hash):
    saved_compatibility = manifest.get("compatibility_hash", manifest.get("config_hash"))
    checks = (
        ("run_id", manifest.get("run_id"), run_id),
        ("seed", manifest.get("seed"), cfg["seed"]),
        ("compatibility_hash", saved_compatibility, compatibility_hash),
        ("executable_sha256", manifest.get("executable_sha256"), executable_hash),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise SystemExit(
                f"run manifest {field} does not match the current run; "
                "use --fresh to start a new run"
            )


def validate_metrics(path, cfg, run_id, checkpoint_generation):
    if not path.is_file():
        raise SystemExit(f"cannot resume: metrics file {path} is missing")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or ()
        if "run_id" not in fields or "generation" not in fields or "start_level" not in fields:
            raise SystemExit(
                f"cannot resume: {path} was written by an older format; use --fresh"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit(f"cannot resume: {path} contains no generations")
    try:
        first_generation = int(rows[0]["generation"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot resume: malformed first metrics row: {error}") from error
    if first_generation != 0:
        raise SystemExit("cannot resume: metrics do not start at generation 0")
    for expected_generation, row in enumerate(rows):
        try:
            row_run_id = row["run_id"]
            generation = int(row["generation"])
            start_level = int(row["start_level"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                f"cannot resume: malformed metrics row {expected_generation}: {error}"
            ) from error
        if row_run_id != run_id:
            raise SystemExit("cannot resume: metrics contain rows from another run")
        if generation != expected_generation:
            raise SystemExit(
                f"cannot resume: metrics jump from generation {expected_generation - 1} "
                f"to {generation}"
            )
        expected_level = expected_start_level(cfg, generation)
        if start_level != expected_level:
            raise SystemExit(
                f"cannot resume: generation {generation} start level does not match "
                "the configured seed"
            )
    try:
        last_generation = int(rows[-1]["generation"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot resume: malformed final metrics row: {error}") from error
    if last_generation != checkpoint_generation:
        raise SystemExit(
            "cannot resume: metrics and checkpoint end at different generations"
        )


def initial_genome(shape, cfg, rng):
    """Fan-in scaled initialization.

    `initial_weight_sigma` is the target standard deviation of a neuron's
    pre-activation, so each layer's weights are divided by `sqrt(fan_in)`. A
    fixed per-weight sigma would saturate `tanh` in the wide input layer while
    leaving narrow deeper layers nearly linear.
    """
    genome = []
    sigma = cfg["initial_weight_sigma"]
    for fan_in, fan_out in zip(shape, shape[1:]):
        scale = sigma / math.sqrt(fan_in)
        genome.extend(rng.gauss(0.0, scale) for _ in range(fan_out * (fan_in + 1)))
    return genome


def mutation_scales(shape, cfg):
    """Per-gene mutation sigma, fan-in normalized like the initialization."""
    sigma = cfg["mutation_sigma"]
    scales = []
    for fan_in, fan_out in zip(shape, shape[1:]):
        scales.extend([sigma / math.sqrt(fan_in)] * (fan_out * (fan_in + 1)))
    return scales


def mutate(parent, scales, cfg, rng, radius=1.0):
    """Mutate one genome.

    The per-gene `random()` then `gauss()` sequence is deliberately unchanged
    from the earlier implementation: the trainer's RNG stream is part of the
    checkpoint, so altering the draw order would silently break resume. Only
    attribute lookups are hoisted.
    """
    rate = cfg["mutation_rate"]
    child = parent[:]
    draw = rng.random
    gauss = rng.gauss
    for index, scale in enumerate(scales):
        if draw() < rate:
            child[index] += gauss(0.0, scale * radius)
    return child


_WORKER = threading.local()
_EVALUATORS = []
_EVALUATORS_LOCK = threading.Lock()


def initialize_worker(executable, cfg):
    """Give every thread its own long-lived evaluator process.

    The Rust evaluator owns all of the simulated state, so the Python side
    only forwards a weight vector and reads one JSON result. Threads therefore
    spend their time blocked on a pipe instead of holding the GIL, and we avoid
    re-spawning the simulator for each of the thousands of episodes per
    generation.
    """
    evaluator = Evaluator(executable, shape=policy_shape(cfg), profile=PROFILE_ENABLED)
    evaluator.warm_up(cfg)
    _WORKER.cfg = cfg
    _WORKER.evaluator = evaluator
    with _EVALUATORS_LOCK:
        _EVALUATORS.append(evaluator)


def close_evaluators():
    with _EVALUATORS_LOCK:
        evaluators, _EVALUATORS[:] = list(_EVALUATORS), []
    for evaluator in evaluators:
        evaluator.close()


def run_job(job):
    """Run one episode. The caller passes a prebuilt weight blob.

    Keeping the blob out of this function means a worker never formats floats:
    it writes bytes, flushes, and blocks on a line of JSON. That is what lets
    the threads overlap with the evaluator processes instead of serializing on
    the interpreter lock.
    """
    blob, seed = job
    return _WORKER.evaluator.evaluate(None, _WORKER.cfg, seed, blob=blob)


def make_pool(executable, cfg, workers):
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="doom-eval",
        initializer=initialize_worker,
        initargs=(str(executable), cfg),
    )


def profile_totals(episodes):
    """Sum Rust-side timings across every episode in one generation."""
    totals = {key: 0 for key in PROFILE_KEYS}
    for episode in episodes:
        stats = episode.get("profile") or {}
        for key in PROFILE_KEYS:
            totals[key] += stats.get(key, 0)
    return totals


def nanoseconds(value):
    return value / 1_000_000_000.0


def summarize_episodes(episodes):
    return {
        "fitness": sum(run["fitness"] for run in episodes) / len(episodes),
        "levels": sum(run["levels"] for run in episodes) / len(episodes),
        "kills": sum(run["kills"] for run in episodes) / len(episodes),
        "health": sum(run["health"] for run in episodes) / len(episodes),
        "seconds": sum(run["seconds"] for run in episodes) / len(episodes),
        "start_level": episodes[0]["start_level"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--build", action="store_true", help="build the release binaries first")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start a new run")
    parser.add_argument("--generations", type=int, help="override configured generation count")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    if args.generations is not None:
        cfg["generations"] = args.generations
    if args.fresh:
        cfg["resume"] = False
    cfg = validate_config(cfg)
    workers = tester_count(cfg["cpu_threads"])
    # Every worker owns a simulator process, so more workers than episodes just
    # pays process startup and level parsing for threads that never run a job.
    total_jobs = cfg["population_size"] * cfg["evaluation_episodes"]
    workers = max(1, min(workers, total_jobs))

    print("Effective configuration:")
    print(json.dumps(cfg, indent=2, sort_keys=True))

    if args.build:
        subprocess.run([str(ROOT / "exec_mgmt.sh"), "-b"], cwd=ROOT, check=True)
    executable = resolve_executable()
    if not executable.is_file():
        raise SystemExit(f"{executable} is missing. Build with: ./exec_mgmt.sh -b")

    executable_hash = sha256_file(executable)
    config_hash = compute_config_hash(cfg)
    compatibility_hash = compute_compatibility_hash(cfg)
    shape = policy_shape(cfg)
    parameters = parameter_count(shape)
    commit, dirty = git_metadata()
    rust = rust_version()

    output = (ROOT / cfg["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / cfg["checkpoint_file"]
    manifest_path = output / cfg["manifest_file"]
    metrics_path = output / cfg["metrics_file"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    saved = None
    loaded_checkpoint = False
    if cfg["resume"] and checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"checkpoint is not valid JSON: {error}") from error
        checkpoint_metadata = validate_checkpoint(saved, cfg, compatibility_hash, executable_hash)
        run_id = checkpoint_metadata["run_id"]
        start_generation = saved["generation"] + 1
        population = checkpoint_population(saved)
        for genome in population:
            if len(genome) != parameters:
                raise SystemExit(
                    "checkpoint genomes do not match the configured policy shape; "
                    "use --fresh to start a new run"
                )
        mutation_radius = float(saved.get("mutation_radius", 1.0))
        best_ever = saved.get("best_ever_fitness")
        stale_generations = int(saved.get("stale_generations", 0))
        rng = random.Random()
        rng.setstate(tuple_state(saved["random_state"]))
        created_at = checkpoint_metadata["created_at"] or utc_now()
        loaded_checkpoint = True
        print(f"Resuming at generation {start_generation} from {checkpoint_path}")
        validate_metrics(metrics_path, cfg, run_id, saved["generation"])
    else:
        if cfg["resume"]:
            print("resume is enabled but no checkpoint exists; starting a fresh run")
        else:
            print("Starting a fresh run")
        run_id = make_run_id(cfg["seed"], config_hash)
        rng = random.Random(cfg["seed"])
        population = [initial_genome(shape, cfg, rng) for _ in range(cfg["population_size"])]
        mutation_radius = 1.0
        best_ever = None
        stale_generations = 0
        start_generation = 0
        created_at = utc_now()

    manifest = {
        "run_id": run_id,
        "seed": cfg["seed"],
        "config_hash": config_hash,
        "compatibility_hash": compatibility_hash,
        "config": cfg,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "rust_version": rust,
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "created_at": created_at,
        "updated_at": utc_now(),
        "resume": {
            "enabled": cfg["resume"],
            "loaded_checkpoint": loaded_checkpoint,
            "checkpoint_generation": start_generation - 1 if loaded_checkpoint else None,
        },
        "checkpoint_generation": start_generation - 1 if loaded_checkpoint else None,
        "status": "running",
    }
    if loaded_checkpoint and manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"run manifest is not valid JSON: {error}") from error
        validate_manifest(existing_manifest, run_id, cfg, compatibility_hash, executable_hash)
        manifest["created_at"] = existing_manifest.get("created_at", created_at)
    elif loaded_checkpoint:
        print(f"Run manifest {manifest_path} was missing; recreating it")
    write_json(manifest_path, manifest)

    print(f"Run ID: {run_id}")
    print(f"Output directory: {output}")
    print(f"Seed: {cfg['seed']}")
    print(f"Policy shape: {list(shape)} ({parameters} parameters)")
    print(f"Configuration hash: {config_hash}")
    print(f"Resume fingerprint: {compatibility_hash}")
    print(f"Checkpoint loaded: {'yes' if loaded_checkpoint else 'no'}")

    if loaded_checkpoint:
        metrics_mode = "a"
    else:
        metrics_mode = "w"
    profile_path = output / cfg["profile_file"]
    profile_mode = "a" if loaded_checkpoint and profile_path.exists() else "w"
    profile_rows = []
    main_start = time.perf_counter()
    scales = mutation_scales(shape, cfg)
    pool = make_pool(executable, cfg, workers)
    try:
        with metrics_path.open(metrics_mode, newline="") as metrics_file, \
                profile_path.open(profile_mode, newline="") as profile_file:
            writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS)
            if not loaded_checkpoint:
                writer.writeheader()
            profiler = csv.DictWriter(profile_file, fieldnames=PROFILE_COLUMNS)
            if profile_mode == "w":
                profiler.writeheader()
            # (thread, launch time, profile row) of the artifact write that is
            # currently overlapping this generation's evaluation.
            pending_write = None
            for generation in range(start_generation, cfg["generations"]):
                generation_start = time.perf_counter()
                episode_count = cfg["evaluation_episodes"]
                seeds = [cfg["seed"] + generation * episode_count + i
                         for i in range(episode_count)]
                expected_levels = [
                    resolve_start_level(cfg, seed) for seed in seeds
                ]
                # Pack each genome once per generation instead of once per job.
                # The trainer holds the GIL while packing, so doing it per job
                # (population x episodes times) serializes the workers and
                # leaves the evaluator processes idle.
                blobs = [weight_blob(genome) for genome in population]
                jobs = [(blob, seed) for blob in blobs for seed in seeds]
                jobs_done = time.perf_counter()
                episode_results = list(pool.map(run_job, jobs))
                evaluation_done = time.perf_counter()
                # Summed job latency: with `workers` threads this is the wall
                # time they were occupied, so it doubles as the answer to "are
                # the cores actually busy?".
                episode_wall = sum(run["wall_seconds"] for run in episode_results)
                for index, result in enumerate(episode_results):
                    expected_level = expected_levels[index % episode_count]
                    if result["start_level"] != expected_level:
                        raise SystemExit(
                            "evaluation seed produced an unexpected start level: "
                            f"generation {generation}, expected {expected_level}, "
                            f"got {result['start_level']}"
                        )
                results = [summarize_episodes(episode_results[i:i + episode_count])
                           for i in range(0, len(episode_results), episode_count)]
                ranked = sorted(zip(results, population),
                                key=lambda item: item[0]["fitness"], reverse=True)
                ranked_done = time.perf_counter()
                best, best_genome = ranked[0]
                mean_fitness = sum(result["fitness"] for result, _ in ranked) / len(ranked)
                progressed = best_ever is None or best["fitness"] > best_ever + 1e-9
                if progressed:
                    best_ever = best["fitness"]
                    stale_generations = 0
                    mutation_radius = 1.0
                else:
                    stale_generations += 1
                    patience = cfg["mutation_adapt_patience"]
                    if patience and stale_generations >= patience:
                        mutation_radius = min(
                            cfg["mutation_adapt_max"],
                            mutation_radius * cfg["mutation_adapt_growth"],
                        )
                        stale_generations = 0
                radius_note = "" if mutation_radius == 1.0 else f" | radius {mutation_radius:.2f}"
                eval_span = max(evaluation_done - jobs_done, 1e-9)
                busy = min(100.0, 100.0 * episode_wall / (eval_span * workers))
                print(f"generation {generation + 1}/{cfg['generations']} | testers {workers} | "
                      f"best {best['fitness']:.1f} | levels {best['levels']} | "
                      f"kills {best['kills']} | hp {best['health']} | "
                      f"{best['seconds']:.1f}s | eval busy {busy:.0f}%{radius_note}",
                      flush=True)
                writer.writerow({"run_id": run_id, "generation": generation,
                                 "best_fitness": round(best["fitness"], 3),
                                 "mean_fitness": round(mean_fitness, 3),
                                 **{k: best[k] for k in
                                    ("levels", "kills", "health", "seconds", "start_level")}})
                metrics_file.flush()

                elite_count = max(1, min(cfg["elite_count"], cfg["population_size"]))
                next_population = [genome[:] for _, genome in ranked[:elite_count]]
                while len(next_population) < cfg["population_size"]:
                    parent = ranked[rng.randrange(min(cfg["parent_pool_size"], len(ranked)))][1]
                    next_population.append(mutate(parent, scales, cfg, rng, mutation_radius))
                population = next_population
                evolution_done = time.perf_counter()
                checkpoint = {
                    "run_id": run_id,
                    "seed": cfg["seed"],
                    "config_hash": config_hash,
                    "compatibility_hash": compatibility_hash,
                    "config": cfg,
                    "executable": str(executable),
                    "executable_sha256": executable_hash,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "python_version": platform.python_version(),
                    "rust_version": rust,
                    "start_time": manifest["created_at"],
                    "resume_state": manifest["resume"],
                    "generation": generation,
                    "evaluation_seed_offset": generation * episode_count,
                    "next_evaluation_seed": cfg["seed"] + (generation + 1) * episode_count,
                    "population_encoding": "base64-f64-le",
                    "population_shape": [len(population), parameters],
                    "population_blob": "",
                    "best_result": best,
                    "best_genome": best_genome,
                    "mutation_radius": mutation_radius,
                    "best_ever_fitness": best_ever,
                    "stale_generations": stale_generations,
                    "random_state": list_state(rng.getstate()),
                    "created_at": manifest["created_at"],
                    "updated_at": utc_now(),
                }
                manifest["checkpoint_generation"] = generation
                manifest["updated_at"] = checkpoint["updated_at"]
                policy_payload = {
                    "architecture": {
                        "inputs": INPUT_COUNT,
                        "hidden_layers": list(cfg["hidden_layers"]),
                        "outputs": OUTPUT_COUNT,
                        "activation": "tanh",
                        "layout": "layer-major; per neuron fan_in inputs then bias",
                    },
                    "policy_shape": ",".join(str(width) for width in shape),
                    "run_id": run_id,
                    "seed": cfg["seed"],
                    "config_hash": config_hash,
                    "compatibility_hash": compatibility_hash,
                    "generation": generation,
                    "doom_difficulty": cfg["doom_difficulty"],
                    "result": best,
                    "weights": best_genome,
                }
                # Join here, not right after launching: that is what makes the
                # overlap work. The previous generation's artifacts are written
                # while *this* generation's episodes run.
                if pending_write is not None:
                    thread, stats, pending_row = pending_write
                    thread.join()
                    pending_row["write_s"] = round(stats.get("write_s", 0.0), 6)
                    emit_profile_row(profiler, profile_file, profile_rows, pending_row,
                                     cfg, workers)
                    pending_write = None
                stats = {}
                thread = threading.Thread(
                    target=write_run_artifacts,
                    args=(checkpoint_path, checkpoint, population, manifest_path,
                          dict(manifest), output / cfg["best_policy_file"],
                          policy_payload, stats),
                    name="artifact-writer",
                    daemon=True,
                )
                write_launch = time.perf_counter()
                thread.start()
                writes_done = write_launch

                totals = profile_totals(episode_results)
                rust_main = nanoseconds(totals["main_ns"])
                row = {
                    "generation": generation,
                    "episodes": len(episode_results),
                    "steps": totals["steps"],
                    "generation_s": round(writes_done - generation_start, 6),
                    "jobs_s": round(jobs_done - generation_start, 6),
                    "eval_s": round(evaluation_done - jobs_done, 6),
                    "rank_s": round(ranked_done - evaluation_done, 6),
                    "evolve_s": round(evolution_done - ranked_done, 6),
                    # Filled in once the overlapped artifact write finishes.
                    "write_s": 0.0,
                    "generation_other_s": round(
                        (writes_done - generation_start)
                        - (evaluation_done - jobs_done)
                        - (ranked_done - evaluation_done)
                        - (evolution_done - ranked_done),
                        6,
                    ),
                    "episode_wall_s": round(episode_wall, 6),
                    "startup_s": round(max(0.0, episode_wall - rust_main), 6),
                    "rust_main_s": round(rust_main, 6),
                    "rust_eval_s": round(nanoseconds(totals["total_ns"]), 6),
                    "rust_levels_s": round(nanoseconds(totals["levels_ns"]), 6),
                    "setup_s": round(nanoseconds(totals["setup_ns"]), 6),
                    "policy_s": round(nanoseconds(totals["policy_ns"]), 6),
                    "sim_s": round(nanoseconds(totals["sim_ns"]), 6),
                    "killcount_s": round(nanoseconds(totals["killcount_ns"]), 6),
                    "loop_other_s": round(nanoseconds(totals["loop_other_ns"]), 6),
                    "movement_s": round(nanoseconds(totals["movement_ns"]), 6),
                    "fire_s": round(nanoseconds(totals["fire_ns"]), 6),
                    "update_s": round(nanoseconds(totals["update_ns"]), 6),
                    "enemy_s": round(nanoseconds(totals["enemy_ns"]), 6),
                    "danger_s": round(nanoseconds(totals["danger_ns"]), 6),
                    "pickup_s": round(nanoseconds(totals["pickup_ns"]), 6),
                    "level_load_s": round(nanoseconds(totals["level_load_ns"]), 6),
                }
                pending_write = (thread, stats, row)
            # The last generation's write is still in flight.
            if pending_write is not None:
                thread, stats, pending_row = pending_write
                thread.join()
                pending_row["write_s"] = round(stats.get("write_s", 0.0), 6)
                emit_profile_row(profiler, profile_file, profile_rows, pending_row,
                                 cfg, workers)
                pending_write = None
    finally:
        pool.shutdown(wait=True)
        close_evaluators()

    manifest["status"] = "completed"
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)

    if profile_rows:
        numeric_keys = PROFILE_COLUMNS[3:]
        totals = {key: sum(row[key] for row in profile_rows) for key in numeric_keys}
        summary = {
            "profile_enabled": PROFILE_ENABLED,
            "generations": len(profile_rows),
            "wall_seconds": round(time.perf_counter() - main_start, 6),
            "totals": {key: round(value, 6) for key, value in totals.items()},
            "per_generation_mean": {
                key: round(value / len(profile_rows), 6) for key, value in totals.items()
            },
            "percent_of_episode_wall": {
                key: round(100 * totals[key] / max(totals["episode_wall_s"], 1e-9), 3)
                for key in numeric_keys if key != "episode_wall_s"
            },
        }
        write_json(output / cfg["profile_summary_file"], summary)
        if PROFILE_ENABLED:
            total_wall = summary["wall_seconds"]
            print(
                f"profile summary | {len(profile_rows)} generations in {total_wall:.2f}s "
                f"| episode CPU {totals['episode_wall_s']:.2f}s "
                f"| startup {totals['startup_s']:.2f}s "
                f"({100 * totals['startup_s'] / max(totals['episode_wall_s'], 1e-9):.1f}%) "
                f"| rust {totals['rust_main_s']:.2f}s "
                f"| policy {totals['policy_s']:.2f}s | sim {totals['sim_s']:.2f}s",
                flush=True,
            )


def list_state(value):
    return [value[0], list(value[1]), value[2]]


def tuple_state(value):
    return (value[0], tuple(value[1]), value[2])


if __name__ == "__main__":
    main()
