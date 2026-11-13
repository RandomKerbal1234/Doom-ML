#!/usr/bin/env python3
"""Train a compact DOOM policy with parallel, from-scratch evolution."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from env import FITNESS_WEIGHT_KEYS, evaluate

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = 10 * (42 + 1)
CAMPAIGN_LEVELS = 50

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
    "resume": False,
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

ALLOWED_KEYS = frozenset(REQUIRED_KEYS)
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
                "metrics_file", "manifest_file"):
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


def write_json(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, separators=(",", ":"))
    else:
        text = json.dumps(value, indent=2, sort_keys=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


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
    return random.Random(seed).randrange(cfg["campaign_levels"])


def checkpoint_manifest(saved):
    required = ("run_id", "seed", "config_hash", "config", "executable_sha256",
                "generation", "population", "random_state")
    missing = [key for key in required if key not in saved]
    if missing:
        raise SystemExit(
            "checkpoint is missing provenance metadata "
            f"({', '.join(missing)}); start a new run with --fresh"
        )
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
    if not isinstance(saved["population"], list) or not saved["population"]:
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


def mutate(parent, cfg, rng):
    sigma = cfg["mutation_sigma"]
    child = parent[:]
    for i in range(len(child)):
        if rng.random() < cfg["mutation_rate"]:
            child[i] += rng.gauss(0, sigma)
    return child


def run_job(job):
    genome, executable, cfg, seed = job
    return evaluate(genome, executable, cfg, seed)


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
        population = saved["population"]
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
        population = [[rng.gauss(0, cfg["initial_weight_sigma"]) for _ in range(PARAMETERS)]
                      for _ in range(cfg["population_size"])]
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
    print(f"Configuration hash: {config_hash}")
    print(f"Resume fingerprint: {compatibility_hash}")
    print(f"Checkpoint loaded: {'yes' if loaded_checkpoint else 'no'}")

    if loaded_checkpoint:
        metrics_mode = "a"
    else:
        metrics_mode = "w"
    with metrics_path.open(metrics_mode, newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS)
        if not loaded_checkpoint:
            writer.writeheader()
        for generation in range(start_generation, cfg["generations"]):
            episode_count = cfg["evaluation_episodes"]
            seeds = [cfg["seed"] + generation * episode_count + i for i in range(episode_count)]
            expected_levels = [
                random.Random(seed).randrange(cfg["campaign_levels"])
                for seed in seeds
            ]
            jobs = [(genome, executable, cfg, seed)
                    for genome in population for seed in seeds]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                episode_results = list(pool.map(run_job, jobs))
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
            ranked = sorted(zip(results, population), key=lambda item: item[0]["fitness"], reverse=True)
            best, best_genome = ranked[0]
            mean_fitness = sum(result["fitness"] for result, _ in ranked) / len(ranked)
            print(f"generation {generation + 1}/{cfg['generations']} | testers {workers} | "
                  f"best {best['fitness']:.1f} | levels {best['levels']} | kills {best['kills']} | "
                  f"hp {best['health']} | {best['seconds']:.1f}s", flush=True)
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
                next_population.append(mutate(parent, cfg, rng))
            population = next_population
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
                "population": population,
                "best_result": best,
                "best_genome": best_genome,
                "random_state": list_state(rng.getstate()),
                "created_at": manifest["created_at"],
                "updated_at": utc_now(),
            }
            write_json(checkpoint_path, checkpoint, compact=True)
            manifest["checkpoint_generation"] = generation
            manifest["updated_at"] = checkpoint["updated_at"]
            write_json(manifest_path, manifest)
            write_json(output / cfg["best_policy_file"], {
                "architecture": {"inputs": 42, "outputs": 10, "activation": "tanh-linear"},
                "run_id": run_id,
                "seed": cfg["seed"],
                "config_hash": config_hash,
                "compatibility_hash": compatibility_hash,
                "generation": generation,
                "doom_difficulty": cfg["doom_difficulty"],
                "result": best,
                "weights": best_genome,
            })

    manifest["status"] = "completed"
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)


def list_state(value):
    return [value[0], list(value[1]), value[2]]


def tuple_state(value):
    return (value[0], tuple(value[1]), value[2])


if __name__ == "__main__":
    main()
