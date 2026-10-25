#!/usr/bin/env python3
"""Train a compact DOOM policy with parallel, from-scratch evolution."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import random
import subprocess
from pathlib import Path

from env import evaluate

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = 10 * (42 + 1)


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
    parser.add_argument("--generations", type=int, help="override configured generation count")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    if args.generations is not None:
        cfg["generations"] = args.generations
    workers = tester_count(cfg["cpu_threads"])
    if cfg["doom_difficulty"] not in ("easy", "medium", "hard"):
        raise SystemExit("doom_difficulty must be easy, medium, or hard")
    if cfg["campaign_levels"] != 10:
        raise SystemExit("campaign_levels must be 10 for the full DOOM campaign")
    if cfg["evaluation_episodes"] < 1 or cfg["population_size"] < 1 or cfg["generations"] < 1:
        raise SystemExit("evaluation_episodes, population_size, and generations must be positive")
    if not 0 < cfg["time_step_seconds"] <= 0.05 or cfg["max_steps"] < 1:
        raise SystemExit("time_step_seconds must be in (0, 0.05] and max_steps must be positive")
    executable = ROOT / "doom/target/release/doom_headless"
    if args.build:
        subprocess.run([str(ROOT / "exec_mgmt.sh"), "-b"], cwd=ROOT, check=True)
    if not executable.is_file():
        raise SystemExit(f"{executable} is missing. Build with: ./exec_mgmt.sh -b")

    output = (ROOT / cfg["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / cfg["checkpoint_file"]
    rng = random.Random(cfg["seed"])
    if cfg["resume"] and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        population = saved["population"]
        start_generation = saved["generation"] + 1
        rng.setstate(tuple_state(saved["random_state"]))
        print(f"Resuming at generation {start_generation} from {checkpoint_path}")
    else:
        population = [[rng.gauss(0, cfg["initial_weight_sigma"]) for _ in range(PARAMETERS)]
                      for _ in range(cfg["population_size"])]
        start_generation = 0

    metrics_path = output / cfg["metrics_file"]
    exists = metrics_path.exists() and start_generation > 0
    with metrics_path.open("a" if exists else "w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=("generation", "best_fitness", "mean_fitness",
                                "levels", "kills", "health", "seconds", "start_level"))
        if not exists:
            writer.writeheader()
        for generation in range(start_generation, cfg["generations"]):
            episode_count = cfg["evaluation_episodes"]
            seeds = [cfg["seed"] + generation * episode_count + i for i in range(episode_count)]
            jobs = [(genome, executable, cfg, seed)
                    for genome in population for seed in seeds]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                episode_results = list(pool.map(run_job, jobs))
            results = [summarize_episodes(episode_results[i:i + episode_count])
                       for i in range(0, len(episode_results), episode_count)]
            ranked = sorted(zip(results, population), key=lambda item: item[0]["fitness"], reverse=True)
            best, best_genome = ranked[0]
            mean_fitness = sum(result["fitness"] for result, _ in ranked) / len(ranked)
            print(f"generation {generation + 1}/{cfg['generations']} | testers {workers} | "
                  f"best {best['fitness']:.1f} | levels {best['levels']} | kills {best['kills']} | "
                  f"hp {best['health']} | {best['seconds']:.1f}s", flush=True)
            writer.writerow({"generation": generation, "best_fitness": round(best["fitness"], 3),
                             "mean_fitness": round(mean_fitness, 3), **{k: best[k] for k in
                             ("levels", "kills", "health", "seconds", "start_level")}})
            metrics_file.flush()

            elite_count = max(1, min(cfg["elite_count"], cfg["population_size"]))
            next_population = [genome[:] for _, genome in ranked[:elite_count]]
            while len(next_population) < cfg["population_size"]:
                parent = ranked[rng.randrange(min(cfg["parent_pool_size"], len(ranked)))][1]
                next_population.append(mutate(parent, cfg, rng))
            population = next_population
            checkpoint_path.write_text(json.dumps({
                "generation": generation,
                "population": population,
                "best_result": best,
                "best_genome": best_genome,
                "random_state": list_state(rng.getstate()),
            }))
            (output / cfg["best_policy_file"]).write_text(json.dumps({
                "architecture": {"inputs": 42, "outputs": 10, "activation": "tanh-linear"},
                "generation": generation,
                "doom_difficulty": cfg["doom_difficulty"],
                "result": best,
                "weights": best_genome,
            }))


def list_state(value):
    return [value[0], list(value[1]), value[2]]


def tuple_state(value):
    return (value[0], tuple(value[1]), value[2])


if __name__ == "__main__":
    main()
