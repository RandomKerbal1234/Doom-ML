#!/usr/bin/env python3
"""Train a compact DOOM policy with parallel, from-scratch evolution."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import csv
import json
import multiprocessing
import os
import random
import subprocess
import time
from pathlib import Path

import gpu
from env import evaluate, evaluate_with_policy_backend

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = 10 * (42 + 1)
CAMPAIGN_LEVELS = 50
# Below this size, per-step device launch overhead dominates and the in-process
# Rust policy loop is far faster.  ``auto`` therefore stays on the CPU unless
# the GPU is requested explicitly.
GPU_SMALL_POLICY = 100_000
_GPU_BACKEND = None


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


def run_job_gpu(job):
    genome, executable, cfg, seed = job
    try:
        return evaluate_with_policy_backend(genome, executable, cfg, seed, _GPU_BACKEND)
    except Exception as exc:
        # Returning the error keeps the thread pool from propagating it and
        # lets the caller retry the whole generation on the CPU path.
        return exc


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
    global _GPU_BACKEND

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--build", action="store_true", help="build the release binaries first")
    parser.add_argument("--generations", type=int, help="override configured generation count")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", action="store_true",
                           help="use the GPU policy backend when an accelerator is available")
    gpu_group.add_argument("--force-gpu", action="store_true",
                           help="require the GPU policy backend and fail if it is unavailable")
    gpu_group.add_argument("--cpu", action="store_true",
                           help="disable the GPU backend and use the fast Rust path")
    parser.add_argument("--gpu-report", action="store_true",
                        help="print detected GPU, driver, and tuning details, then exit")
    parser.add_argument("--gpu-benchmark", action="store_true",
                        help="compare CPU and GPU policy inference, then exit")
    parser.add_argument("--gpu-backend", choices=("auto", "cuda", "rocm", "mps", "xpu", "cpu"),
                        help="override the configured GPU backend")
    parser.add_argument("--gpu-batch-size", type=int, default=256,
                        help="sample count for --gpu-benchmark (default: 256)")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    if args.generations is not None:
        cfg["generations"] = args.generations
    workers = tester_count(cfg["cpu_threads"])
    if cfg["doom_difficulty"] not in ("easy", "medium", "hard"):
        raise SystemExit("doom_difficulty must be easy, medium, or hard")
    if cfg["campaign_levels"] != CAMPAIGN_LEVELS:
        raise SystemExit(f"campaign_levels must be {CAMPAIGN_LEVELS} for the full DOOM campaign")
    if cfg["evaluation_episodes"] < 1 or cfg["population_size"] < 1 or cfg["generations"] < 1:
        raise SystemExit("evaluation_episodes, population_size, and generations must be positive")
    if not 0 < cfg["time_step_seconds"] <= 0.05 or cfg["max_steps"] < 1:
        raise SystemExit("time_step_seconds must be in (0, 0.05] and max_steps must be positive")

    gpu_cfg = cfg.setdefault("gpu", {})
    if not isinstance(gpu_cfg, dict):
        raise SystemExit("gpu must be a JSON object")
    mode = str(gpu_cfg.get("mode", "off"))
    explicit_gpu = bool(args.gpu or args.force_gpu)
    if args.gpu:
        mode = "auto"
    elif args.force_gpu:
        mode = "force"
        gpu_cfg = dict(gpu_cfg)
        gpu_cfg["fallback_to_cpu"] = False
    elif args.cpu:
        mode = "off"
    if (args.gpu_report or args.gpu_benchmark) and mode == "off":
        mode = "auto"
    plan = gpu.resolve(gpu_cfg, mode=mode, requested_backend=args.gpu_backend)
    if (
        mode == "auto"
        and not explicit_gpu
        and not args.gpu_report
        and not args.gpu_benchmark
        and plan.use_gpu
        and PARAMETERS <= GPU_SMALL_POLICY
    ):
        # "auto" should pick the faster path, not merely the available one.
        # Driving the simulation from Python makes every step pay a host/device
        # round trip, which is far slower than the in-process Rust loop.
        name = getattr(plan.selected, "name", None) or "A GPU"
        plan.use_gpu = False
        plan.device = "cpu"
        plan.dtype = "float32"
        plan.batch_size = 1
        plan.reason = (
            f"{name} was detected, but this {PARAMETERS}-weight policy is much faster "
            "on the CPU path. Pass --gpu (or set gpu.mode to force) to use the GPU "
            "anyway."
        )
    if gpu_cfg.get("report", True) or args.gpu_report or args.gpu_benchmark:
        print(plan.report(), flush=True)
    if args.gpu_report:
        return 0 if not plan.error else 1
    if args.gpu_benchmark:
        from gpu_backend import benchmark

        result = benchmark(plan, batch_size=args.gpu_batch_size)
        print(json.dumps(result, indent=2))
        return 0 if not result.get("error") else 1
    if plan.error and mode == "force" and not gpu_cfg.get("fallback_to_cpu", True):
        raise SystemExit(plan.error)
    fallback_allowed = bool(gpu_cfg.get("fallback_to_cpu", True))
    if plan.use_gpu:
        from gpu_backend import TorchPolicyBackend

        if PARAMETERS <= GPU_SMALL_POLICY:
            print(
                f"Warning: the GPU path drives one simulation step at a time from "
                f"Python, so a {PARAMETERS}-weight policy is typically far slower "
                "than the CPU. Progress is reported during each generation.",
                flush=True,
            )
        try:
            _GPU_BACKEND = TorchPolicyBackend(plan)
            atexit.register(_GPU_BACKEND.close)
        except Exception as exc:
            if mode == "force" and not fallback_allowed:
                raise SystemExit(f"GPU backend initialization failed: {exc}")
            plan.use_gpu = False
            plan.device = "cpu"
            plan.dtype = "float32"
            plan.batch_size = 1
            plan.reason = f"GPU backend initialization failed: {exc}; using the CPU path."
            plan.warnings.append(str(exc))
            print(
                f"GPU backend initialization failed ({exc}); using the CPU path.",
                flush=True,
            )
        else:
            print(
                f"GPU policy backend enabled on {plan.device} ({plan.dtype}). "
                "This tiny policy is usually faster on the CPU; pass --cpu to disable.",
                flush=True,
            )

    executable_name = "doom_headless.exe" if os.name == "nt" else "doom_headless"
    executable = ROOT / "doom/target/release" / executable_name
    if args.build:
        subprocess.run([str(ROOT / "exec_mgmt.sh"), "-b"], cwd=ROOT, check=True)
    if not executable.is_file():
        raise SystemExit(f"{executable} is missing. Build with: ./exec_mgmt.sh -b")

    output = (ROOT / cfg["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if gpu_cfg.get("report", True):
        (output / "gpu_report.json").write_text(json.dumps(plan.to_dict(), indent=2))
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
            if _GPU_BACKEND is not None:
                total_jobs = len(jobs)
                progress_step = max(1, total_jobs // 20)
                started = time.perf_counter()
                print(
                    f"generation {generation + 1}: evaluating {total_jobs} episodes "
                    f"on {_GPU_BACKEND.device} (step-by-step GPU mode)",
                    flush=True,
                )
                episode_results = [None] * total_jobs
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(run_job_gpu, job): index
                        for index, job in enumerate(jobs)
                    }
                    for done, future in enumerate(
                        concurrent.futures.as_completed(futures), 1
                    ):
                        episode_results[futures[future]] = future.result()
                        if done % progress_step == 0 or done == total_jobs:
                            print(
                                f"  gpu episodes {done}/{total_jobs} | "
                                f"{time.perf_counter() - started:.1f}s",
                                flush=True,
                            )
                failures = [result for result in episode_results
                            if isinstance(result, BaseException)]
                if failures:
                    error = failures[0]
                    if mode == "force" and not fallback_allowed:
                        raise SystemExit(f"GPU backend failed: {error}")
                    print(
                        f"GPU backend failed ({error}); retrying this generation "
                        "on the CPU path.",
                        flush=True,
                    )
                    _GPU_BACKEND.close()
                    _GPU_BACKEND = None
                    plan.use_gpu = False
                    plan.device = "cpu"
                    plan.dtype = "float32"
                    plan.batch_size = 1
                    plan.reason = f"GPU backend failed: {error}; using the CPU path."
                    plan.warnings.append(str(error))
                    if gpu_cfg.get("report", True):
                        (output / "gpu_report.json").write_text(
                            json.dumps(plan.to_dict(), indent=2)
                        )
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as pool:
                        episode_results = list(pool.map(run_job, jobs))
            else:
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
            backend_label = "gpu" if _GPU_BACKEND is not None else "cpu"
            print(f"generation {generation + 1}/{cfg['generations']} | testers {workers} ({backend_label}) | "
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

    if _GPU_BACKEND is not None:
        _GPU_BACKEND.close()
        _GPU_BACKEND = None


def list_state(value):
    return [value[0], list(value[1]), value[2]]


def tuple_state(value):
    return (value[0], tuple(value[1]), value[2])


if __name__ == "__main__":
    raise SystemExit(main())
