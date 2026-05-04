#!/usr/bin/env python3
"""
run_all_experiments.py
----------------------
Runs all 60 experiments (3 algs x 4 envs x 5 seeds) sequentially,
exactly like running:
  python main.py --config=maddpg --env-config=lbf with t_max=2000000 seed=1 use_cuda=True 2>&1 | tee ~/marl_cv/logs/maddpg_lbf_s1.log

Usage (from ~/marl_cv/epymarl):
  python run_all_experiments.py
  python run_all_experiments.py --dry-run        # print commands only
  python run_all_experiments.py --env lbf        # single env
  python run_all_experiments.py --alg mappo      # single alg
  python run_all_experiments.py --env lbf --alg mappo --seed 1  # single run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Experiment grid ────────────────────────────────────────────────────────────
ALGORITHMS = ["qmix", "mappo", "maddpg"]
ENVIRONMENTS = ["lbf", "rware", "mpe", "overcooked"]
SEEDS = [1, 2, 3, 4, 5]
T_MAX = 2000000
# ──────────────────────────────────────────────────────────────────────────────

LOG_ROOT = Path.home() / "marl_cv" / "logs"


def build_command(alg: str, env: str, seed: int) -> list[str]:
    return [
        "python", "src/main.py",
        f"--config={alg}",
        f"--env-config={env}",
        "with",
        f"t_max={T_MAX}",
        f"seed={seed}",
        "use_cuda=True",
    ]


def run_experiment(alg: str, env: str, seed: int, dry_run: bool) -> bool:
    cmd = build_command(alg, env, seed)
    log_file = LOG_ROOT / f"{alg}_{env}_s{seed}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    label = f"{alg}__{env}__seed{seed}"
    cmd_str = " ".join(cmd) + f" 2>&1 | tee {log_file}"

    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return True

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ START  {label}")
    print(f"           CMD: {cmd_str}\n")

    t_start = time.time()
    with open(log_file, "w") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        process.wait()

    duration = int(time.time() - t_start)
    success = process.returncode == 0

    status = "✔ DONE " if success else "✘ FAILED"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}  {label}  ({duration}s)  log: {log_file}")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--alg",  choices=ALGORITHMS,    default=None, help="Run only this algorithm")
    parser.add_argument("--env",  choices=ENVIRONMENTS,  default=None, help="Run only this environment")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None, help="Run only this seed")
    args = parser.parse_args()

    algs  = [args.alg]  if args.alg  else ALGORITHMS
    envs  = [args.env]  if args.env  else ENVIRONMENTS
    seeds = [args.seed] if args.seed else SEEDS

    runs = [(alg, env, seed) for alg in algs for env in envs for seed in seeds]
    total = len(runs)

    print("═" * 55)
    print(f"  MARL Experiment Sweep")
    print(f"  Algorithms   : {algs}")
    print(f"  Environments : {envs}")
    print(f"  Seeds        : {seeds}")
    print(f"  Total runs   : {total}")
    print(f"  Logs         : {LOG_ROOT}")
    print("═" * 55)

    failed = []
    sweep_start = time.time()

    for i, (alg, env, seed) in enumerate(runs, 1):
        success = run_experiment(alg, env, seed, args.dry_run)
        if not success:
            failed.append(f"{alg}__{env}__seed{seed}")

        elapsed = int(time.time() - sweep_start)
        eta = int(elapsed * (total - i) / i) if i > 0 else 0
        print(f"  Progress: {i}/{total}  |  Elapsed: {elapsed}s  |  ETA: ~{eta}s")

    total_time = int(time.time() - sweep_start)
    print("\n" + "═" * 55)
    print(f"  Sweep complete in {total_time}s")
    if failed:
        print(f"  ⚠  FAILED runs ({len(failed)}):")
        for f in failed:
            print(f"     - {f}")
    else:
        print("  All runs succeeded ✔")
    print("═" * 55)


if __name__ == "__main__":
    main()