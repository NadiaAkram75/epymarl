#!/usr/bin/env python
"""
Working baseline experiments for MARL+CV project.
Uses the direct approach that we know works.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch as th
import numpy as np
import yaml

# Configuration
ENVS = [
    ("mpe", "pz-mpe-simple-spread", 25),
    ("lbf", "Foraging-8x8-2p-3f-v3", 50),  # Updated LBF name
]

ALGORITHMS = ["qmix"]  # Start with qmix for testing
SEEDS = [1]  # Just test seed 1 first
T_MAX = 100  # Quick test first
# T_MAX = 2050000  # Full training (commented out for testing)

def create_config(env_key, time_limit, algorithm, seed):
    """Create config dictionary for experiment."""
    
    # Load base configs
    with open('src/config/default.yaml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    with open('src/config/envs/gymma.yaml', 'r') as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(f'src/config/algs/{algorithm}.yaml', 'r') as f:
        alg_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Merge configs
    config_dict.update(env_config)
    config_dict.update(alg_config)
    
    # Set experiment parameters
    config_dict['env_args']['key'] = env_key
    print(f"DEBUG: Setting env_key to: {env_key}")  # ADD THIS LINE
    config_dict['env_args']['time_limit'] = time_limit
    config_dict['env_args']['seed'] = seed
    config_dict['seed'] = seed
    config_dict['t_max'] = T_MAX
    config_dict['use_cuda'] = False
    config_dict['log_interval'] = 10000
    config_dict['runner_log_interval'] = 10000
    config_dict['learner_log_interval'] = 10000
    config_dict['test_interval'] = 50000
    config_dict['test_nepisode'] = 32
    config_dict['batch_size_run'] = 1
    
    return config_dict

def run_experiment(env_name, env_key, time_limit, algorithm, seed):
    """Run a single experiment."""
    
    # Create experiment ID and directories
    exp_id = f"{env_name}_{algorithm}_seed{seed}"
    results_dir = Path("results") / "baselines" / env_name / algorithm / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting: {exp_id} at {datetime.now()}")
    print(f"{'='*60}")
    print(f"Results will be saved to: {results_dir}")
    
    # Create config
    config = create_config(env_key, time_limit, algorithm, seed)
    
    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Import run function
    from run import run
    
    # Create logger
    class ExperimentLogger:
        def info(self, msg): 
            print(f"INFO: {msg}")
            with open(results_dir / "log.txt", "a") as f:
                f.write(f"INFO: {msg}\n")
        def warning(self, msg): 
            print(f"WARNING: {msg}")
            with open(results_dir / "log.txt", "a") as f:
                f.write(f"WARNING: {msg}\n")
        def debug(self, msg): 
            print(f"DEBUG: {msg}")
        def error(self, msg): 
            print(f"ERROR: {msg}")
            with open(results_dir / "log.txt", "a") as f:
                f.write(f"ERROR: {msg}\n")
    
    # Create dummy sacred run object
    class ExperimentRun:
        def __init__(self):
            self.info = {}
            self._id = 0
        def log_scalar(self, key, value, t):
            print(f"LOG: {key} = {value} at t={t}")
            with open(results_dir / "metrics.txt", "a") as f:
                f.write(f"{t},{key},{value}\n")
        def log_metric(self, key, value, t):
            pass
    
    # Run experiment
    start_time = time.time()
    try:
        run(ExperimentRun(), config, ExperimentLogger())
        elapsed = time.time() - start_time
        print(f"✓ {exp_id} completed in {elapsed:.1f}s")
        
        # Save success marker
        with open(results_dir / "SUCCESS", "w") as f:
            f.write(f"Completed at {datetime.now()}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {exp_id} failed after {elapsed:.1f}s: {e}")
        
        # Save error log
        with open(results_dir / "ERROR", "w") as f:
            f.write(f"Failed at {datetime.now()}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write(f"Error: {e}\n")
        
        return False

def main():
    """Run all baseline experiments."""
    
    print("="*60)
    print("MARL + CV Baseline Experiments (Working Version)")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Environments: {[e[0] for e in ENVS]}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(ENVS) * len(ALGORITHMS) * len(SEEDS)}")
    print("="*60)
    
    # Create results directory
    Path("results/baselines").mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for env_name, env_key, time_limit in ENVS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                success = run_experiment(env_name, env_key, time_limit, algorithm, seed)
                results.append({
                    "env": env_name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "success": success
                })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Save summary
    summary_file = Path("results") / "baselines" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        import json
        json.dump({
            "timestamp": str(datetime.now()),
            "total": total,
            "successful": successful,
            "results": results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"\nAll experiments completed at {datetime.now()}")

if __name__ == "__main__":
    main()