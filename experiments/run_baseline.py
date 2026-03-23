"""
Run baseline experiments for MARL+CV project.
"""

import os
import sys
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
ENVS = [
    ("mpe", "pz-mpe-simple-spread", 25),
]

ALGORITHMS = ["qmix", "mappo"]
SEEDS = [1, 2, 3, 4, 5]
T_MAX = 2050000

def run_experiment(env_name, env_key, time_limit, algorithm, seed):
    """Run a single experiment."""
    
    # Create experiment ID
    exp_id = f"{env_name}_{algorithm}_seed{seed}"
    print(f"\n{'='*50}")
    print(f"Running: {exp_id}")
    print(f"{'='*50}")
    
    # Build command
    cmd = [
        "python", "src/main.py",
        f"--config={algorithm}",
        "--env-config=gymma",
        "with",
        f"env_args.key='{env_key}'",
        f"env_args.time_limit={time_limit}",
        f"seed={seed}",
        f"t_max={T_MAX}",
        "log_interval=10000",
        "test_nepisode=32",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✓ {exp_id} completed")
    else:
        print(f"✗ {exp_id} failed")

if __name__ == "__main__":
    print("Starting baseline experiments...")
    print(f"Time: {datetime.now()}")
    
    for env_name, env_key, time_limit in ENVS:
        for alg in ALGORITHMS:
            for seed in SEEDS:
                run_experiment(env_name, env_key, time_limit, alg, seed)
    
    print(f"\nAll experiments completed at {datetime.now()}")