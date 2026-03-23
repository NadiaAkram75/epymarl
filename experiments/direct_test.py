import sys
import os
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch as th
import numpy as np
import yaml
from copy import deepcopy

print("="*50)
print("DEBUG: Running direct test")
print("="*50)
print(f"Python path: {sys.path}")

# Load configs manually
print("Loading default.yaml...")
with open('src/config/default.yaml', 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

print("Loading gymma.yaml...")
with open('src/config/envs/gymma.yaml', 'r') as f:
    env_config = yaml.load(f, Loader=yaml.FullLoader)

print("Loading qmix.yaml...")
with open('src/config/algs/qmix.yaml', 'r') as f:
    alg_config = yaml.load(f, Loader=yaml.FullLoader)

# Update configs
print("Merging configs...")
config_dict.update(env_config)
config_dict.update(alg_config)

# Set environment args
config_dict['env_args']['key'] = 'pz-mpe-simple-spread'
config_dict['env_args']['time_limit'] = 25
config_dict['env_args']['seed'] = 123
config_dict['test_nepisode'] = 1
config_dict['seed'] = 123
config_dict['use_cuda'] = False  # Disable CUDA for testing
config_dict['log_interval'] = 1  # Log every episode
config_dict['runner_log_interval'] = 1
config_dict['learner_log_interval'] = 1
config_dict['batch_size_run'] = 1
config_dict['batch_size'] = 32
config_dict['buffer_size'] = 5000
config_dict['t_max'] = 100  # Run for just 100 timesteps for testing

print(f"Config loaded: {list(config_dict.keys())}")
print(f"Environment args: {config_dict['env_args']}")

# Import run function
print("Importing run function...")
from run import run
from utils.logging import Logger

print("Run function imported successfully")

# Create a proper logger
class DummyLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")
    
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    
    def error(self, msg):
        print(f"ERROR: {msg}")

# Create dummy sacred run object with logging methods
class DummyRun:
    def __init__(self):
        self.info = {}
        self._id = 0
    
    def log_scalar(self, key, value, t):
        print(f"LOG: {key} = {value} at t={t}")
    
    def log_metric(self, key, value, t):
        print(f"METRIC: {key} = {value} at t={t}")

_run = DummyRun()
_log = DummyLogger()

# Call run directly
print("Calling run function...")
run(_run, config_dict, _log)
print("Run function completed")