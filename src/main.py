try:
    # until python 3.10
    from collections import Mapping
except:
    # from python 3.10
    from collections.abc import Mapping
from copy import deepcopy
import os
from os.path import dirname, abspath
import sys
import yaml

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th

from utils.logging import get_logger
from run import run

SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"


@ex.main
def my_main(_run, _config, _log):
    print("="*50)
    print("DEBUG: ENTERED MY_MAIN FUNCTION!")
    print("="*50)
    print(f"DEBUG: _run: {_run}")
    print(f"DEBUG: _config keys: {list(_config.keys())}")
    print(f"DEBUG: _log: {_log}")
    
    # Setting the random seed throughout the modules
    print("DEBUG: Copying config...")
    config = config_copy(_config)
    print("DEBUG: Setting seeds...")
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    print(f"DEBUG: Seed set to {config['seed']}")
    print(f"DEBUG: Environment args: {config['env_args']}")

    # run the framework
    print("DEBUG: Calling run() function")
    run(_run, config, _log)
    print("DEBUG: run() function completed")


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "config",
            subfolder,
            "{}.yaml".format(config_name),
        )
        print(f"DEBUG: Loading config from {file_path}")
        with open(file_path, "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                print(f"DEBUG: Successfully loaded {config_name}.yaml")
                return config_dict
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
    return {}


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == "__main__":
    print("="*50)
    print("DEBUG: Starting main.py")
    print("="*50)
    
    params = deepcopy(sys.argv)
    print(f"DEBUG: Command line params: {params}")
    
    th.set_num_threads(1)
    print("DEBUG: Set torch threads to 1")

    # Get the defaults from default.yaml
    print("DEBUG: Loading default.yaml")
    default_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    print(f"DEBUG: Default config path: {default_path}")
    
    with open(default_path, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            print("DEBUG: default.yaml loaded successfully")
            print(f"DEBUG: Default config keys: {list(config_dict.keys())}")
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    print("DEBUG: Loading environment config")
    env_config = _get_config(params, "--env-config", "envs")
    print(f"DEBUG: Environment config: {env_config}")
    
    print("DEBUG: Loading algorithm config")
    alg_config = _get_config(params, "--config", "algs")
    print(f"DEBUG: Algorithm config: {alg_config}")
    
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    print("DEBUG: Configs merged successfully")

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]
    print(f"DEBUG: Map name: {map_name}")

    # now add all the config to sacred
    print("DEBUG: Adding config to sacred experiment")
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}"
    )
    print(f"DEBUG: Results will be saved to: {file_obs_path}")

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    print("DEBUG: FileStorageObserver added")
    # ex.observers.append(MongoObserver())

    print("DEBUG: About to run sacred experiment")
    ex.run_commandline(params)
    print("DEBUG: Sacred experiment finished")