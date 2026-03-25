import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
# from .smaclite_wrapper import SMACliteWrapper  # Disabled — not needed


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["gymma"] = gymma_fn


def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn