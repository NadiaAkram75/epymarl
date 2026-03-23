from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .cnn_agent_shared import CNNAgentShared
from .cnn_agent_shared_enc import CNNAgentSharedEnc
from .cnn_agent_independent import CNNAgentIndependent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent


# CNN agents — 3 variants for pixel observations
REGISTRY["cnn_shared"] = CNNAgentShared
REGISTRY["cnn_shared_enc"] = CNNAgentSharedEnc
REGISTRY["cnn_independent"] = CNNAgentIndependent