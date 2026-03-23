import torch.nn as nn
import torch.nn.functional as F
from modules.agents.cnn_encoder import SharedCNN


class CNNAgentShared(nn.Module):
    """
    Variant A: Fully Shared
    One CNN encoder + one policy MLP shared across ALL agents.
    """
    def __init__(self, input_shape, args):
        super(CNNAgentShared, self).__init__()
        self.args = args
        self.cnn = SharedCNN(in_channels=3, feature_dim=args.cnn_feature_dim)
        self.fc1 = nn.Linear(args.cnn_feature_dim, args.hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        features = self.cnn(inputs)
        x = F.relu(self.fc1(features))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h