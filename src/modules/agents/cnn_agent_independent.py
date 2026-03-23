import torch.nn as nn
import torch.nn.functional as F
from modules.agents.cnn_encoder import IndependentCNNs


class CNNAgentIndependent(nn.Module):
    """
    Variant C: Fully Independent
    N separate CNN encoders + N independent policy MLPs.
    Used as baseline and for CKA analysis.
    """
    def __init__(self, input_shape, args):
        super(CNNAgentIndependent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.cnns = IndependentCNNs(
            n_agents=self.n_agents,
            in_channels=3,
            feature_dim=args.cnn_feature_dim
        )
        self.fc1 = nn.ModuleList([
            nn.Linear(args.cnn_feature_dim, args.hidden_dim)
            for _ in range(self.n_agents)
        ])
        if args.use_rnn:
            self.rnn = nn.ModuleList([
                nn.GRUCell(args.hidden_dim, args.hidden_dim)
                for _ in range(self.n_agents)
            ])
        else:
            self.rnn = nn.ModuleList([
                nn.Linear(args.hidden_dim, args.hidden_dim)
                for _ in range(self.n_agents)
            ])
        self.fc2 = nn.ModuleList([
            nn.Linear(args.hidden_dim, args.n_actions)
            for _ in range(self.n_agents)
        ])

    def init_hidden(self):
        return self.fc1[0].weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, agent_id=0):
        features = self.cnns(inputs, agent_id)
        x = F.relu(self.fc1[agent_id](features))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn[agent_id](x, h_in)
        else:
            h = F.relu(self.rnn[agent_id](x))
        q = self.fc2[agent_id](h)
        return q, h