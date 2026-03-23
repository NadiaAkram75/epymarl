import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
    Nature DQN CNN Encoder for pixel observations.
    Input:  (batch, 3, 84, 84) RGB image
    Output: (batch, 512) feature vector
    """
    def __init__(self, in_channels=3, feature_dim=512):
        super(NatureCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, in_channels, 84, 84)
        conv_out_size = self._get_conv_output(dummy)
        self.fc = nn.Linear(conv_out_size, feature_dim)
        self.feature_dim = feature_dim

    def _get_conv_output(self, x):
        with torch.no_grad():
            out = self.conv(x)
        return out.shape[1]

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class SharedCNN(nn.Module):
    """Single CNN shared across all agents — Variants A and B."""
    def __init__(self, in_channels=3, feature_dim=512):
        super(SharedCNN, self).__init__()
        self.encoder = NatureCNN(in_channels, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.encoder(x)


class IndependentCNNs(nn.Module):
    """N separate CNNs, one per agent — Variant C."""
    def __init__(self, n_agents, in_channels=3, feature_dim=512):
        super(IndependentCNNs, self).__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.encoders = nn.ModuleList([
            NatureCNN(in_channels, feature_dim) for _ in range(n_agents)
        ])

    def forward(self, x, agent_id):
        return self.encoders[agent_id](x)