import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(observation_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.LongTensor:
        return self.q_net(observation)
