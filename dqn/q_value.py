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

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.q_net(observation)


class OneValueDQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        fc_dims: int,
        action_emb_dims: int,
        n_actions: int,
        device,
    ):
        super(OneValueDQN, self).__init__()
        self.device = device
        self._n_actions = n_actions
        self.action_emb = nn.Embedding(n_actions, action_emb_dims)
        self.q_net = nn.Sequential(
            nn.Linear(observation_size + action_emb_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        actions_q_values = []
        for action in range(self._n_actions):
            actions_q_values.append(
                self.forward_per_action(
                    observation,
                    torch.full((observation.shape[0],), action, device=self.device),
                )
            )
        action_q_values = torch.stack(actions_q_values, dim=1)
        return action_q_values

    def forward_per_action(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        action_embs = self.action_emb(action)
        input = torch.cat((observation, action_embs), dim=-1)
        action_q_value = self.q_net(input)
        return action_q_value
