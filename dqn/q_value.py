from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DQNLSTM(nn.Module):
    def __init__(
        self,
        observation_size: int,
        hidden_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super().__init__()
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._observation_size = observation_size

        self.input_net = nn.Sequential(
            nn.Linear(n_actions + observation_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, self._hidden_size),
        )
        self.rnn_encoder = nn.LSTM(fc_dims, hidden_size, batch_first=True)
        self.output_net = nn.Sequential(
            nn.Linear(self._hidden_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.LongTensor:
        prev_action_embs = F.one_hot(prev_action, num_classes=self._n_actions)
        input = torch.cat((observation, prev_action_embs), dim=-1)
        rnn_input = self.input_net(input)
        output, (_, _) = self.rnn_encoder(rnn_input)
        assert (
            len(output.shape) == 3
            and output.shape[0] == observation.shape[0]
            and output.shape[1] == observation.shape[1]
            and output.shape[2] == self._hidden_size
        ), f"{output.shape}"
        output_t = output.transpose(0, 1)
        masks = sequence_lengths.view(1, -1, 1).expand(
            sequence_lengths.max().item() + 1, output_t.size(1), output_t.size(2)
        )
        final_outputs = output_t.gather(0, masks)[0]
        actions_q = self.output_net(final_outputs)
        return actions_q

    def forward_step(
        self,
        observation: torch.Tensor,
        prev_action: int,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            len(observation.shape) == 2
            and observation.shape[0] == 1
            and observation.shape[1] == self._observation_size
        ), observation.shape
        prev_action_emb = F.one_hot(
            torch.tensor([prev_action], device=observation.device),
            num_classes=self._n_actions,
        )
        input = torch.cat((observation, prev_action_emb), dim=-1)
        rnn_input = self.input_net(input)
        if h_prev is None or c_prev is None:
            output, (hn, cn) = self.rnn_encoder(rnn_input)
        else:
            output, (hn, cn) = self.rnn_encoder(rnn_input, (h_prev, c_prev))
        actions_q = self.output_net(output)
        return actions_q, hn, cn


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
