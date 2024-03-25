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
            # nn.BatchNorm1d(observation_size),
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


class GradScalerFunctional(torch.autograd.Function):
    """
    A torch.autograd.Function works as Identity on forward pass
    and scales the gradient by scale_factor on backward pass.
    """
    @staticmethod
    def forward(ctx, input, scale_factor):
        ctx.scale_factor = scale_factor
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale_factor = ctx.scale_factor
        grad_input = grad_output * scale_factor
        return grad_input, None


class GradScaler(nn.Module):
    """
    An nn.Module incapsulating GradScalerFunctional
    """
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return GradScalerFunctional.apply(x, self.scale_factor)


class DuelingDQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DuelingDQN, self).__init__()
        self.q_net = nn.Sequential(
            # nn.BatchNorm1d(observation_size),
            nn.Linear(observation_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
        )

        self.grad_scaler = GradScaler(1 / 2 ** 0.5)

        self.adv_stream = nn.Sequential(
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.q_net(observation)
        x = self.grad_scaler(x)
        adv = self.adv_stream(x)
        adv_mean = torch.mean(adv, dim=1)[..., None]
        value = self.value_stream(x)
        q_value = adv + value - adv_mean
        return q_value


class DQNWithObservationSequenceEncoder(nn.Module):
    def __init__(
        self,
        observation_size: int,
        token_dims: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.q_net = nn.Sequential(
            # nn.BatchNorm1d(observation_size),
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
        self.rnn_encoder = nn.LSTM(
            self._hidden_size, self._hidden_size, batch_first=True
        )
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
    ) -> torch.Tensor:
        prev_action_embs = F.one_hot(prev_action, num_classes=self._n_actions)
        input_batch = torch.cat((observation, prev_action_embs), dim=-1)
        assert (
            len(input_batch.shape) == 3
            and input_batch.shape[0] == observation.shape[0]
            and input_batch.shape[1] == observation.shape[1]
            and input_batch.shape[2] == self._n_actions + self._observation_size
        ), f"{input_batch.shape}"
        rnn_input = self.input_net(input_batch)
        output, (_, _) = self.rnn_encoder(rnn_input)
        assert (
            len(output.shape) == 3
            and output.shape[0] == observation.shape[0]
            and output.shape[1] == observation.shape[1]
            and output.shape[2] == self._hidden_size
        ), f"{output.shape}"
        output_t = output.transpose(0, 1)
        masks = (
            (sequence_lengths - 1)
            .view(1, -1, 1)
            .expand(
                sequence_lengths.max().item(),
                output_t.size(1),
                output_t.size(2),
            )
        )
        final_outputs = output_t.gather(0, masks)[0]
        actions_q = self.output_net(final_outputs)
        return actions_q

    def forward_step(
        self,
        observation: torch.Tensor,
        prev_action: int,
        h_prev: Optional[torch.Tensor],
        c_prev: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_action_emb = F.one_hot(
            torch.tensor(prev_action, device=observation.device),
            num_classes=self._n_actions,
        )
        input_vector = torch.cat((observation, prev_action_emb), dim=-1)
        assert (
            len(input_vector.shape) == 1
            and input_vector.shape[0] == self._n_actions + self._observation_size
        ), f"{input_vector.shape}"
        rnn_input = self.input_net(input_vector[None, None, ...])
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
        input_dims = observation_size + action_emb_dims
        self.q_net = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Linear(input_dims, fc_dims),
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
