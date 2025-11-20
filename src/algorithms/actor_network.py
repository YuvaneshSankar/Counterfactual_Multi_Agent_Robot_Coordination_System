

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: list = [256, 256],
        activation: str = 'relu',
        output_activation: str = 'tanh',
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
        init_std: float = 0.5,
        learn_std: bool = True,
    ):
        """Learn_std: Whether standard deviation is learnable (not just observation-dependent)"""
        super(ActorNetwork, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.learn_std = learn_std
        self.init_std = init_std

        if activation.lower() == 'relu':
            self.activation_fn = F.relu
        elif activation.lower() == 'tanh':
            self.activation_fn = torch.tanh
        elif activation.lower() == 'elu':
            self.activation_fn = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if output_activation.lower() == 'tanh':
            self.output_activation_fn = torch.tanh
        elif output_activation.lower() == 'relu':
            self.output_activation_fn = F.relu
        elif output_activation.lower() == 'sigmoid':
            self.output_activation_fn = torch.sigmoid
        else:
            self.output_activation_fn = None

        """Shared Feature Extraction Layers"""
        self.shared_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        input_dim = observation_dim
        for hidden_dim in hidden_sizes:
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

            input_dim = hidden_dim

        """Mean Head"""
        self.mean_layer = nn.Linear(input_dim, action_dim)

        """Initialize mean layer with small weights"""
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)



        """Log-Std Head"""
        """We have two options:
        1. State-dependent log-std: fully connected layer output
        2. Learnable log-std: single parameter per action dimension"""

        if learn_std:
            """Learnable log-std parameter (not dependent on state)"""
            self.log_std_param = nn.Parameter(
                torch.ones(action_dim) * np.log(init_std)
            )
            self.log_std_layer = None
        else:
            """State-dependent log-std"""
            self.log_std_layer = nn.Linear(input_dim, action_dim)
            nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)

        logger.info(f"ActorNetwork initialized: {observation_dim} -> {hidden_sizes} -> {action_dim}")

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = observations

        for i, layer in enumerate(self.shared_layers):
            x = layer(x)

            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            x = self.activation_fn(x)
            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=True)
        mean = self.mean_layer(x)

        if self.output_activation_fn is not None:
            mean = self.output_activation_fn(mean)


        if self.learn_std:
            """Expand learnable parameter to batch size"""
            batch_size = observations.shape[0]
            log_std = self.log_std_param.expand(batch_size, -1)
        else:
            log_std = self.log_std_layer(x)


        log_std = torch.clamp(log_std, min=-2.0, max=2.0)

        return mean, log_std

    def get_action_distribution(
        self,
        observations: torch.Tensor
    ) -> torch.distributions.Normal:
        mean, log_std = self.forward(observations)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6, max=1e1)

        return torch.distributions.Normal(mean, std)

    def sample_action(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        return_log_prob: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mean, log_std = self.forward(observations)
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-6, max=1e1)

        if deterministic:
            action = mean
            if return_log_prob:
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                return action, log_prob
            return action, None
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0)

            if return_log_prob:
                log_prob = dist.log_prob(action)

                if self.output_activation_fn == torch.tanh:
                    log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)

                log_prob = log_prob.sum(dim=-1, keepdim=True)
                return action, log_prob

            return action, None

    def clone_network(self):
        clone = ActorNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_sizes=[self.shared_layers[i].out_features
                         for i in range(len(self.shared_layers) - 1)] +
                        [self.shared_layers[-1].out_features],
            activation='relu',
            output_activation='tanh' if self.output_activation_fn == torch.tanh else 'none',
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            init_std=self.init_std,
            learn_std=self.learn_std,
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def get_regularization_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return reg_loss
