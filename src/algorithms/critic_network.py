
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_sizes: list = [512, 512],
        activation: str = 'relu',
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
    ):

        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.joint_action_dim = action_dim * num_agents
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        if activation.lower() == 'relu':
            self.activation_fn = F.relu
        elif activation.lower() == 'tanh':
            self.activation_fn = torch.tanh
        elif activation.lower() == 'elu':
            self.activation_fn = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        """State encoder"""
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU() if activation.lower() == 'relu' else nn.Tanh(),
        )

        """Action encoder"""
        self.action_encoder = nn.Sequential(
            nn.Linear(self.joint_action_dim, hidden_sizes[0]),
            nn.ReLU() if activation.lower() == 'relu' else nn.Tanh(),
        )

        """Fusion Layers: combine state and action information"""
        self.fusion_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()


        fusion_input_dim = 2 * hidden_sizes[0]

        for hidden_dim in hidden_sizes:
            self.fusion_layers.append(nn.Linear(fusion_input_dim, hidden_dim))

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

            fusion_input_dim = hidden_dim


        self.value_head = nn.Linear(fusion_input_dim, 1)


        nn.init.uniform_(self.value_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value_head.bias, -3e-3, 3e-3)

        logger.info(
            f"CriticNetwork initialized: state_dim={state_dim}, "
            f"joint_action_dim={self.joint_action_dim}, "
            f"hidden_sizes={hidden_sizes}"
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:

        batch_size = states.shape[0]

        """If actions are given per agent, flatten them"""
        if len(actions.shape) == 3:  # [batch_size, num_agents, action_dim]
            actions = actions.reshape(batch_size, -1)  # [batch_size, joint_action_dim]



        state_encoding = self.state_encoder(states)


        action_encoding = self.action_encoder(actions)

        x = torch.cat([state_encoding, action_encoding], dim=-1)

        for i, layer in enumerate(self.fusion_layers):
            x = layer(x)


            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)


            x = self.activation_fn(x)


            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=True)


        q_value = self.value_head(x)
        return q_value

    def compute_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:

        return self.forward(states, actions)

    def clone_network(self):
        cloned = CriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_sizes=[layer.out_features for layer in self.fusion_layers],
            activation='relu',
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )

        cloned.load_state_dict(self.state_dict())

        return cloned

    def get_regularization_loss(self, weight_decay: float = 0.0001) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return weight_decay * reg_loss

    def get_state_action_importance(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        """This function computes the importance (saliency) of each input feature in the states and actions to the critic's output (Q-value), by looking at gradients of the output w.r.t. inputs. It tells you how sensitive the Q-value is to each input dimension."""
        batch_size = states.shape[0]

        states_clone = states.clone().requires_grad_(True)
        actions_clone = actions.clone().requires_grad_(True)

        if len(actions_clone.shape) == 3:
            actions_clone = actions_clone.reshape(batch_size, -1)

        q_values = self.forward(states_clone, actions_clone)
        loss = q_values.sum()
        loss.backward()


        state_grads = states_clone.grad.abs().mean(dim=0)
        action_grads = actions_clone.grad.abs().mean(dim=0)

        return state_grads, action_grads
