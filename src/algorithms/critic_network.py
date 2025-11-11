
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    """
    Centralized Critic Network for Multi-Agent Systems.

    Estimates Q(s, u) = expected return when taking joint action u in state s.

    Inputs:
    - Global state s: Full state information visible during training
    - Joint actions u: Actions of all agents concatenated

    The network processes both state and actions to produce value estimates
    that guide policy learning for all agents.

    Architecture:
    - State encoding: processes global state
    - Action encoding: processes joint actions
    - Fusion layers: combines state and action representations
    - Value head: outputs Q-value
    """

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
        """
        Initialize critic network.

        Args:
            state_dim: Dimension of global state
            action_dim: Dimension of single agent's action
            num_agents: Number of agents
            hidden_sizes: List of hidden layer dimensions
            activation: Activation function for hidden layers
            use_layer_norm: Whether to use layer normalization
            dropout_rate: Dropout rate for regularization
        """
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.joint_action_dim = action_dim * num_agents
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation.lower() == 'relu':
            self.activation_fn = F.relu
        elif activation.lower() == 'tanh':
            self.activation_fn = torch.tanh
        elif activation.lower() == 'elu':
            self.activation_fn = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ============================================================
        # State Encoder: processes global state
        # ============================================================
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU() if activation.lower() == 'relu' else nn.Tanh(),
        )

        # ============================================================
        # Action Encoder: processes joint actions
        # ============================================================
        self.action_encoder = nn.Sequential(
            nn.Linear(self.joint_action_dim, hidden_sizes[0]),
            nn.ReLU() if activation.lower() == 'relu' else nn.Tanh(),
        )

        # ============================================================
        # Fusion Layers: combine state and action information
        # ============================================================
        self.fusion_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Input to fusion is concatenation of state and action encodings
        fusion_input_dim = 2 * hidden_sizes[0]

        for hidden_dim in hidden_sizes:
            self.fusion_layers.append(nn.Linear(fusion_input_dim, hidden_dim))

            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

            fusion_input_dim = hidden_dim

        # ============================================================
        # Value Head: outputs Q-value
        # ============================================================
        self.value_head = nn.Linear(fusion_input_dim, 1)

        # Initialize value head with small weights
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
        """
        Forward pass through critic network.

        Args:
            states: Global state batch [batch_size, state_dim]
            actions: Joint actions batch [batch_size, num_agents, action_dim]
                    or [batch_size, joint_action_dim]

        Returns:
            q_values: Estimated Q-values [batch_size, 1]
        """
        batch_size = states.shape[0]

        # Reshape actions if needed
        if len(actions.shape) == 3:  # [batch_size, num_agents, action_dim]
            actions = actions.reshape(batch_size, -1)  # [batch_size, joint_action_dim]

        # ============================================================
        # Encode state
        # ============================================================
        state_encoding = self.state_encoder(states)  # [batch_size, hidden_sizes[0]]

        # ============================================================
        # Encode actions
        # ============================================================
        action_encoding = self.action_encoder(actions)  # [batch_size, hidden_sizes[0]]

        # ============================================================
        # Fuse state and action information
        # ============================================================
        x = torch.cat([state_encoding, action_encoding], dim=-1)  # [batch_size, 2*hidden]

        # Process through fusion layers
        for i, layer in enumerate(self.fusion_layers):
            x = layer(x)

            # Layer normalization
            if self.use_layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)

            # Activation
            x = self.activation_fn(x)

            # Dropout
            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=True)

        # ============================================================
        # Output Q-value
        # ============================================================
        q_value = self.value_head(x)  # [batch_size, 1]

        return q_value

    def compute_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values for state-action pairs.

        Alias for forward() for clarity in code.
        """
        return self.forward(states, actions)

    def clone_network(self):
        """
        Create a clone of this network with same architecture but independent weights.

        Used for creating target networks in stable training.

        Returns:
            cloned_network: Independent copy of this critic network
        """
        cloned = CriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_sizes=[layer.out_features for layer in self.fusion_layers],
            activation='relu',
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )

        # Copy weights
        cloned.load_state_dict(self.state_dict())

        return cloned

    def get_regularization_loss(self, weight_decay: float = 0.0001) -> torch.Tensor:
        """
        Compute L2 regularization loss for network parameters.

        Args:
            weight_decay: Regularization coefficient

        Returns:
            reg_loss: Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return weight_decay * reg_loss

    def get_state_action_importance(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute importance scores for state and action components.

        Used for interpretability and debugging.

        Args:
            states: Global state batch
            actions: Joint actions batch

        Returns:
            state_importance: Importance of state features
            action_importance: Importance of action features
        """
        batch_size = states.shape[0]

        # Compute gradients w.r.t. state and actions
        states_clone = states.clone().requires_grad_(True)
        actions_clone = actions.clone().requires_grad_(True)

        if len(actions_clone.shape) == 3:
            actions_clone = actions_clone.reshape(batch_size, -1)

        q_values = self.forward(states_clone, actions_clone)
        loss = q_values.sum()
        loss.backward()

        # Extract gradients
        state_grads = states_clone.grad.abs().mean(dim=0)  # Average over batch
        action_grads = actions_clone.grad.abs().mean(dim=0)

        return state_grads, action_grads
