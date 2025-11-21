
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = rewards.shape[0]

    next_values = next_values * (1 - dones.float())

    deltas = rewards + gamma * next_values - values

    advantages = torch.zeros(batch_size, device=rewards.device)
    gae = 0.0

    for t in reversed(range(batch_size)):
        gae = deltas[t] + gamma * lambda_ * gae * (1 - dones[t])
        advantages[t] = gae

    returns = advantages + values

    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:

    mean = advantages.mean()
    std = advantages.std()

    return (advantages - mean) / (std + epsilon)


def compute_policy_log_prob(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    actions: torch.Tensor,
    squashed: bool = False,
) -> torch.Tensor:
    stds = torch.exp(log_stds)

    dist = torch.distributions.Normal(means, stds)

    if squashed:

        log_prob = dist.log_prob(actions)

        log_prob = log_prob - torch.log(1 - actions.pow(2) + 1e-6)

        return log_prob.sum(dim=-1)
    else:
        log_prob = dist.log_prob(actions)
        return log_prob.sum(dim=-1)


def compute_entropy(
    means: torch.Tensor,
    log_stds: torch.Tensor,
) -> torch.Tensor:
    stds = torch.exp(log_stds)
    dist = torch.distributions.Normal(means, stds)
    entropy = dist.entropy().sum(dim=-1)
    return entropy


def compute_counterfactual_advantage(
    q_values: torch.Tensor,
    counterfactual_baselines: torch.Tensor,
    agent_idx: int,
) -> torch.Tensor:
    advantages = q_values - counterfactual_baselines
    return advantages


def check_numerical_stability(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = False,
) -> bool:

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan:
        logger.warning(f"{name} contains NaN values")
        if raise_error:
            raise RuntimeError(f"{name} contains NaN values")
        return False

    if has_inf:
        logger.warning(f"{name} contains Inf values")
        if raise_error:
            raise RuntimeError(f"{name} contains Inf values")
        return False

    return True


def soft_update_network(
    source_network: torch.nn.Module,
    target_network: torch.nn.Module,
    tau: float = 0.005,
):
   
    for source_param, target_param in zip(
        source_network.parameters(),
        target_network.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update_network(
    source_network: torch.nn.Module,
    target_network: torch.nn.Module,
):

    target_network.load_state_dict(source_network.state_dict())


def get_gradient_norm(network: torch.nn.Module) -> float:

    total_norm = 0.0
    for param in network.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradients(
    network: torch.nn.Module,
    max_norm: float = 1.0,
) -> float:

    return torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm).item()


def compute_td_error(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:

    next_values = next_values * (1 - dones.float())
    td_errors = rewards + gamma * next_values - values
    return td_errors


def compute_n_step_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    n_steps: int = 1,
) -> torch.Tensor:

    batch_size = rewards.shape[0]
    n_step_returns = torch.zeros(batch_size, device=rewards.device)

    for t in range(batch_size):
        n_step_return = 0.0
        for i in range(min(n_steps, batch_size - t)):
            n_step_return += (gamma ** i) * rewards[t + i]

        # Add bootstrapped value
        if t + n_steps < batch_size:
            n_step_return += (gamma ** n_steps) * values[t + n_steps]
        else:
            # Terminal state, no bootstrap
            pass

        n_step_returns[t] = n_step_return

    return n_step_returns


def compute_policy_divergence(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> torch.Tensor:

    old_probs = torch.exp(old_log_probs)
    kl = (old_probs * (old_log_probs - new_log_probs)).mean()
    return kl


def scale_actions(
    actions: torch.Tensor,
    action_scale: float = 1.0,
    action_offset: float = 0.0,
) -> torch.Tensor:

    return actions * action_scale + action_offset


def unscale_actions(
    scaled_actions: torch.Tensor,
    action_scale: float = 1.0,
    action_offset: float = 0.0,
) -> torch.Tensor:

    return (scaled_actions - action_offset) / action_scale
