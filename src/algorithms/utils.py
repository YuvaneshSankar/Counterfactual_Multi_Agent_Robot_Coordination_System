
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
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE provides a trade-off between bias and variance in advantage estimation:
    A^GAE = Σ (γλ)^t δ_{t+l}

    where δ_t = r_t + γ V(s_{t+1}) - V(s_t) is the TD residual.

    Args:
        rewards: Reward trajectory [batch_size]
        values: State value estimates [batch_size]
        next_values: Next state value estimates [batch_size]
        dones: Episode termination flags [batch_size]
        gamma: Discount factor
        lambda_: GAE parameter (0=TD, 1=MC)

    Returns:
        advantages: Advantage estimates [batch_size]
        returns: Value targets [batch_size]
    """
    batch_size = rewards.shape[0]

    # Mask out next values at terminal states
    next_values = next_values * (1 - dones.float())

    # Compute TD residuals
    deltas = rewards + gamma * next_values - values

    # Compute advantages using GAE
    advantages = torch.zeros(batch_size, device=rewards.device)
    gae = 0.0

    for t in reversed(range(batch_size)):
        gae = deltas[t] + gamma * lambda_ * gae * (1 - dones[t])
        advantages[t] = gae

    # Compute returns
    returns = advantages + values

    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.

    Prevents gradient explosion and stabilizes training.

    Args:
        advantages: Advantage tensor [batch_size, ...] or [batch_size, num_agents]
        epsilon: Small constant for numerical stability

    Returns:
        normalized_advantages: Normalized advantages
    """
    mean = advantages.mean()
    std = advantages.std()

    return (advantages - mean) / (std + epsilon)


def compute_policy_log_prob(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    actions: torch.Tensor,
    squashed: bool = False,
) -> torch.Tensor:
    """
    Compute log probability of actions under Gaussian policy.

    For continuous actions a = tanh(u), u ~ N(μ, σ²), we need to apply
    change-of-variables formula to the log probability.

    Args:
        means: Policy means [batch_size, action_dim]
        log_stds: Log standard deviations [batch_size, action_dim]
        actions: Sampled actions [batch_size, action_dim]
        squashed: Whether actions are squashed through tanh

    Returns:
        log_probs: Log probabilities [batch_size]
    """
    stds = torch.exp(log_stds)

    # Create distribution
    dist = torch.distributions.Normal(means, stds)

    # Log probability of unsquashed action
    if squashed:
        # For tanh-squashed actions, we need to compute log prob of pre-tanh values
        # This is complex; simplified version assumes actions are in [-1, 1]
        log_prob = dist.log_prob(actions)

        # Apply Jacobian correction for tanh
        log_prob = log_prob - torch.log(1 - actions.pow(2) + 1e-6)

        return log_prob.sum(dim=-1)
    else:
        log_prob = dist.log_prob(actions)
        return log_prob.sum(dim=-1)


def compute_entropy(
    means: torch.Tensor,
    log_stds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute entropy of Gaussian policy.

    H(π) = 0.5 * log(2πe * σ²) for scalar case
    For multivariate: H = Σ 0.5 * log(2πe * σ_i²)

    Args:
        means: Policy means [batch_size, action_dim]
        log_stds: Log standard deviations [batch_size, action_dim]

    Returns:
        entropy: Policy entropy [batch_size]
    """
    stds = torch.exp(log_stds)
    dist = torch.distributions.Normal(means, stds)
    entropy = dist.entropy().sum(dim=-1)
    return entropy


def compute_counterfactual_advantage(
    q_values: torch.Tensor,
    counterfactual_baselines: torch.Tensor,
    agent_idx: int,
) -> torch.Tensor:
    """
    Compute counterfactual advantage for a specific agent.

    The counterfactual advantage isolates the contribution of agent i:
    A^i(s, u) = Q(s, u) - b^i(s, u^{-i})

    where b^i is the counterfactual baseline that marginalizes agent i's action.

    Args:
        q_values: Q-values [batch_size]
        counterfactual_baselines: Baselines [batch_size]
        agent_idx: Index of agent (for tracking purposes)

    Returns:
        advantages: Counterfactual advantages [batch_size]
    """
    advantages = q_values - counterfactual_baselines
    return advantages


def check_numerical_stability(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN and Inf values in tensor.

    Args:
        tensor: Tensor to check
        name: Name for logging
        raise_error: Whether to raise error on instability

    Returns:
        is_stable: True if tensor is numerically stable
    """
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
    """
    Soft update target network towards source network.

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Args:
        source_network: Network to copy from
        target_network: Network to copy to
        tau: Update coefficient (0 = no update, 1 = full update)
    """
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
    """
    Hard update: copy all parameters from source to target network.

    Args:
        source_network: Network to copy from
        target_network: Network to copy to
    """
    target_network.load_state_dict(source_network.state_dict())


def get_gradient_norm(network: torch.nn.Module) -> float:
    """
    Compute total gradient norm for a network.

    Args:
        network: PyTorch network

    Returns:
        grad_norm: Total gradient norm
    """
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
    """
    Clip gradients to max norm for stable training.

    Args:
        network: PyTorch network
        max_norm: Maximum gradient norm

    Returns:
        grad_norm: Actual gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm).item()


def compute_td_error(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute Temporal Difference (TD) error.

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)

    Args:
        rewards: Rewards [batch_size]
        values: State values [batch_size]
        next_values: Next state values [batch_size]
        dones: Episode termination flags [batch_size]
        gamma: Discount factor

    Returns:
        td_errors: TD errors [batch_size]
    """
    next_values = next_values * (1 - dones.float())
    td_errors = rewards + gamma * next_values - values
    return td_errors


def compute_n_step_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    n_steps: int = 1,
) -> torch.Tensor:
    """
    Compute n-step return.

    G_t^(n) = Σ_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})

    Args:
        rewards: Reward trajectory [batch_size]
        values: Value trajectory [batch_size]
        gamma: Discount factor
        n_steps: Number of steps for n-step return

    Returns:
        n_step_returns: n-step returns [batch_size]
    """
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
    """
    Compute KL divergence between old and new policy.

    KL ≈ mean(exp(old_log_prob) * (old_log_prob - new_log_prob))

    Args:
        old_log_probs: Old policy log probabilities
        new_log_probs: New policy log probabilities

    Returns:
        kl_divergence: KL divergence estimate
    """
    old_probs = torch.exp(old_log_probs)
    kl = (old_probs * (old_log_probs - new_log_probs)).mean()
    return kl


def scale_actions(
    actions: torch.Tensor,
    action_scale: float = 1.0,
    action_offset: float = 0.0,
) -> torch.Tensor:
    """
    Scale and shift actions to valid range.

    Useful when environment expects actions in specific range.

    Args:
        actions: Raw actions [batch_size, action_dim]
        action_scale: Scaling factor
        action_offset: Offset

    Returns:
        scaled_actions: Scaled actions
    """
    return actions * action_scale + action_offset


def unscale_actions(
    scaled_actions: torch.Tensor,
    action_scale: float = 1.0,
    action_offset: float = 0.0,
) -> torch.Tensor:
    """
    Reverse action scaling/shifting.

    Args:
        scaled_actions: Scaled actions
        action_scale: Original scaling factor
        action_offset: Original offset

    Returns:
        actions: Original actions
    """
    return (scaled_actions - action_offset) / action_scale
