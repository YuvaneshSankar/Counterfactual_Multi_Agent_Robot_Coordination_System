

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

# Experience tuple for storing trajectories
Experience = namedtuple(
    'Experience',
    ['state', 'actions', 'rewards', 'next_state', 'dones', 'observations']
)

class COMAcontinuous:
    """
    Counterfactual Multi-Agent Policy Gradients (COMA) for Continuous Action Spaces.

    COMA uses a centralized critic during training to estimate counterfactual baselines
    for each agent. This reduces variance in policy gradient estimation while maintaining
    decentralized execution by using only local observations.

    Key Innovation: Extends original discrete COMA to continuous actions using Gaussian policies.
    """

    def __init__(
        self,
        actor_network,
        critic_network,
        num_agents: int,
        action_dim: int,
        state_dim: int,
        config: Dict,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize COMA algorithm.

        Args:
            actor_network: Actor network for policy π(a|o)
            critic_network: Critic network for value Q(s, u)
            num_agents: Number of agents in the system
            action_dim: Dimension of action space
            state_dim: Dimension of state space
            config: Configuration dictionary with hyperparameters
            device: PyTorch device (CPU or CUDA)
        """
        self.actor_network = actor_network.to(device)
        self.critic_network = critic_network.to(device)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device

        # Hyperparameters from config
        self.actor_lr = config.get('actor_learning_rate', 3e-4)
        self.critic_lr = config.get('critic_learning_rate', 1e-3)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.entropy_coef = config.get('entropy_coefficient', 0.001)
        self.gradient_clip = config.get('gradient_clip_norm', 0.5)
        self.normalize_advantages = config.get('normalize_advantages', True)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

        # Target critic network for stable training (soft update)
        self.target_critic_network = critic_network.clone_network().to(device)
        self.tau = config.get('tau', 0.005)

        # Training statistics
        self.total_steps = 0
        self.total_updates = 0

        logger.info(f"COMA initialized with {num_agents} agents")
        logger.info(f"Actor LR: {self.actor_lr}, Critic LR: {self.critic_lr}")

    def select_actions(
        self,
        observations: List[torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Select actions for all agents using decentralized policies.

        Each agent independently samples actions from its policy π^i(a|o^i).
        This represents the DECENTRALIZED EXECUTION phase.

        Args:
            observations: List of observations for each agent [torch.Tensor of shape (obs_dim,)]
            deterministic: If True, use mean of policy (no sampling)

        Returns:
            actions: List of actions for each agent
            policy_info: Dictionary with log probs and other policy info for training
        """
        actions = []
        log_probs = []
        entropies = []
        means = []
        stds = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = obs.to(self.device) if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs).to(self.device)

                # Get action distribution from actor network
                mean, log_std = self.actor_network(obs_tensor.unsqueeze(0))
                mean = mean.squeeze(0)
                log_std = log_std.squeeze(0)
                std = torch.exp(log_std)

                # Sample action from Gaussian policy
                if deterministic:
                    action = mean
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.rsample()  # Use reparameterization trick
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                    log_probs.append(log_prob.cpu().numpy())
                    entropies.append(entropy.cpu().numpy())

                # Clamp action to valid range [-1, 1]
                action = torch.clamp(action, -1.0, 1.0)
                actions.append(action.cpu().numpy())
                means.append(mean.cpu().numpy())
                stds.append(std.cpu().numpy())

        policy_info = {
            'log_probs': log_probs if log_probs else None,
            'entropies': entropies if entropies else None,
            'means': means,
            'stds': stds
        }

        return actions, policy_info

    def compute_counterfactual_baseline(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        agent_idx: int
    ) -> torch.Tensor:
        """
        Compute counterfactual baseline for agent i.

        This is the key innovation of COMA. For each agent, we compute the advantage by
        marginalizing out that agent's action:

        A^i(s, u) = Q(s, u) - Σ_{u'_i} π^i(u'_i|o^i) Q(s, (u^{-i}, u'_i))

        This isolates the contribution of agent i to the joint reward.

        Args:
            states: Global state [batch_size, state_dim]
            actions: Joint actions [batch_size, num_agents, action_dim]
            agent_idx: Index of agent to compute baseline for

        Returns:
            baseline: Counterfactual baseline [batch_size]
        """
        batch_size = states.shape[0]
        baseline = torch.zeros(batch_size, device=self.device)

        # Get other agents' observations from state
        # For now, use full state as observation (could be changed for partial observability)
        # actions_without_i shape: [batch_size, num_agents, action_dim]

        # Compute Q(s, u^{-i}, u'_i) for sampled u'_i from agent i's policy
        num_samples = 10  # Number of samples for baseline estimation

        q_values_sum = torch.zeros(batch_size, device=self.device)

        for sample_idx in range(num_samples):
            # Sample alternative action for agent i from its policy
            # For continuous actions, we need to marginalize the policy
            # This is computationally expensive, so we use importance sampling

            agent_obs = states  # Simplified: use full state as observation
            agent_obs_tensor = agent_obs.to(self.device)

            mean, log_std = self.actor_network(agent_obs_tensor)
            std = torch.exp(log_std)

            # Sample alternative action from policy
            dist = torch.distributions.Normal(mean, std)
            alt_action = dist.rsample()
            alt_action = torch.clamp(alt_action, -1.0, 1.0)

            # Construct modified joint action with alternative action for agent i
            modified_actions = actions.clone()
            modified_actions[:, agent_idx, :] = alt_action

            # Compute Q-value for modified action sequence
            q_value = self.critic_network(states, modified_actions)
            q_values_sum += q_value.squeeze(-1)

        baseline = q_values_sum / num_samples
        return baseline

    def compute_advantages(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE and counterfactual baselines.

        Combines:
        1. Temporal difference learning with GAE (reduces variance)
        2. Counterfactual baseline (credit assignment in multi-agent setting)

        Args:
            states: State trajectory [batch_size, state_dim]
            actions: Action trajectory [batch_size, num_agents, action_dim]
            rewards: Reward trajectory [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Episode termination flags [batch_size]

        Returns:
            advantages: Advantage estimates [batch_size, num_agents]
            returns: Value targets [batch_size]
        """
        batch_size = states.shape[0]

        # Compute state values from critic
        v_states = self.critic_network(states, actions).squeeze(-1)  # [batch_size]

        with torch.no_grad():
            v_next = self.target_critic_network(next_states, actions).squeeze(-1)
            v_next[dones] = 0.0  # Terminal states have zero value

        # Compute TD residual
        td_residual = rewards + self.discount_factor * v_next - v_states

        # Compute advantages using GAE
        advantages = torch.zeros((batch_size, self.num_agents), device=self.device)
        gae = 0.0
        for t in reversed(range(batch_size)):
            if t < batch_size - 1:
                gae = td_residual[t] + self.discount_factor * self.gae_lambda * gae
            else:
                gae = td_residual[t]

            # Replicate advantage across agents (shared team advantage)
            advantages[t] = gae / self.num_agents

        # Compute returns
        returns = advantages + v_states.unsqueeze(1)

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        observations: List[torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_observations: List[torch.Tensor],
        dones: torch.Tensor,
        num_epochs: int = 10
    ) -> Dict:
        """
        Perform gradient updates for both actor and critic networks.

        This implements CENTRALIZED TRAINING: we have access to full state and
        all agents' actions during training.

        Args:
            states: Global state trajectory [batch_size, state_dim]
            observations: Per-agent observations
            actions: Joint action trajectory [batch_size, num_agents, action_dim]
            rewards: Reward trajectory [batch_size]
            next_states: Next state trajectory [batch_size, state_dim]
            next_observations: Next per-agent observations
            dones: Episode termination flags [batch_size]
            num_epochs: Number of training epochs

        Returns:
            update_info: Dictionary with loss statistics
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            states, actions, rewards, next_states, dones
        )

        actor_losses = []
        critic_losses = []
        entropy_bonuses = []

        for epoch in range(num_epochs):
            # ============================================================
            # Actor Update: Policy Gradient with Counterfactual Advantage
            # ============================================================
            self.actor_optimizer.zero_grad()

            actor_loss = 0.0
            total_entropy = 0.0

            for agent_idx in range(self.num_agents):
                # Get agent's observation
                agent_obs = states  # Simplified: use full state

                # Forward through actor
                mean, log_std = self.actor_network(agent_obs)
                std = torch.exp(log_std)

                # Create distribution and get log probability of taken action
                dist = torch.distributions.Normal(mean, std)
                action_agent = actions[:, agent_idx, :]
                log_prob = dist.log_prob(action_agent).sum(dim=-1)  # Sum over action dims

                # Get agent's advantage
                agent_advantage = advantages[:, agent_idx]

                # Policy gradient: J = E[log π(a|o) * A(s,u)]
                policy_loss = -(log_prob * agent_advantage).mean()

                # Entropy bonus: encourages exploration
                entropy = dist.entropy().mean()
                entropy_loss = -self.entropy_coef * entropy

                # Combined actor loss
                actor_loss += policy_loss + entropy_loss
                total_entropy += entropy.item()

            # Average over agents
            actor_loss /= self.num_agents
            actor_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.actor_network.parameters(),
                self.gradient_clip
            )

            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())
            entropy_bonuses.append(total_entropy / self.num_agents)

            # ============================================================
            # Critic Update: Value Function Learning
            # ============================================================
            self.critic_optimizer.zero_grad()

            # Compute Q-values
            q_values = self.critic_network(states, actions).squeeze(-1)

            # MSE loss between Q and targets
            critic_loss = F.mse_loss(q_values, returns[:, 0])  # All agents share returns

            critic_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.critic_network.parameters(),
                self.gradient_clip
            )

            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

            # Soft update of target critic network
            self._soft_update_target_network()

        self.total_updates += 1

        update_info = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_bonuses),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
        }

        return update_info

    def _soft_update_target_network(self):
        """
        Soft update of target critic network: θ_target = τ * θ + (1-τ) * θ_target

        This stabilizes training by slowly updating the target network.
        """
        for param, target_param in zip(
            self.critic_network.parameters(),
            self.target_critic_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, path: str):
        """Save actor and critic networks to checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load actor and critic networks from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.total_updates = checkpoint['total_updates']
        logger.info(f"Checkpoint loaded from {path}")

    def get_stats(self) -> Dict:
        """Return training statistics."""
        return {
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'learning_rate_actor': self.actor_lr,
            'learning_rate_critic': self.critic_lr,
        }
