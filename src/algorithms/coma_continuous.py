
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)


Experience = namedtuple(
    'Experience',
    ['state', 'actions', 'rewards', 'next_state', 'dones', 'observations']
)

class COMAcontinuous:


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

        self.actor_network = actor_network.to(device)
        self.critic_network = critic_network.to(device)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device

        self.actor_lr = config.get('actor_learning_rate', 3e-4)
        self.critic_lr = config.get('critic_learning_rate', 1e-3)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.entropy_coef = config.get('entropy_coefficient', 0.001)
        self.gradient_clip = config.get('gradient_clip_norm', 0.5)
        self.normalize_advantages = config.get('normalize_advantages', True)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)


        self.target_critic_network = critic_network.clone_network().to(device)
        self.tau = config.get('tau', 0.005)


        self.total_steps = 0
        self.total_updates = 0

        logger.info(f"COMA initialized with {num_agents} agents")
        logger.info(f"Actor LR: {self.actor_lr}, Critic LR: {self.critic_lr}")

    def select_actions(
        self,
        observations: List[torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[List[np.ndarray], Dict]:
        actions = []
        log_probs = []
        entropies = []
        means = []
        stds = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = obs.to(self.device) if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs).to(self.device)


                mean, log_std = self.actor_network(obs_tensor.unsqueeze(0))
                mean = mean.squeeze(0)
                log_std = log_std.squeeze(0)
                std = torch.exp(log_std)


                if deterministic:
                    action = mean
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.rsample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                    log_probs.append(log_prob.cpu().numpy())
                    entropies.append(entropy.cpu().numpy())


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

        batch_size = states.shape[0]
        baseline = torch.zeros(batch_size, device=self.device)


        num_samples = 10

        q_values_sum = torch.zeros(batch_size, device=self.device)

        for sample_idx in range(num_samples):

            agent_obs = states
            agent_obs_tensor = agent_obs.to(self.device)

            mean, log_std = self.actor_network(agent_obs_tensor)
            std = torch.exp(log_std)

            dist = torch.distributions.Normal(mean, std)
            alt_action = dist.rsample()
            alt_action = torch.clamp(alt_action, -1.0, 1.0)


            modified_actions = actions.clone()
            modified_actions[:, agent_idx, :] = alt_action


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

        batch_size = states.shape[0]


        v_states = self.critic_network(states, actions).squeeze(-1)

        with torch.no_grad():
            v_next = self.target_critic_network(next_states, actions).squeeze(-1)
            v_next[dones] = 0.0


        td_residual = rewards + self.discount_factor * v_next - v_states


        advantages = torch.zeros((batch_size, self.num_agents), device=self.device)
        gae = 0.0
        for t in reversed(range(batch_size)):
            if t < batch_size - 1:
                gae = td_residual[t] + self.discount_factor * self.gae_lambda * gae
            else:
                gae = td_residual[t]


            advantages[t] = gae / self.num_agents


        returns = advantages + v_states.unsqueeze(1)


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

            self.actor_optimizer.zero_grad()

            actor_loss = 0.0
            total_entropy = 0.0

            for agent_idx in range(self.num_agents):

                agent_obs = states


                mean, log_std = self.actor_network(agent_obs)
                std = torch.exp(log_std)


                dist = torch.distributions.Normal(mean, std)
                action_agent = actions[:, agent_idx, :]
                log_prob = dist.log_prob(action_agent).sum(dim=-1)


                agent_advantage = advantages[:, agent_idx]


                policy_loss = -(log_prob * agent_advantage).mean()


                entropy = dist.entropy().mean()
                entropy_loss = -self.entropy_coef * entropy


                actor_loss += policy_loss + entropy_loss
                total_entropy += entropy.item()

            actor_loss /= self.num_agents
            actor_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                self.actor_network.parameters(),
                self.gradient_clip
            )

            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())
            entropy_bonuses.append(total_entropy / self.num_agents)


            self.critic_optimizer.zero_grad()


            q_values = self.critic_network(states, actions).squeeze(-1)


            critic_loss = F.mse_loss(q_values, returns[:, 0])

            critic_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                self.critic_network.parameters(),
                self.gradient_clip
            )

            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())


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

        return {
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'learning_rate_actor': self.actor_lr,
            'learning_rate_critic': self.critic_lr,
        }
