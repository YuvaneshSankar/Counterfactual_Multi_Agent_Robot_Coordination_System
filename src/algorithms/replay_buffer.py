

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience Replay Buffer for Multi-Agent Reinforcement Learning.

    Stores transitions and allows efficient batch sampling for training.
    Implements circular buffer to maintain fixed memory footprint.
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        obs_dim: int,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            state_dim: Dimension of global state
            action_dim: Dimension of single agent action
            num_agents: Number of agents
            obs_dim: Dimension of agent observation
            device: PyTorch device
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self.position = 0  # Current position in circular buffer
        self.is_full = False  # Whether buffer has wrapped around

        # Allocate memory for circular buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

        # Observations for each agent (used for decentralized execution)
        self.observations = [
            np.zeros((buffer_size, obs_dim), dtype=np.float32)
            for _ in range(num_agents)
        ]
        self.next_observations = [
            np.zeros((buffer_size, obs_dim), dtype=np.float32)
            for _ in range(num_agents)
        ]

        logger.info(
            f"ReplayBuffer initialized: size={buffer_size}, "
            f"state_dim={state_dim}, num_agents={num_agents}"
        )

    def add(
        self,
        state: np.ndarray,
        observations: List[np.ndarray],
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_observations: List[np.ndarray],
        done: bool,
    ):
        """
        Add a transition to the replay buffer.

        Args:
            state: Global state [state_dim]
            observations: Per-agent observations [List of obs_dim]
            actions: Joint actions [num_agents, action_dim]
            reward: Scalar reward
            next_state: Next global state [state_dim]
            next_observations: Next per-agent observations
            done: Episode termination flag
        """
        # Store in circular buffer at current position
        idx = self.position

        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = actions
        self.rewards[idx] = reward
        self.dones[idx] = done

        for agent_idx, obs in enumerate(observations):
            self.observations[agent_idx][idx] = obs
            self.next_observations[agent_idx][idx] = next_observations[agent_idx]

        # Move to next position
        self.position = (self.position + 1) % self.buffer_size

        # Mark as full once we wrap around
        if self.position == 0:
            self.is_full = True

    def add_trajectory(
        self,
        states: np.ndarray,
        observations: List[np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Add a complete episode trajectory to the buffer.

        Args:
            states: State trajectory [episode_length, state_dim]
            observations: Observation trajectories [num_agents, episode_length, obs_dim]
            actions: Action trajectory [episode_length, num_agents, action_dim]
            rewards: Reward trajectory [episode_length]
            dones: Done flags [episode_length]
        """
        episode_length = states.shape[0]

        for t in range(episode_length):
            state = states[t]
            next_state = states[t + 1] if t + 1 < episode_length else states[t]
            obs = [observations[i][t] for i in range(self.num_agents)]
            next_obs = [observations[i][t + 1] if t + 1 < episode_length
                       else observations[i][t] for i in range(self.num_agents)]
            action = actions[t]
            reward = rewards[t]
            done = dones[t]

            self.add(state, obs, action, reward, next_state, next_obs, done)

    def sample(
        self,
        batch_size: int,
        strategy: str = 'uniform'
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample
            strategy: Sampling strategy ('uniform', 'recent', 'prioritized')

        Returns:
            batch: Dictionary with sampled transitions as torch tensors
        """
        # Determine valid indices
        if self.is_full:
            valid_size = self.buffer_size
        else:
            valid_size = self.position

        if valid_size < batch_size:
            logger.warning(
                f"Batch size ({batch_size}) > valid buffer size ({valid_size}). "
                f"Returning smaller batch."
            )
            batch_size = valid_size

        # Sample indices based on strategy
        if strategy == 'uniform':
            indices = np.random.choice(valid_size, batch_size, replace=False)
        elif strategy == 'recent':
            # Bias towards recent samples
            weights = np.linspace(0.1, 1.0, valid_size)
            weights = weights / weights.sum()
            indices = np.random.choice(valid_size, batch_size, p=weights, replace=False)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Extract batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'observations': [
                torch.FloatTensor(self.observations[i][indices]).to(self.device)
                for i in range(self.num_agents)
            ],
            'next_observations': [
                torch.FloatTensor(self.next_observations[i][indices]).to(self.device)
                for i in range(self.num_agents)
            ],
        }

        return batch

    def sample_episodes(
        self,
        num_episodes: int,
        min_episode_length: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample complete episodes from the buffer.

        Useful for computing long-horizon returns and advantages.

        Args:
            num_episodes: Number of episodes to sample
            min_episode_length: Minimum episode length to sample

        Returns:
            episodes: Dictionary with episode trajectories
        """
        # Find episode boundaries (where done=True)
        done_indices = np.where(self.dones[:self.position if not self.is_full else self.buffer_size])[0]

        if len(done_indices) < num_episodes:
            logger.warning(f"Only {len(done_indices)} episodes in buffer, requested {num_episodes}")
            num_episodes = len(done_indices)

        # Sample episode indices
        episode_starts = [0] + list((done_indices + 1) % self.buffer_size)
        valid_episodes = []

        for i in range(len(episode_starts) - 1):
            start = episode_starts[i]
            end = episode_starts[i + 1]
            length = (end - start) % self.buffer_size if self.is_full else (end - start)

            if length >= min_episode_length:
                valid_episodes.append((start, end, length))

        if not valid_episodes:
            logger.warning("No valid episodes found")
            return self.sample(batch_size=min_episode_length)

        # Sample episodes
        sampled_episodes = np.random.choice(len(valid_episodes), num_episodes, replace=True)

        episodes = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        for ep_idx in sampled_episodes:
            start, end, length = valid_episodes[ep_idx]
            if self.is_full:
                indices = np.arange(start, end) % self.buffer_size
            else:
                indices = np.arange(start, end)

            episodes['states'].append(self.states[indices])
            episodes['actions'].append(self.actions[indices])
            episodes['rewards'].append(self.rewards[indices])
            episodes['dones'].append(self.dones[indices])

        # Convert to tensors
        max_length = max(e.shape[0] for e in episodes['states'])

        padded_states = np.zeros((num_episodes, max_length, self.state_dim))
        padded_actions = np.zeros((num_episodes, max_length, self.num_agents, self.action_dim))
        padded_rewards = np.zeros((num_episodes, max_length))
        padded_dones = np.zeros((num_episodes, max_length))

        for i, (s, a, r, d) in enumerate(zip(
            episodes['states'], episodes['actions'],
            episodes['rewards'], episodes['dones']
        )):
            length = s.shape[0]
            padded_states[i, :length] = s
            padded_actions[i, :length] = a
            padded_rewards[i, :length] = r
            padded_dones[i, :length] = d

        return {
            'states': torch.FloatTensor(padded_states).to(self.device),
            'actions': torch.FloatTensor(padded_actions).to(self.device),
            'rewards': torch.FloatTensor(padded_rewards).to(self.device),
            'dones': torch.BoolTensor(padded_dones).to(self.device),
            'lengths': torch.LongTensor([e.shape[0] for e in episodes['states']]),
        }

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.buffer_size if self.is_full else self.position

    def clear(self):
        """Clear the replay buffer."""
        self.position = 0
        self.is_full = False
        logger.info("Replay buffer cleared")

    def get_statistics(self) -> Dict:
        """Get statistics about the buffer contents."""
        valid_size = self.buffer_size if self.is_full else self.position

        if valid_size == 0:
            return {
                'buffer_size': 0,
                'fill_percentage': 0,
                'mean_reward': 0,
                'std_reward': 0,
            }

        valid_rewards = self.rewards[:valid_size]

        return {
            'buffer_size': valid_size,
            'fill_percentage': (valid_size / self.buffer_size) * 100,
            'mean_reward': float(np.mean(valid_rewards)),
            'std_reward': float(np.std(valid_rewards)),
            'min_reward': float(np.min(valid_rewards)),
            'max_reward': float(np.max(valid_rewards)),
        }
