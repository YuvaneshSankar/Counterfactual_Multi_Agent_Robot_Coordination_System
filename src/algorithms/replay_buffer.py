

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:


    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        obs_dim: int,
        device: torch.device = torch.device('cpu'),
    ):

        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self.position = 0  # Current position in circular buffer
        self.is_full = False  # Whether buffer has wrapped around

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)


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

        if strategy == 'uniform':
            indices = np.random.choice(valid_size, batch_size, replace=False)
        elif strategy == 'recent':
            weights = np.linspace(0.1, 1.0, valid_size)
            weights = weights / weights.sum()
            indices = np.random.choice(valid_size, batch_size, p=weights, replace=False)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

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

        done_indices = np.where(self.dones[:self.position if not self.is_full else self.buffer_size])[0]

        if len(done_indices) < num_episodes:
            logger.warning(f"Only {len(done_indices)} episodes in buffer, requested {num_episodes}")
            num_episodes = len(done_indices)

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
        return self.buffer_size if self.is_full else self.position

    def clear(self):
        self.position = 0
        self.is_full = False
        logger.info("Replay buffer cleared")

    def get_statistics(self) -> Dict:
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
