
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)


class COMATrainer:


    def __init__(
        self,
        config: Dict,
        env,
        algorithm,
        device: torch.device = torch.device('cpu'),
    ):

        self.config = config
        self.env = env
        self.algorithm = algorithm
        self.device = device

        training_config = config.get('training', {})
        self.total_timesteps = training_config.get('total_timesteps', 500000)
        self.num_envs = training_config.get('num_envs', 4)
        self.rollout_steps = training_config.get('rollout_steps', 2000)
        self.batch_size = training_config.get('batch_size', 64)
        self.num_epochs = training_config.get('num_epochs', 10)
        self.eval_frequency = training_config.get('eval_frequency', 5000)
        self.checkpoint_frequency = training_config.get('checkpoint_frequency', 10000)

        logging_config = config.get('logging', {})
        self.log_dir = logging_config.get('log_dir', 'results/logs')
        self.checkpoint_dir = logging_config.get('checkpoint_dir', 'results/checkpoints')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -np.inf

        self.trajectories: List = []

        logger.info("COMATrainer initialized")

    def train(self) -> Dict:

        logger.info(f"Starting training for {self.total_timesteps} timesteps")

        stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
        }

        with tqdm(total=self.total_timesteps, desc='Training') as pbar:
            while self.total_steps < self.total_timesteps:
                rollout_stats = self._collect_rollout()


                self.total_steps += self.rollout_steps
                self.total_episodes += rollout_stats['episodes']
                pbar.update(self.rollout_steps)

                stats['total_steps'] = self.total_steps
                stats['total_episodes'] = self.total_episodes
                stats['episode_rewards'].append(np.mean(rollout_stats['rewards']))
                stats['actor_losses'].append(update_stats['actor_loss'])
                stats['critic_losses'].append(update_stats['critic_loss'])

                if self.total_steps % self.eval_frequency == 0:
                    eval_reward = self._evaluate()
                    logger.info(f"Evaluation reward at step {self.total_steps}: {eval_reward:.2f}")

                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self._save_checkpoint('best_model')

                if self.total_steps % self.checkpoint_frequency == 0:
                    self._save_checkpoint(f'model_{self.total_steps}')

        logger.info(f"Training completed! Best reward: {self.best_reward:.2f}")
        return stats

    def _collect_rollout(self) -> Dict:
        """
        Collect trajectory rollout from environment.

        Returns:
            Rollout statistics
        """
        states = []
        observations = []
        actions = []
        rewards = []
        next_states = []
        next_observations = []
        dones = []

        episode_rewards = []
        episodes_collected = 0

        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(self.rollout_steps):
            state = self.env.get_global_state()

            action_list, policy_info = self.algorithm.select_actions(obs, deterministic=False)
            actions_array = np.array(action_list)

            obs_next, reward, done, info = self.env.step(actions_array)

            state_next = self.env.get_global_state()

            states.append(state)
            observations.append(obs)
            actions.append(actions_array)
            rewards.append(reward)
            next_states.append(state_next)
            next_observations.append(obs_next)
            dones.append(done)

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episodes_collected += 1

                #
                obs, _ = self.env.reset()
                episode_reward = 0.0
            else:
                obs = obs_next

        if episode_reward > 0:
            episode_rewards.append(episode_reward)
            episodes_collected += 1

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.BoolTensor(np.array(dones)).to(self.device)

=        for i in range(len(states)):
            self.algorithm.replay_buffer.add(
                states[i],
                observations[i],
                actions[i],
                rewards[i],
                next_states[i],
                next_observations[i],
                dones[i]
            )

        return {
            'episodes': episodes_collected,
            'rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        }

    def _update_networks(self) -> Dict:
=
        batch = self.algorithm.replay_buffer.sample(self.batch_size)

=        update_info = self.algorithm.update(
            states=batch['states'],
            observations=batch['observations'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            next_states=batch['next_states'],
            next_observations=batch['next_observations'],
            dones=batch['dones'],
            num_epochs=self.num_epochs
        )

        return update_info

    def _evaluate(self, num_episodes: int = 10) -> float:
=
        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
=                action_list, _ = self.algorithm.select_actions(obs, deterministic=True)
                actions_array = np.array(action_list)

=                obs, reward, done, info = self.env.step(actions_array)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        return mean_reward

    def _save_checkpoint(self, name: str):
=
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        self.algorithm.save_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
=
        self.algorithm.load_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def get_training_info(self) -> Dict:
=        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'buffer_size': len(self.algorithm.replay_buffer),
        }
