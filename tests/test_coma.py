"""
Test COMA Algorithm - Unit Tests for COMA Components

Tests actor, critic, replay buffer, and algorithm.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.coma_continuous import COMAcontinuous
from src.algorithms.actor_network import ActorNetwork
from src.algorithms.critic_network import CriticNetwork
from src.algorithms.replay_buffer import ReplayBuffer


class TestActorNetwork(unittest.TestCase):
    """Test actor network."""

    def setUp(self):
        """Setup actor."""
        self.obs_dim = 20
        self.action_dim = 2
        self.hidden_dim = 64

        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )

    def test_initialization(self):
        """Test network initializes."""
        self.assertIsNotNone(self.actor)

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 32
        obs = torch.randn(batch_size, self.obs_dim)

        mean, log_std = self.actor(obs)

        self.assertEqual(mean.shape, (batch_size, self.action_dim))
        self.assertEqual(log_std.shape, (batch_size, self.action_dim))

    def test_sample_action(self):
        """Test action sampling."""
        obs = torch.randn(1, self.obs_dim)

        action, log_prob = self.actor.sample_action(obs)

        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertEqual(log_prob.shape, (1,))

    def test_log_prob_computation(self):
        """Test log probability computation."""
        obs = torch.randn(10, self.obs_dim)
        actions = torch.randn(10, self.action_dim)

        log_probs = self.actor.log_prob(obs, actions)

        self.assertEqual(log_probs.shape, (10,))

    def test_deterministic_action(self):
        """Test deterministic action selection."""
        obs = torch.randn(1, self.obs_dim)

        action1 = self.actor.get_action(obs, deterministic=True)
        action2 = self.actor.get_action(obs, deterministic=True)

        # Should be same
        self.assertTrue(torch.allclose(action1, action2))


class TestCriticNetwork(unittest.TestCase):
    """Test critic network."""

    def setUp(self):
        """Setup critic."""
        self.state_dim = 100
        self.action_dim = 2
        self.num_agents = 5
        self.hidden_dim = 128

        self.critic = CriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_dim=self.hidden_dim
        )

    def test_initialization(self):
        """Test critic initializes."""
        self.assertIsNotNone(self.critic)

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.num_agents, self.action_dim)

        q_values = self.critic(state, actions)

        self.assertEqual(q_values.shape, (batch_size, self.num_agents))

    def test_counterfactual_baseline(self):
        """Test counterfactual baseline computation."""
        batch_size = 16
        state = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.num_agents, self.action_dim)
        agent_id = 0

        baseline = self.critic.compute_counterfactual_baseline(
            state, actions, agent_id
        )

        self.assertEqual(baseline.shape, (batch_size,))


class TestReplayBuffer(unittest.TestCase):
    """Test replay buffer."""

    def setUp(self):
        """Setup buffer."""
        self.capacity = 1000
        self.state_dim = 50
        self.obs_dim = 20
        self.num_agents = 3
        self.action_dim = 2

        self.buffer = ReplayBuffer(
            capacity=self.capacity,
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_dim=self.action_dim
        )

    def test_initialization(self):
        """Test buffer initializes."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.capacity, self.capacity)

    def test_add_experience(self):
        """Test adding experience."""
        state = np.random.randn(self.state_dim)
        obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
        actions = np.random.randn(self.num_agents, self.action_dim)
        reward = 1.0
        next_state = np.random.randn(self.state_dim)
        next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
        done = False

        self.buffer.add(state, obs, actions, reward, next_state, next_obs, done)

        self.assertEqual(len(self.buffer), 1)

    def test_sample(self):
        """Test sampling from buffer."""
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(self.state_dim)
            obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            actions = np.random.randn(self.num_agents, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            done = False

            self.buffer.add(state, obs, actions, reward, next_state, next_obs, done)

        # Sample
        batch = self.buffer.sample(32)

        self.assertIn('states', batch)
        self.assertIn('actions', batch)
        self.assertIn('rewards', batch)
        self.assertEqual(batch['states'].shape[0], 32)

    def test_buffer_overflow(self):
        """Test buffer handles overflow."""
        # Add more than capacity
        for i in range(self.capacity + 100):
            state = np.random.randn(self.state_dim)
            obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            actions = np.random.randn(self.num_agents, self.action_dim)
            reward = 1.0
            next_state = np.random.randn(self.state_dim)
            next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            done = False

            self.buffer.add(state, obs, actions, reward, next_state, next_obs, done)

        # Should not exceed capacity
        self.assertEqual(len(self.buffer), self.capacity)


class TestCOMAAlgorithm(unittest.TestCase):
    """Test COMA algorithm."""

    def setUp(self):
        """Setup COMA."""
        self.state_dim = 50
        self.obs_dim = 20
        self.action_dim = 2
        self.num_agents = 3

        self.config = {
            'algorithm': {
                'actor_lr': 3e-4,
                'critic_lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'buffer_size': 10000,
            }
        }

        self.device = torch.device('cpu')

        self.coma = COMAcontinuous(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            config=self.config,
            device=self.device
        )

    def test_initialization(self):
        """Test COMA initializes."""
        self.assertIsNotNone(self.coma)
        self.assertEqual(self.coma.num_agents, self.num_agents)

    def test_select_actions(self):
        """Test action selection."""
        obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]

        actions, info = self.coma.select_actions(obs, deterministic=False)

        self.assertEqual(len(actions), self.num_agents)
        self.assertIsInstance(info, dict)

    def test_update(self):
        """Test algorithm update."""
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(self.state_dim)
            obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            actions = np.random.randn(self.num_agents, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim)
            next_obs = [np.random.randn(self.obs_dim) for _ in range(self.num_agents)]
            done = False

            self.coma.replay_buffer.add(
                state, obs, actions, reward, next_state, next_obs, done
            )

        # Sample and update
        batch = self.coma.replay_buffer.sample(32)

        update_info = self.coma.update(
            states=batch['states'],
            observations=batch['observations'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            next_states=batch['next_states'],
            next_observations=batch['next_observations'],
            dones=batch['dones']
        )

        self.assertIn('actor_loss', update_info)
        self.assertIn('critic_loss', update_info)

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        import tempfile
        import os

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            checkpoint_path = f.name

        try:
            # Save
            self.coma.save_checkpoint(checkpoint_path)

            # Load
            self.coma.load_checkpoint(checkpoint_path)

            # Should load without error
            self.assertTrue(True)

        finally:
            # Cleanup
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


if __name__ == '__main__':
    unittest.main()
