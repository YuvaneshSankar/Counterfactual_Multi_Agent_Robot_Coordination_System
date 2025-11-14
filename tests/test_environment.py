"""
Test Environment - Unit Tests for Warehouse Environment

Tests environment components, physics, and task generation.
"""

import unittest
import numpy as np
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.warehouse_env import WarehouseEnv
from src.environments.robot import Robot
from src.environments.task_generator import TaskGenerator
from src.environments.collision_checker import CollisionChecker
from src.environments.warehouse_layout import WarehouseLayout


class TestWarehouseEnv(unittest.TestCase):
    """Test warehouse environment."""

    def setUp(self):
        """Setup test environment."""
        self.config = {
            'environment': {
                'num_robots': 3,
                'warehouse_size': {'width': 50, 'height': 40},
                'max_episode_steps': 100,
                'task_arrival_rate': 0.2,
                'robot': {
                    'radius': 0.5,
                    'max_linear_velocity': 2.0,
                    'max_angular_velocity': 3.14,
                    'battery_capacity': 100.0,
                },
            },
            'rewards': {
                'task_completion': 10.0,
                'time_step_penalty': -0.01,
                'collision': -5.0,
            }
        }
        self.env = WarehouseEnv(config=self.config, render=False)

    def tearDown(self):
        """Cleanup after tests."""
        self.env.close()

    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.num_robots, 3)
        self.assertEqual(len(self.env.robots), 3)

    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()

        self.assertIsInstance(obs, list)
        self.assertEqual(len(obs), self.env.num_robots)
        self.assertIsInstance(info, dict)

    def test_step(self):
        """Test environment step."""
        self.env.reset()

        # Random actions
        actions = np.random.randn(self.env.num_robots, 2)
        obs, reward, done, info = self.env.step(actions)

        self.assertIsInstance(obs, list)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_action_space(self):
        """Test action space dimensions."""
        self.assertEqual(self.env.action_space.shape, (3, 2))

    def test_observation_space(self):
        """Test observation space."""
        self.assertIsNotNone(self.env.observation_space)

    def test_global_state(self):
        """Test global state computation."""
        self.env.reset()
        state = self.env.get_global_state()

        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state.shape), 1)


class TestRobot(unittest.TestCase):
    """Test robot class."""

    def setUp(self):
        """Setup test robot."""
        import pybullet as p
        self.client = p.connect(p.DIRECT)

        self.config = {
            'robot': {
                'radius': 0.5,
                'mass': 50.0,
                'max_linear_velocity': 2.0,
                'max_angular_velocity': 3.14,
                'battery_capacity': 100.0,
                'battery_discharge_rate': 0.1,
                'battery_charge_rate': 0.5,
            }
        }

        self.robot = Robot(
            robot_id=0,
            initial_position=(10, 10, 0),
            config=self.config,
            client=self.client
        )

    def tearDown(self):
        """Cleanup after tests."""
        import pybullet as p
        p.disconnect(self.client)

    def test_robot_initialization(self):
        """Test robot initializes correctly."""
        self.assertEqual(self.robot.robot_id, 0)
        self.assertEqual(self.robot.battery_level, 100.0)
        self.assertFalse(self.robot.has_task)

    def test_set_velocity(self):
        """Test velocity setting."""
        self.robot.set_velocity(1.0, 0.5)
        self.assertEqual(self.robot.command_velocity[0], 1.0)
        self.assertEqual(self.robot.command_velocity[1], 0.5)

    def test_battery_discharge(self):
        """Test battery discharges during movement."""
        initial_battery = self.robot.battery_level

        self.robot.set_velocity(2.0, 0)
        for _ in range(100):
            self.robot.update()

        self.assertLess(self.robot.battery_level, initial_battery)

    def test_task_assignment(self):
        """Test task assignment."""
        task = (5, 5, 8, 20, 20, 500)
        self.robot.assign_task(task)

        self.assertTrue(self.robot.has_task)
        self.assertEqual(self.robot.current_task, task)

    def test_can_accept_task(self):
        """Test can_accept_task logic."""
        self.assertTrue(self.robot.can_accept_task())

        # Assign task
        self.robot.assign_task((5, 5, 8, 20, 20, 500))
        self.assertFalse(self.robot.can_accept_task())


class TestTaskGenerator(unittest.TestCase):
    """Test task generator."""

    def setUp(self):
        """Setup task generator."""
        self.generator = TaskGenerator(
            warehouse_width=100,
            warehouse_height=80,
            arrival_rate=0.5,
            max_queue=20
        )

    def test_initialization(self):
        """Test generator initializes."""
        self.assertEqual(len(self.generator.pending_tasks), 0)
        self.assertEqual(len(self.generator.completed_tasks), 0)

    def test_task_generation(self):
        """Test task generation."""
        for _ in range(100):
            self.generator.step()

        # Should have generated some tasks
        self.assertGreater(self.generator.task_counter, 0)

    def test_task_format(self):
        """Test generated task format."""
        self.generator.step()

        if self.generator.pending_tasks:
            task = self.generator.pending_tasks[0]
            self.assertEqual(len(task), 6)  # (pickup_x, pickup_y, priority, delivery_x, delivery_y, deadline)

    def test_get_nearest_task(self):
        """Test get nearest task."""
        # Add tasks manually
        self.generator.add_task((10, 10, 5, 50, 50, 500))
        self.generator.add_task((80, 80, 5, 20, 20, 500))

        # Query near first task
        nearest = self.generator.get_nearest_task((15, 15))
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest[0], 10)


class TestCollisionChecker(unittest.TestCase):
    """Test collision checker."""

    def setUp(self):
        """Setup collision checker."""
        self.checker = CollisionChecker(robot_radius=0.5)

    def test_initialization(self):
        """Test checker initializes."""
        self.assertIsNotNone(self.checker)

    def test_robot_collision_detection(self):
        """Test robot collision detection."""
        import pybullet as p
        client = p.connect(p.DIRECT)

        config = {'robot': {'radius': 0.5, 'mass': 50.0}}

        # Create two robots close together
        robot1 = Robot(0, (10, 10, 0), config, client)
        robot2 = Robot(1, (10.5, 10, 0), config, client)  # Very close

        # Should detect collision
        collision = self.checker.check_robot_collision(robot1, robot2)
        self.assertTrue(collision)

        p.disconnect(client)

    def test_no_collision_when_far(self):
        """Test no collision when robots are far."""
        import pybullet as p
        client = p.connect(p.DIRECT)

        config = {'robot': {'radius': 0.5, 'mass': 50.0}}

        robot1 = Robot(0, (10, 10, 0), config, client)
        robot2 = Robot(1, (50, 50, 0), config, client)  # Far apart

        collision = self.checker.check_robot_collision(robot1, robot2)
        self.assertFalse(collision)

        p.disconnect(client)


class TestWarehouseLayout(unittest.TestCase):
    """Test warehouse layout."""

    def setUp(self):
        """Setup layout."""
        import pybullet as p
        self.client = p.connect(p.DIRECT)

        self.layout = WarehouseLayout(
            width=100,
            height=80,
            num_shelves=5,
            num_charging_stations=2,
            client=self.client
        )

    def tearDown(self):
        """Cleanup."""
        import pybullet as p
        p.disconnect(self.client)

    def test_layout_creation(self):
        """Test layout creates shelves."""
        self.assertEqual(len(self.layout.shelves), 5)
        self.assertEqual(len(self.layout.charging_stations), 2)

    def test_valid_position_check(self):
        """Test valid position checking."""
        # Center should be valid
        self.assertTrue(self.layout.is_valid_position((50, 40)))

        # Near wall should be invalid
        self.assertFalse(self.layout.is_valid_position((1, 40)))


if __name__ == '__main__':
    unittest.main()
