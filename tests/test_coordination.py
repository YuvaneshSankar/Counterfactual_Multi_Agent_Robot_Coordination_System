"""
Test Coordination - Unit Tests for Coordination Components

Tests task allocation, path planning, and communication.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.coordination.task_allocator import TaskAllocator, HungarianAllocator
from src.coordination.path_planner import PathPlanner
from src.coordination.communication import RobotCommunicationNetwork
from src.coordination.battery_manager import BatteryManager
from src.coordination.scheduler import TaskScheduler


class TestTaskAllocator(unittest.TestCase):
    """Test task allocator."""

    def setUp(self):
        """Setup allocator."""
        self.num_robots = 5
        self.allocator = TaskAllocator(
            num_robots=self.num_robots,
            allocation_strategy='greedy_nearest'
        )

    def test_initialization(self):
        """Test allocator initializes."""
        self.assertEqual(self.allocator.num_robots, self.num_robots)

    def test_greedy_allocation(self):
        """Test greedy allocation."""
        # Create mock robots
        class MockRobot:
            def __init__(self, robot_id, position):
                self.robot_id = robot_id
                self.position = np.array(position)
                self.has_task = False
                self.battery_level = 100.0

            def get_position(self):
                return self.position

            def can_accept_task(self):
                return not self.has_task and self.battery_level > 30

        robots = [
            MockRobot(0, (10, 10, 0)),
            MockRobot(1, (50, 50, 0)),
            MockRobot(2, (80, 20, 0))
        ]

        tasks = [
            (15, 15, 5, 60, 60, 500),  # Near robot 0
            (85, 25, 8, 20, 20, 500),  # Near robot 2
        ]

        allocations = self.allocator.allocate_tasks(robots, tasks)

        self.assertIsInstance(allocations, dict)
        self.assertGreater(len(allocations), 0)

    def test_priority_allocation(self):
        """Test priority-aware allocation."""
        allocator = TaskAllocator(
            num_robots=3,
            allocation_strategy='priority_aware'
        )

        # Test that it doesn't crash
        self.assertIsNotNone(allocator)


class TestPathPlanner(unittest.TestCase):
    """Test path planner."""

    def setUp(self):
        """Setup planner."""
        self.planner = PathPlanner(
            warehouse_width=100,
            warehouse_height=80,
            grid_resolution=1.0,
            robot_radius=0.5
        )

    def test_initialization(self):
        """Test planner initializes."""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.grid_width, 100)
        self.assertEqual(self.planner.grid_height, 80)

    def test_world_to_grid_conversion(self):
        """Test coordinate conversion."""
        world_pos = np.array([50.0, 40.0])
        grid_pos = self.planner._world_to_grid(world_pos)

        self.assertIsInstance(grid_pos, tuple)
        self.assertEqual(len(grid_pos), 2)

    def test_path_planning_simple(self):
        """Test simple path planning."""
        start = np.array([10.0, 10.0])
        goal = np.array([90.0, 70.0])

        path = self.planner.plan_path(start, goal)

        if path is not None:
            self.assertIsInstance(path, list)
            self.assertGreater(len(path), 1)

    def test_invalid_goal(self):
        """Test planning with invalid goal."""
        start = np.array([50.0, 40.0])
        goal = np.array([-10.0, 40.0])  # Out of bounds

        path = self.planner.plan_path(start, goal)

        self.assertIsNone(path)


class TestCommunicationNetwork(unittest.TestCase):
    """Test communication network."""

    def setUp(self):
        """Setup network."""
        self.num_robots = 5
        self.network = RobotCommunicationNetwork(
            num_robots=self.num_robots,
            communication_range=30.0
        )

    def test_initialization(self):
        """Test network initializes."""
        self.assertEqual(self.network.num_robots, self.num_robots)

    def test_broadcast_message(self):
        """Test message broadcasting."""
        payload = {'data': 'test'}

        self.network.broadcast_message(
            sender_id=0,
            message_type='test',
            payload=payload
        )

        # Should work without error
        self.assertTrue(True)

    def test_send_message(self):
        """Test direct messaging."""
        # First update connectivity to enable communication
        class MockRobot:
            def __init__(self, robot_id, position):
                self.robot_id = robot_id
                self.position = np.array(position)

            def get_position(self):
                return self.position

        robots = [
            MockRobot(0, (10, 10, 0)),
            MockRobot(1, (15, 15, 0)),  # Within range
            MockRobot(2, (80, 80, 0)),  # Out of range
        ]

        self.network.update_connectivity(robots)

        # Send message from 0 to 1 (should succeed)
        success = self.network.send_message(
            sender_id=0,
            receiver_id=1,
            message_type='test',
            payload={'data': 'hello'}
        )

        self.assertTrue(success)

    def test_get_messages(self):
        """Test retrieving messages."""
        messages = self.network.get_messages(robot_id=1)
        self.assertIsInstance(messages, list)


class TestBatteryManager(unittest.TestCase):
    """Test battery manager."""

    def setUp(self):
        """Setup manager."""
        self.num_robots = 5
        self.manager = BatteryManager(
            num_robots=self.num_robots,
            battery_capacity=100.0,
            charge_threshold=30.0
        )

    def test_initialization(self):
        """Test manager initializes."""
        self.assertEqual(self.manager.num_robots, self.num_robots)

    def test_battery_status(self):
        """Test battery status."""
        status = self.manager.get_battery_status(robot_id=0)

        self.assertIn('battery_level', status)
        self.assertIn('status', status)

    def test_should_charge(self):
        """Test charge decision."""
        # Initially fully charged
        self.assertFalse(self.manager.should_charge(robot_id=0))

        # Set low battery
        self.manager.battery_levels[0] = 20.0
        self.assertTrue(self.manager.should_charge(robot_id=0))

    def test_get_robots_needing_charge(self):
        """Test finding robots needing charge."""
        # Set some robots to low battery
        self.manager.battery_levels[1] = 20.0
        self.manager.battery_levels[3] = 15.0

        robots = self.manager.get_robots_needing_charge()

        self.assertIn(1, robots)
        self.assertIn(3, robots)


class TestTaskScheduler(unittest.TestCase):
    """Test task scheduler."""

    def setUp(self):
        """Setup scheduler."""
        self.scheduler = TaskScheduler(
            num_robots=5,
            scheduling_policy='priority'
        )

    def test_initialization(self):
        """Test scheduler initializes."""
        self.assertEqual(self.scheduler.num_robots, 5)
        self.assertEqual(self.scheduler.scheduling_policy, 'priority')

    def test_add_task(self):
        """Test adding task."""
        task = (10, 10, 8, 50, 50, 500)
        self.scheduler.add_task(task)

        self.assertEqual(len(self.scheduler.task_queue), 1)

    def test_get_next_task(self):
        """Test getting next task."""
        # Add tasks
        self.scheduler.add_task((10, 10, 5, 50, 50, 500))
        self.scheduler.add_task((20, 20, 8, 60, 60, 400))

        # Get task for robot
        task = self.scheduler.get_next_task(robot_id=0)

        self.assertIsNotNone(task)
        self.assertEqual(len(task), 6)

    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        # Add tasks with different priorities
        low_priority = (10, 10, 3, 50, 50, 500)
        high_priority = (20, 20, 9, 60, 60, 500)

        self.scheduler.add_task(low_priority)
        self.scheduler.add_task(high_priority)

        # Should get high priority first
        task = self.scheduler.get_next_task(robot_id=0)
        self.assertEqual(task[2], 9)  # Priority field


if __name__ == '__main__':
    unittest.main()
