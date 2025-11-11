"""
Warehouse Environment - PyBullet Physics Simulation

Core environment implementation for multi-agent warehouse robot coordination.
Provides a Gymnasium-compatible interface with physics simulation using PyBullet.

Key Features:
- Multiple robots with differential drive kinematics
- Dynamic task generation with priorities
- Physics-based collision detection
- Realistic battery management
- Configurable warehouse layouts
- Real-time rendering support
"""

import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import logging
from typing import Tuple, Dict, List, Optional
import os

logger = logging.getLogger(__name__)


class WarehouseEnv(gym.Env):
    """
    Multi-Agent Warehouse Environment.

    Simulates autonomous mobile robots (AMRs) in a warehouse environment
    performing pickup and delivery tasks.

    State Space:
    - Global state: positions, velocities, battery levels of all robots
    - Tasks: locations, priorities, deadlines

    Action Space (per robot):
    - Linear velocity: [-1.0, 1.0] m/s
    - Angular velocity: [-π/2, π/2] rad/s

    Reward:
    - Task completion: +10
    - Time penalty: -0.01 per step
    - Collision: -5 (robot-robot), -2 (obstacle)
    - Battery depletion: -10
    - Charging: +1
    """

    def __init__(self, config: Dict, render: bool = False):
        """
        Initialize warehouse environment.

        Args:
            config: Configuration dictionary with environment parameters
            render: Whether to enable PyBullet GUI
        """
        self.config = config
        self.render_mode = "human" if render else None

        # Extract configuration
        env_config = config.get('environment', {})
        self.num_robots = env_config.get('num_robots', 5)
        self.warehouse_width = env_config.get('warehouse_size', {}).get('width', 100)
        self.warehouse_height = env_config.get('warehouse_size', {}).get('height', 80)
        self.max_episode_steps = env_config.get('max_episode_steps', 2000)
        self.task_arrival_rate = env_config.get('task_arrival_rate', 0.5)
        self.max_tasks_queue = env_config.get('max_tasks_queue', 20)

        # Import environment components
        from .robot import Robot
        from .task_generator import TaskGenerator
        from .warehouse_layout import WarehouseLayout
        from .collision_checker import CollisionChecker
        from .sensors import LiDARSensor

        self.Robot = Robot
        self.TaskGenerator = TaskGenerator
        self.WarehouseLayout = WarehouseLayout
        self.CollisionChecker = CollisionChecker
        self.LiDARSensor = LiDARSensor

        # Physics client
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physics parameters
        physics_config = env_config.get('physics', {})
        gravity = physics_config.get('gravity', 9.81)
        p.setGravity(0, 0, -gravity, physicsClientId=self.client)

        # Ground plane
        self.ground_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, 0],
            physicsClientId=self.client
        )

        # Initialize environment components
        self.warehouse_layout = None
        self.robots: List[Robot] = []
        self.task_generator = None
        self.collision_checker = None
        self.lidar_sensors = []

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.completed_tasks = 0
        self.collisions = 0

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_robots, 2),  # [linear_vel, angular_vel] per robot
            dtype=np.float32
        )

        # Observation space (for each robot)
        lidar_dim = env_config.get('lidar', {}).get('num_beams', 16)
        obs_per_robot = lidar_dim + 10  # LiDAR + position + velocity + battery + task info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_per_robot,),
            dtype=np.float32
        )

        # Global state space
        state_dim = (
            self.num_robots * 8 +  # 8 features per robot (x, y, vx, vy, theta, battery, task_assigned, idle)
            self.max_tasks_queue * 3  # 3 features per task (x, y, priority)
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        logger.info(f"WarehouseEnv initialized: {self.num_robots} robots")

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Returns:
            observations: List of observations for each agent
            info: Additional information
        """
        self.current_step = 0
        self.episode_reward = 0.0
        self.completed_tasks = 0
        self.collisions = 0

        # Create warehouse layout
        self.warehouse_layout = self.WarehouseLayout(
            width=self.warehouse_width,
            height=self.warehouse_height,
            num_shelves=self.config.get('environment', {}).get('num_shelves', 8),
            num_charging_stations=self.config.get('environment', {}).get('num_charging_stations', 2),
            client=self.client
        )
        self.warehouse_layout.build()

        # Create robots
        self.robots = []
        for i in range(self.num_robots):
            x = np.random.uniform(5, self.warehouse_width - 5)
            y = np.random.uniform(5, self.warehouse_height - 5)

            robot = self.Robot(
                robot_id=i,
                initial_position=(x, y, 0),
                config=self.config.get('environment', {}),
                client=self.client
            )
            self.robots.append(robot)

        # Create task generator
        self.task_generator = self.TaskGenerator(
            warehouse_width=self.warehouse_width,
            warehouse_height=self.warehouse_height,
            arrival_rate=self.task_arrival_rate,
            max_queue=self.max_tasks_queue
        )

        # Create collision checker
        self.collision_checker = self.CollisionChecker(
            robot_radius=self.config.get('environment', {}).get('robot', {}).get('radius', 0.5),
            warehouse_layout=self.warehouse_layout,
            client=self.client
        )

        # Create LiDAR sensors
        self.lidar_sensors = [
            self.LiDARSensor(robot=robot, config=self.config.get('environment', {}).get('lidar', {}))
            for robot in self.robots
        ]

        # Get initial observations
        observations = self._get_observations()

        return observations, {}

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], float, bool, Dict]:
        """
        Execute one environment step.

        Args:
            actions: Joint actions [num_robots, 2] with [linear_vel, angular_vel]

        Returns:
            observations: Updated observations
            reward: Scalar reward (shared across agents)
            done: Episode termination flag
            info: Additional information
        """
        self.current_step += 1

        # Execute actions
        for i, robot in enumerate(self.robots):
            linear_vel = actions[i, 0] * self.config.get('environment', {}).get('robot', {}).get('max_linear_velocity', 2.0)
            angular_vel = actions[i, 1] * self.config.get('environment', {}).get('robot', {}).get('max_angular_velocity', 3.14)
            robot.set_velocity(linear_vel, angular_vel)

        # Step physics simulation
        p.stepSimulation(physicsClientId=self.client)

        # Update robot states
        for robot in self.robots:
            robot.update()

        # Generate new tasks
        self.task_generator.step()

        # Allocate tasks to robots
        self._allocate_tasks()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self.current_step >= self.max_episode_steps

        # Check collisions
        self._check_collisions()

        # Get observations
        observations = self._get_observations()

        info = {
            'episode_step': self.current_step,
            'completed_tasks': self.completed_tasks,
            'collisions': self.collisions,
            'active_tasks': len(self.task_generator.pending_tasks),
        }

        self.episode_reward += reward

        return observations, reward, done, info

    def _get_observations(self) -> List[np.ndarray]:
        """
        Get observations for each robot.

        Returns:
            observations: List of observation arrays [num_robots, obs_dim]
        """
        observations = []

        for i, robot in enumerate(self.robots):
            # Get LiDAR readings
            lidar_readings = self.lidar_sensors[i].get_readings()

            # Get robot state
            pos = robot.get_position()
            vel = robot.get_velocity()
            battery = robot.battery_level / robot.config.get('battery_capacity', 100.0)

            # Get nearest task info
            if self.task_generator.pending_tasks:
                nearest_task = self._get_nearest_task(pos)
                task_x = nearest_task[0] - pos[0]
                task_y = nearest_task[1] - pos[1]
                task_priority = nearest_task[2] / 10.0  # Normalize priority
            else:
                task_x = task_y = task_priority = 0.0

            # Combine into observation
            obs = np.concatenate([
                lidar_readings,
                [pos[0] / self.warehouse_width, pos[1] / self.warehouse_height],
                [vel[0], vel[1], vel[2]],
                [battery, task_x, task_y, task_priority]
            ]).astype(np.float32)

            observations.append(obs)

        return observations

    def _compute_reward(self) -> float:
        """
        Compute shared reward for all agents.

        Returns:
            reward: Scalar reward value
        """
        reward_config = self.config.get('rewards', {})
        reward = 0.0

        # Time penalty
        reward += reward_config.get('time_step_penalty', -0.01)

        # Check for task completions
        for robot in self.robots:
            if robot.completed_task:
                reward += reward_config.get('task_completion', 10.0)
                self.completed_tasks += 1
                robot.completed_task = False

        # Battery penalties
        for robot in self.robots:
            if robot.battery_level <= 0:
                reward += reward_config.get('battery_depleted', -10.0)
            elif robot.battery_level < 30 and not robot.charging:
                reward += reward_config.get('idle_penalty', -0.05)

        # Collision penalties handled in _check_collisions()

        return reward

    def _check_collisions(self):
        """Check and handle collisions."""
        reward_config = self.config.get('rewards', {})

        # Robot-robot collisions
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                if self.collision_checker.check_robot_collision(self.robots[i], self.robots[j]):
                    self.collisions += 1
                    # Reward penalty handled in _compute_reward()

        # Robot-obstacle collisions
        for robot in self.robots:
            if self.collision_checker.check_obstacle_collision(robot, self.warehouse_layout):
                self.collisions += 1

    def _allocate_tasks(self):
        """Allocate pending tasks to available robots."""
        available_robots = [r for r in self.robots if not r.has_task and r.battery_level > 20]

        for robot in available_robots:
            if self.task_generator.pending_tasks:
                task = self.task_generator.pending_tasks.pop(0)
                robot.assign_task(task)

    def _get_nearest_task(self, position: Tuple) -> Tuple:
        """Get nearest pending task to a position."""
        if not self.task_generator.pending_tasks:
            return (0, 0, 0)

        nearest = min(
            self.task_generator.pending_tasks,
            key=lambda t: np.linalg.norm(np.array(position[:2]) - np.array(t[:2]))
        )
        return nearest

    def render(self, mode: str = 'human'):
        """Render environment."""
        if mode == 'human' and self.render_mode:
            p.stepSimulation(physicsClientId=self.client)

    def close(self):
        """Close the environment."""
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)

    def get_global_state(self) -> np.ndarray:
        """
        Get full global state (for centralized training).

        Returns:
            state: Flattened global state vector
        """
        state_parts = []

        # Robot states
        for robot in self.robots:
            pos = robot.get_position()
            vel = robot.get_velocity()
            battery = robot.battery_level / 100.0
            task_assigned = 1.0 if robot.has_task else 0.0
            idle = 0.0 if robot.has_task else 1.0

            state_parts.extend([
                pos[0] / self.warehouse_width,
                pos[1] / self.warehouse_height,
                vel[0], vel[1], vel[2],
                battery,
                task_assigned,
                idle
            ])

        # Task states
        for i in range(self.max_tasks_queue):
            if i < len(self.task_generator.pending_tasks):
                task = self.task_generator.pending_tasks[i]
                state_parts.extend([
                    task[0] / self.warehouse_width,
                    task[1] / self.warehouse_height,
                    task[2] / 10.0  # Priority
                ])
            else:
                state_parts.extend([0.0, 0.0, 0.0])

        return np.array(state_parts, dtype=np.float32)

    def get_info(self) -> Dict:
        """Get environment information."""
        return {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'completed_tasks': self.completed_tasks,
            'collisions': self.collisions,
            'active_tasks': len(self.task_generator.pending_tasks),
            'num_robots': self.num_robots,
            'warehouse_width': self.warehouse_width,
            'warehouse_height': self.warehouse_height,
        }
