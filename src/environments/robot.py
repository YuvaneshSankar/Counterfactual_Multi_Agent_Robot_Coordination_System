

import numpy as np
import pybullet as p
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Robot:

    def __init__(
        self,
        robot_id: int,
        initial_position: Tuple[float, float, float],
        config: Dict,
        client: int,
    ):

        self.robot_id = robot_id
        self.client = client
        self.config = config

        # Physical parameters
        self.radius = config.get('robot', {}).get('radius', 0.5)
        self.mass = config.get('robot', {}).get('mass', 50.0)
        self.max_linear_velocity = config.get('robot', {}).get('max_linear_velocity', 2.0)
        self.max_angular_velocity = config.get('robot', {}).get('max_angular_velocity', 3.14)

        # Battery parameters
        self.battery_capacity = config.get('robot', {}).get('battery_capacity', 100.0)
        self.battery_level = self.battery_capacity
        self.battery_discharge_rate = config.get('robot', {}).get('battery_discharge_rate', 0.1)
        self.battery_charge_rate = config.get('robot', {}).get('battery_charge_rate', 0.5)
        self.battery_charge_threshold = config.get('robot', {}).get('battery_charge_threshold', 30.0)

        # State
        self.position = np.array(initial_position, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, omega]
        self.command_velocity = np.array([0.0, 0.0], dtype=np.float32)  # [linear, angular]

        # Create robot body in PyBullet
        self.body_id = self._create_body(initial_position)

        # Task tracking
        self.current_task = None
        self.has_task = False
        self.task_progress = 0.0
        self.completed_task = False
        self.at_pickup = False
        self.at_delivery = False

        # Charging state
        self.charging = False
        self.charging_station_id = None

        # History for filtering
        self.velocity_history = []
        self.max_history = 5

        logger.info(f"Robot {robot_id} created at position {initial_position}")

    def _create_body(self, position: Tuple[float, float, float]) -> int:

        # Create a cylinder to represent the robot
        shape_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            height=0.3,
            physicsClientId=self.client
        )

        visual_shape_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            length=0.3,
            rgbaColor=[0.2, 0.6, 0.9, 1.0],
            physicsClientId=self.client
        )

        body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[position[0], position[1], 0.15],
            physicsClientId=self.client
        )

        # Disable default damping
        p.changeDynamics(
            body_id,
            -1,
            linearDamping=0.5,
            angularDamping=0.5,
            physicsClientId=self.client
        )

        return body_id

    def set_velocity(self, linear_vel: float, angular_vel: float):

        # Clamp to limits
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)

        self.command_velocity = np.array([linear_vel, angular_vel], dtype=np.float32)

    def update(self):

        # Get current position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
        pos = np.array(pos[:2])  # Only x, y

        # Convert quaternion to yaw angle
        yaw = p.getEulerFromQuaternion(orn)[2]

        # Differential drive kinematics
        # For a robot with wheelbase L and wheel velocities v_left, v_right:
        # v = (v_left + v_right) / 2
        # omega = (v_right - v_left) / L

        # Since we command linear and angular velocity directly:
        linear_vel = self.command_velocity[0]
        angular_vel = self.command_velocity[1]

        # Update position using kinematics
        dt = 0.01  # Physics timestep

        # Forward kinematics
        new_yaw = yaw + angular_vel * dt
        new_yaw = np.clip(new_yaw, -np.pi, np.pi)

        # Position update
        dx = linear_vel * np.cos(new_yaw) * dt
        dy = linear_vel * np.sin(new_yaw) * dt
        new_pos = pos + np.array([dx, dy])

        # Sync PyBullet body with computed state
        p.resetBasePositionAndOrientation(
            self.body_id,
            [new_pos[0], new_pos[1], 0.15],
            p.getQuaternionFromEuler([0, 0, new_yaw]),
            physicsClientId=self.client
        )

        # Update internal state
        self.position = np.array([new_pos[0], new_pos[1], new_yaw], dtype=np.float32)
        self.velocity = np.array([linear_vel, 0.0, angular_vel], dtype=np.float32)

        # Update battery
        self._update_battery(linear_vel)

        # Check for task completion
        self._check_task_progress()

    def _update_battery(self, linear_vel: float):

        if self.charging and self.battery_level < self.battery_capacity:
            # Charging
            charge_amount = self.battery_charge_rate
            self.battery_level = min(self.battery_level + charge_amount, self.battery_capacity)
        else:
            # Discharging based on activity
            # Battery drain increases with velocity
            discharge = self.battery_discharge_rate * (1.0 + abs(linear_vel))
            self.battery_level = max(0.0, self.battery_level - discharge)

            if self.battery_level <= 0:
                self.charging = False
                self.has_task = False
                logger.warning(f"Robot {self.robot_id} battery depleted")

    def _check_task_progress(self):
        """Check if robot has reached task location and update progress."""
        if not self.has_task or self.current_task is None:
            return

        pickup_loc = np.array(self.current_task[:2])
        delivery_loc = np.array(self.current_task[3:5])

        # Check if at pickup location
        dist_to_pickup = np.linalg.norm(self.position[:2] - pickup_loc)
        if dist_to_pickup < 2.0 and not self.at_pickup:
            self.at_pickup = True
            self.task_progress = 50.0

        # Check if at delivery location
        if self.at_pickup:
            dist_to_delivery = np.linalg.norm(self.position[:2] - delivery_loc)
            if dist_to_delivery < 2.0:
                self.at_delivery = True
                self.completed_task = True
                self.has_task = False
                self.task_progress = 100.0

    def assign_task(self, task: Tuple):

        self.current_task = task
        self.has_task = True
        self.at_pickup = False
        self.at_delivery = False
        self.task_progress = 0.0
        self.completed_task = False

    def start_charging(self, station_id: int):

        self.charging = True
        self.charging_station_id = station_id

    def stop_charging(self):
        """Stop charging."""
        self.charging = False
        self.charging_station_id = None

    def get_position(self) -> np.ndarray:
        """Get robot position (x, y, theta)."""
        return self.position.copy()

    def get_velocity(self) -> np.ndarray:
        """Get robot velocity (vx, vy, omega)."""
        return self.velocity.copy()

    def get_battery_level(self) -> float:
        """Get battery level (0-100)."""
        return self.battery_level

    def get_state(self) -> Dict:
        """Get complete robot state."""
        return {
            'robot_id': self.robot_id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'battery_level': self.battery_level,
            'has_task': self.has_task,
            'task_progress': self.task_progress,
            'charging': self.charging,
        }

    def get_distance_to_point(self, point: np.ndarray) -> float:
        """Get Euclidean distance to a point."""
        return np.linalg.norm(self.position[:2] - point)

    def is_near_point(self, point: np.ndarray, distance_threshold: float = 2.0) -> bool:
        """Check if robot is near a point."""
        return self.get_distance_to_point(point) < distance_threshold

    def reset_task(self):
        """Reset task state."""
        self.current_task = None
        self.has_task = False
        self.task_progress = 0.0
        self.completed_task = False
        self.at_pickup = False
        self.at_delivery = False

    def can_accept_task(self) -> bool:
        """Check if robot can accept a new task."""
        return (
            not self.has_task and
            self.battery_level > self.battery_charge_threshold and
            not self.charging
        )
