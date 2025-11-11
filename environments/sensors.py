"""
Sensors Module - Robot Perception Sensors

Implements LiDAR and other sensors for robot perception.
Provides realistic observations for decentralized policies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LiDARSensor:
    """
    2D LiDAR Sensor for obstacle detection.

    Simulates a 2D LiDAR scanner with configurable:
    - Number of beams (rays)
    - Maximum range
    - Field of view
    - Angular resolution
    """

    def __init__(
        self,
        robot,
        config: Dict,
        num_beams: int = 16,
        max_range: float = 20.0,
        fov: float = 360.0,
    ):
        """
        Initialize LiDAR sensor.

        Args:
            robot: Robot object that this sensor belongs to
            config: Configuration dictionary
            num_beams: Number of laser beams
            max_range: Maximum sensing range (meters)
            fov: Field of view (degrees, 360 for full circle)
        """
        self.robot = robot
        self.config = config
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov = fov

        # Precompute beam angles
        self.beam_angles = np.linspace(
            -fov / 2,
            fov / 2,
            num_beams,
            endpoint=False
        )
        self.beam_angles_rad = np.deg2rad(self.beam_angles)

        # Last readings
        self.last_readings = np.ones(num_beams) * max_range
        self.last_update_step = -1

        logger.info(
            f"LiDARSensor initialized: {num_beams} beams, "
            f"range={max_range}m, fov={fov}°"
        )

    def get_readings(self, warehouse_layout=None) -> np.ndarray:
        """
        Get LiDAR readings.

        Simulates laser rays from robot position, checks for intersections
        with obstacles and other robots.

        Args:
            warehouse_layout: WarehouseLayout object for obstacle checking

        Returns:
            readings: Normalized distance readings [num_beams] (0-1, where 1=max_range)
        """
        robot_pos = self.robot.get_position()[:2]
        robot_yaw = self.robot.get_position()[2]

        readings = np.zeros(self.num_beams)

        for i, beam_angle_rad in enumerate(self.beam_angles_rad):
            # Global beam angle
            global_angle = robot_yaw + beam_angle_rad

            # Ray direction
            ray_dir = np.array([
                np.cos(global_angle),
                np.sin(global_angle)
            ])

            # Find nearest intersection
            min_distance = self.max_range

            # Check intersection with obstacles (if layout provided)
            if warehouse_layout is not None:
                # Check walls
                wall_distance = self._check_wall_intersection(
                    robot_pos, ray_dir, warehouse_layout
                )
                min_distance = min(min_distance, wall_distance)

                # Check shelves
                shelf_distance = self._check_shelf_intersection(
                    robot_pos, ray_dir, warehouse_layout
                )
                min_distance = min(min_distance, shelf_distance)

            # Normalize to [0, 1]
            readings[i] = min_distance / self.max_range

        self.last_readings = readings
        return readings.astype(np.float32)

    def _check_wall_intersection(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        warehouse_layout
    ) -> float:
        """
        Check ray-wall intersection distance.

        Args:
            start: Ray start position
            direction: Ray direction (normalized)
            warehouse_layout: WarehouseLayout object

        Returns:
            Distance to wall (max_range if no intersection)
        """
        width = warehouse_layout.width
        height = warehouse_layout.height

        min_distance = self.max_range

        # Check all four walls
        walls = [
            {'x': 0, 'y': 0},  # Left wall (x=0)
            {'x': width, 'y': 0},  # Right wall (x=width)
            {'x': 0, 'y': 0},  # Bottom wall (y=0)
            {'x': 0, 'y': height},  # Top wall (y=height)
        ]

        # Ray-line intersection tests
        # Left wall (x=0)
        if abs(direction[0]) > 1e-6:
            t = -start[0] / direction[0]
            if 0 < t < self.max_range:
                hit_y = start[1] + t * direction[1]
                if 0 <= hit_y <= height:
                    min_distance = min(min_distance, t)

        # Right wall (x=width)
        if abs(direction[0]) > 1e-6:
            t = (width - start[0]) / direction[0]
            if 0 < t < self.max_range:
                hit_y = start[1] + t * direction[1]
                if 0 <= hit_y <= height:
                    min_distance = min(min_distance, t)

        # Bottom wall (y=0)
        if abs(direction[1]) > 1e-6:
            t = -start[1] / direction[1]
            if 0 < t < self.max_range:
                hit_x = start[0] + t * direction[0]
                if 0 <= hit_x <= width:
                    min_distance = min(min_distance, t)

        # Top wall (y=height)
        if abs(direction[1]) > 1e-6:
            t = (height - start[1]) / direction[1]
            if 0 < t < self.max_range:
                hit_x = start[0] + t * direction[0]
                if 0 <= hit_x <= width:
                    min_distance = min(min_distance, t)

        return min_distance

    def _check_shelf_intersection(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        warehouse_layout
    ) -> float:
        """
        Check ray-shelf intersection distance.

        Args:
            start: Ray start position
            direction: Ray direction (normalized)
            warehouse_layout: WarehouseLayout object

        Returns:
            Distance to nearest shelf
        """
        min_distance = self.max_range

        for shelf in warehouse_layout.shelves:
            shelf_x = shelf['x']
            shelf_y = shelf['y']
            shelf_w = shelf['width']
            shelf_h = shelf['height']

            # Check intersection with shelf bounding box
            distance = self._ray_box_intersection(
                start, direction,
                shelf_x, shelf_y, shelf_w, shelf_h
            )

            if distance < self.max_range:
                min_distance = min(min_distance, distance)

        return min_distance

    def _ray_box_intersection(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        box_x: float,
        box_y: float,
        box_w: float,
        box_h: float
    ) -> float:
        """
        Compute ray-box intersection using slab method.

        Args:
            start: Ray start position
            direction: Ray direction
            box_x, box_y: Box position
            box_w, box_h: Box dimensions

        Returns:
            Distance to box (max_range if no intersection)
        """
        # Box bounds
        x_min = box_x
        x_max = box_x + box_w
        y_min = box_y
        y_max = box_y + box_h

        # Initialize t ranges for all slabs
        t_min = 0
        t_max = self.max_range

        # X slab
        if abs(direction[0]) > 1e-6:
            t1 = (x_min - start[0]) / direction[0]
            t2 = (x_max - start[0]) / direction[0]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:
            # Ray parallel to YZ plane
            if not (x_min <= start[0] <= x_max):
                return self.max_range

        # Y slab
        if abs(direction[1]) > 1e-6:
            t1 = (y_min - start[1]) / direction[1]
            t2 = (y_max - start[1]) / direction[1]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:
            # Ray parallel to XZ plane
            if not (y_min <= start[1] <= y_max):
                return self.max_range

        # Check if intersection exists
        if t_min < t_max and t_max > 0:
            if t_min > 0:
                return min(t_min, self.max_range)
            elif t_max > 0:
                return min(t_max, self.max_range)

        return self.max_range

    def get_visualization_rays(self) -> List[Tuple]:
        """
        Get rays for visualization.

        Returns:
            List of (start, end) positions for each ray
        """
        robot_pos = self.robot.get_position()[:2]
        robot_yaw = self.robot.get_position()[2]

        rays = []
        for i, beam_angle_rad in enumerate(self.beam_angles_rad):
            global_angle = robot_yaw + beam_angle_rad
            distance = self.last_readings[i] * self.max_range

            end = robot_pos + distance * np.array([
                np.cos(global_angle),
                np.sin(global_angle)
            ])

            rays.append((robot_pos.copy(), end.copy()))

        return rays


class OdometrySensor:
    """
    Odometry Sensor - Robot Pose and Velocity Tracking

    Provides reliable estimates of robot position, orientation, and velocity.
    """

    def __init__(self, robot, noise_level: float = 0.0):
        """
        Initialize odometry sensor.

        Args:
            robot: Robot object
            noise_level: Gaussian noise standard deviation (0=no noise)
        """
        self.robot = robot
        self.noise_level = noise_level

        # Accumulated drift
        self.position_drift = np.zeros(2)
        self.heading_drift = 0.0

    def get_odometry(self) -> Dict:
        """
        Get odometry reading with optional noise.

        Returns:
            Dictionary with position, orientation, velocity
        """
        pos = self.robot.get_position()
        vel = self.robot.get_velocity()

        # Add noise if configured
        if self.noise_level > 0:
            pos = pos + np.random.normal(0, self.noise_level, size=3)
            vel = vel + np.random.normal(0, self.noise_level, size=3)

        return {
            'position': pos.copy(),
            'velocity': vel.copy(),
            'timestamp': 0,  # Could track real time
        }


class RangeToGoalSensor:
    """
    Range to Goal Sensor - Relative Position to Task Location

    Provides relative distance and angle to current task destination.
    """

    def __init__(self, robot):
        """
        Initialize range-to-goal sensor.

        Args:
            robot: Robot object
        """
        self.robot = robot

    def get_range_to_goal(self, goal_position: Tuple) -> Dict:
        """
        Get range and angle to goal.

        Args:
            goal_position: Target (x, y) position

        Returns:
            Dictionary with distance and angle to goal
        """
        robot_pos = self.robot.get_position()[:2]
        robot_yaw = self.robot.get_position()[2]

        # Vector to goal
        to_goal = np.array(goal_position) - robot_pos
        distance = np.linalg.norm(to_goal)

        # Angle to goal
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        relative_angle = angle_to_goal - robot_yaw

        # Normalize angle to [-pi, pi]
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi

        return {
            'distance': distance,
            'angle': relative_angle,
            'to_goal': to_goal,
        }
