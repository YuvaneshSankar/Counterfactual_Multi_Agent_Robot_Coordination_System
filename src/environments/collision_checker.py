"""
Collision Checker - Spatial Collision Detection

Detects collisions between robots, with obstacles, and validates paths.
Uses spatial hashing for efficient collision queries.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class CollisionChecker:
    """
    Spatial collision detection for multi-robot systems.
 
    Supports:
    - Robot-robot collisions
    - Robot-obstacle collisions
    - Path validation
    - Spatial queries using grid-based hashing
    """

    def __init__(
        self,
        robot_radius: float = 0.5,
        warehouse_layout=None,
        client: int = None,
        grid_size: float = 5.0,
    ):
        """
        Initialize collision checker.

        Args:
            robot_radius: Radius of robot for collision detection
            warehouse_layout: WarehouseLayout object
            client: PyBullet client ID
            grid_size: Grid cell size for spatial hashing
        """
        self.robot_radius = robot_radius
        self.warehouse_layout = warehouse_layout
        self.client = client
        self.grid_size = grid_size

        # Spatial hash grid
        self.collision_grid: dict = {}

        logger.info(f"CollisionChecker initialized: robot_radius={robot_radius}")

    def check_robot_collision(self, robot1, robot2) -> bool:
        """
        Check collision between two robots.

        Args:
            robot1: First robot
            robot2: Second robot

        Returns:
            True if robots collide
        """
        pos1 = robot1.get_position()[:2]
        pos2 = robot2.get_position()[:2]

        distance = np.linalg.norm(pos1 - pos2)
        collision_distance = 2 * self.robot_radius

        collides = distance < collision_distance

        if collides:
            logger.debug(f"Collision detected: Robot {robot1.robot_id} - Robot {robot2.robot_id}")

        return collides

    def check_obstacle_collision(self, robot, warehouse_layout) -> bool:
        """
        Check collision between robot and static obstacles.

        Args:
            robot: Robot object
            warehouse_layout: WarehouseLayout object

        Returns:
            True if robot collides with obstacle
        """
        if warehouse_layout is None:
            return False

        robot_pos = robot.get_position()[:2]

        # Check collision with shelves
        for shelf in warehouse_layout.shelves:
            if self._point_in_obstacle(robot_pos, shelf):
                return True

        # Check collision with walls
        if not self._point_in_bounds(robot_pos, warehouse_layout):
            return True

        return False

    def _point_in_obstacle(self, point: np.ndarray, obstacle: dict) -> bool:
        """
        Check if point is inside an obstacle (shelf).

        Args:
            point: (x, y) point
            obstacle: Obstacle dictionary with 'x', 'y', 'width', 'height'

        Returns:
            True if point is inside obstacle with buffer
        """
        x, y = point
        obs_x = obstacle['x']
        obs_y = obstacle['y']
        obs_w = obstacle['width']
        obs_h = obstacle['height']
        buffer = self.robot_radius

        return (
            obs_x - buffer < x < obs_x + obs_w + buffer and
            obs_y - buffer < y < obs_y + obs_h + buffer
        )

    def _point_in_bounds(self, point: np.ndarray, warehouse_layout) -> bool:
        """
        Check if point is within warehouse bounds.

        Args:
            point: (x, y) point
            warehouse_layout: WarehouseLayout object

        Returns:
            True if point is in bounds with buffer
        """
        x, y = point
        buffer = self.robot_radius

        return (
            buffer < x < warehouse_layout.width - buffer and
            buffer < y < warehouse_layout.height - buffer
        )

    def check_path_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        warehouse_layout=None,
        num_samples: int = 10
    ) -> bool:
        """
        Check if a straight-line path collides with obstacles.

        Args:
            start: Start position (x, y)
            end: End position (x, y)
            warehouse_layout: WarehouseLayout object
            num_samples: Number of points to sample along path

        Returns:
            True if path collides with obstacle
        """
        if warehouse_layout is None:
            return False

        # Sample points along path
        for t in np.linspace(0, 1, num_samples):
            point = start + t * (end - start)

            # Check bounds
            if not self._point_in_bounds(point, warehouse_layout):
                return True

            # Check shelves
            for shelf in warehouse_layout.shelves:
                if self._point_in_obstacle(point, shelf):
                    return True

        return False

    def get_nearby_robots(
        self,
        robot,
        nearby_robots_list: List,
        radius: float = 10.0
    ) -> List:
        """
        Get robots within a given radius.

        Args:
            robot: Query robot
            nearby_robots_list: List of all robots
            radius: Query radius

        Returns:
            List of nearby robots (excluding query robot)
        """
        robot_pos = robot.get_position()[:2]
        nearby = []

        for other_robot in nearby_robots_list:
            if other_robot.robot_id == robot.robot_id:
                continue

            other_pos = other_robot.get_position()[:2]
            distance = np.linalg.norm(robot_pos - other_pos)

            if distance < radius:
                nearby.append((other_robot, distance))

        return sorted(nearby, key=lambda x: x[1])

    def get_collision_grid(self, robots: List) -> dict:
        """
        Build spatial hash grid for fast collision queries.

        Args:
            robots: List of robots

        Returns:
            Grid dictionary with robot indices at each cell
        """
        grid = {}

        for robot in robots:
            pos = robot.get_position()[:2]
            cell = self._pos_to_grid_cell(pos)

            if cell not in grid:
                grid[cell] = []
            grid[cell].append(robot.robot_id)

        return grid

    def _pos_to_grid_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert position to grid cell.

        Args:
            pos: (x, y) position

        Returns:
            Grid cell (grid_x, grid_y)
        """
        cell_x = int(pos[0] / self.grid_size)
        cell_y = int(pos[1] / self.grid_size)
        return (cell_x, cell_y)

    def predict_collision(
        self,
        robot,
        future_position: np.ndarray,
        other_robots: List
    ) -> bool:
        """
        Predict if robot will collide if moved to future position.

        Args:
            robot: Robot to check
            future_position: Predicted position (x, y)
            other_robots: List of other robots

        Returns:
            True if collision predicted
        """
        collision_distance = 2 * self.robot_radius

        for other_robot in other_robots:
            if other_robot.robot_id == robot.robot_id:
                continue

            other_pos = other_robot.get_position()[:2]
            distance = np.linalg.norm(future_position - other_pos)

            if distance < collision_distance:
                return True

        return False

    def get_collision_pairs(self, robots: List) -> List[Tuple]:
        """
        Get all colliding robot pairs.

        Args:
            robots: List of robots

        Returns:
            List of (robot_id1, robot_id2) pairs that collide
        """
        collisions = []

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                if self.check_robot_collision(robots[i], robots[j]):
                    collisions.append((robots[i].robot_id, robots[j].robot_id))

        return collisions

    def get_obstacle_clearance(
        self,
        position: np.ndarray,
        warehouse_layout
    ) -> float:
        """
        Get minimum clearance to nearest obstacle.

        Args:
            position: (x, y) position
            warehouse_layout: WarehouseLayout object

        Returns:
            Minimum distance to obstacle (can be negative if inside obstacle)
        """
        if warehouse_layout is None:
            return float('inf')

        min_clearance = float('inf')

        # Check distance to walls
        margin_left = position[0]
        margin_right = warehouse_layout.width - position[0]
        margin_bottom = position[1]
        margin_top = warehouse_layout.height - position[1]

        min_clearance = min(
            min_clearance,
            margin_left,
            margin_right,
            margin_bottom,
            margin_top
        )

        # Check distance to shelves
        for shelf in warehouse_layout.shelves:
            clearance = self._point_to_obstacle_distance(position, shelf)
            min_clearance = min(min_clearance, clearance)

        return min_clearance - self.robot_radius

    def _point_to_obstacle_distance(
        self,
        point: np.ndarray,
        obstacle: dict
    ) -> float:
        """
        Get distance from point to nearest part of obstacle.

        Args:
            point: (x, y) point
            obstacle: Obstacle dictionary

        Returns:
            Distance to obstacle (negative if inside)
        """
        x, y = point
        obs_x = obstacle['x']
        obs_y = obstacle['y']
        obs_w = obstacle['width']
        obs_h = obstacle['height']

        # Find closest point on obstacle to query point
        closest_x = np.clip(x, obs_x, obs_x + obs_w)
        closest_y = np.clip(y, obs_y, obs_y + obs_h)

        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)

        # Negative if inside
        if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
            distance = -distance

        return distance
