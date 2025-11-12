"""
Path Planner - Motion Planning for Warehouse Navigation

Implements A* pathfinding with obstacle avoidance and trajectory planning.
Computes collision-free paths through warehouse environment.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from heapq import heappush, heappop
import logging

logger = logging.getLogger(__name__)


class PathPlanner:
    """
    A* Path Planning Algorithm for warehouse navigation.

    Features:
    - Grid-based A* search
    - Obstacle avoidance
    - Heuristic-guided search
    - Path smoothing
    """

    def __init__(
        self,
        warehouse_width: float = 100.0,
        warehouse_height: float = 80.0,
        grid_resolution: float = 1.0,
        robot_radius: float = 0.5,
    ):
        """
        Initialize path planner.

        Args:
            warehouse_width: Warehouse width
            warehouse_height: Warehouse height
            grid_resolution: Grid cell size for pathfinding
            robot_radius: Robot collision radius
        """
        self.width = warehouse_width
        self.height = warehouse_height
        self.grid_resolution = grid_resolution
        self.robot_radius = robot_radius

        # Grid dimensions
        self.grid_width = int(np.ceil(warehouse_width / grid_resolution))
        self.grid_height = int(np.ceil(warehouse_height / grid_resolution))

        # Occupancy grid (0=free, 1=occupied)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Path cache
        self.path_cache: Dict = {}

        logger.info(
            f"PathPlanner initialized: {self.grid_width}x{self.grid_height} grid, "
            f"resolution={grid_resolution}m"
        )

    def plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        warehouse_layout=None,
    ) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal using A*.

        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            warehouse_layout: Warehouse layout for obstacle checking

        Returns:
            Path as list of waypoints, or None if no path found
        """
        # Build occupancy grid
        if warehouse_layout is not None:
            self._update_occupancy_grid(warehouse_layout)

        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # Check if start/goal are valid
        if not self._is_valid_grid_cell(start_grid):
            logger.warning(f"Start position not valid: {start}")
            return None

        if not self._is_valid_grid_cell(goal_grid):
            logger.warning(f"Goal position not valid: {goal}")
            return None

        # A* search
        path_grid = self._astar_search(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert path back to world coordinates
        path = [self._grid_to_world(p) for p in path_grid]

        # Smooth path
        path = self._smooth_path(path)

        return path

    def _update_occupancy_grid(self, warehouse_layout):
        """
        Update occupancy grid based on warehouse obstacles.

        Args:
            warehouse_layout: Warehouse layout
        """
        self.occupancy_grid.fill(0)

        # Mark shelves as occupied
        for shelf in warehouse_layout.shelves:
            x_min = max(0, int((shelf['x'] - self.robot_radius) / self.grid_resolution))
            x_max = min(self.grid_width, int((shelf['x'] + shelf['width'] + self.robot_radius) / self.grid_resolution))
            y_min = max(0, int((shelf['y'] - self.robot_radius) / self.grid_resolution))
            y_max = min(self.grid_height, int((shelf['y'] + shelf['height'] + self.robot_radius) / self.grid_resolution))

            self.occupancy_grid[y_min:y_max, x_min:x_max] = 1

        # Mark boundary as occupied
        boundary_cells = int(np.ceil(self.robot_radius / self.grid_resolution))
        self.occupancy_grid[:boundary_cells, :] = 1
        self.occupancy_grid[-boundary_cells:, :] = 1
        self.occupancy_grid[:, :boundary_cells] = 1
        self.occupancy_grid[:, -boundary_cells:] = 1

    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x = int(np.clip(pos[0] / self.grid_resolution, 0, self.grid_width - 1))
        y = int(np.clip(pos[1] / self.grid_resolution, 0, self.grid_height - 1))
        return (x, y)

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        x = (grid_pos[0] + 0.5) * self.grid_resolution
        y = (grid_pos[1] + 0.5) * self.grid_resolution
        return np.array([x, y])

    def _is_valid_grid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if grid cell is valid (free)."""
        x, y = cell

        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False

        return self.occupancy_grid[y, x] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic for A*: Euclidean distance."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm.

        Args:
            start: Start grid cell
            goal: Goal grid cell

        Returns:
            Path as list of grid cells, or None if no path
        """
        # Priority queue: (f_score, counter, position)
        open_set = []
        counter = 0
        heappush(open_set, (0, counter, start))
        counter += 1

        # g_score: cost from start
        g_score = {start: 0}

        # f_score: g + h
        f_score = {start: self._heuristic(start, goal)}

        # Came from tracking
        came_from = {}

        # Closed set
        closed_set = set()

        while open_set:
            _, _, current = heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            closed_set.add(current)

            # Expand neighbors (8-connected grid)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid_grid_cell(neighbor):
                    continue

                if neighbor in closed_set:
                    continue

                # Movement cost
                move_cost = 1.0 if abs(dx) + abs(dy) == 1 else 1.414
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1

        return None  # No path found

    def _smooth_path(self, path: List[np.ndarray], max_iterations: int = 10) -> List[np.ndarray]:
        """
        Smooth path using line-of-sight shortcutting.

        Args:
            path: Original path
            max_iterations: Maximum smoothing iterations

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            # Try to shortcut to point further ahead
            for j in range(len(path) - 1, i, -1):
                # Check if line from current to j-th point is collision-free
                if self._is_path_valid(smoothed[-1], path[j]):
                    smoothed.append(path[j])
                    break
            else:
                # Couldn't shortcut, add next point
                smoothed.append(path[i])

        smoothed.append(path[-1])
        return smoothed

    def _is_path_valid(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 20
    ) -> bool:
        """
        Check if straight path is collision-free.

        Args:
            start: Start position
            end: End position
            num_samples: Number of points to sample

        Returns:
            True if path is valid
        """
        for t in np.linspace(0, 1, num_samples):
            point = start + t * (end - start)
            grid_pos = self._world_to_grid(point)

            if not self._is_valid_grid_cell(grid_pos):
                return False

        return True

    def get_path_distance(self, path: List[np.ndarray]) -> float:
        """Compute total path distance."""
        if len(path) <= 1:
            return 0.0

        distance = 0.0
        for i in range(len(path) - 1):
            distance += np.linalg.norm(path[i+1] - path[i])

        return distance


class TrajectoryPlanner:
    """
    Generate smooth robot trajectories with velocity profiles.

    Produces time-parameterized trajectories with velocity and acceleration constraints.
    """

    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 1.0,
    ):
        """
        Initialize trajectory planner.

        Args:
            max_velocity: Maximum robot velocity
            max_acceleration: Maximum acceleration
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_trajectory(
        self,
        path: List[np.ndarray],
        dt: float = 0.01,
    ) -> List[Dict]:
        """
        Generate time-parameterized trajectory from path.

        Args:
            path: List of waypoints
            dt: Time step

        Returns:
            List of trajectory points with position, velocity
        """
        trajectory = []

        for i, waypoint in enumerate(path):
            # Compute desired velocity direction
            if i < len(path) - 1:
                direction = path[i + 1] - waypoint
                distance = np.linalg.norm(direction)

                if distance > 1e-6:
                    direction = direction / distance
                    velocity = self.max_velocity * direction
                else:
                    velocity = np.zeros(2)
            else:
                velocity = np.zeros(2)

            trajectory_point = {
                'position': waypoint.copy(),
                'velocity': velocity.copy(),
                'time': i * dt,
            }
            trajectory.append(trajectory_point)

        return trajectory
