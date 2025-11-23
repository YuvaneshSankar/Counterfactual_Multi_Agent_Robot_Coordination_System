

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TaskAllocator:


    def __init__(
        self,
        num_robots: int = 5,
        allocation_strategy: str = 'greedy_nearest',
    ):

        self.num_robots = num_robots
        self.allocation_strategy = allocation_strategy

        # Tracking
        self.task_assignments: Dict = {}
        self.robot_loads: Dict = {i: 0 for i in range(num_robots)}
        self.allocation_history: List = []

        logger.info(f"TaskAllocator initialized: strategy={allocation_strategy}")

    def allocate_tasks(
        self,
        robots: List,
        tasks: List[Tuple],
        warehouse_layout=None
    ) -> Dict[int, Optional[Tuple]]:

        allocations = {}
        available_robots = [r for r in robots if r.can_accept_task()]

        if self.allocation_strategy == 'greedy_nearest':
            allocations = self._allocate_greedy_nearest(available_robots, tasks)
        elif self.allocation_strategy == 'load_balanced':
            allocations = self._allocate_load_balanced(available_robots, tasks)
        elif self.allocation_strategy == 'priority_aware':
            allocations = self._allocate_priority_aware(available_robots, tasks)
        else:
            raise ValueError(f"Unknown strategy: {self.allocation_strategy}")

        # Record allocation
        self.allocation_history.append({
            'step': len(self.allocation_history),
            'allocations': allocations,
            'pending_tasks': len(tasks),
            'available_robots': len(available_robots),
        })

        return allocations

    def _allocate_greedy_nearest(
        self,
        available_robots: List,
        tasks: List[Tuple]
    ) -> Dict[int, Optional[Tuple]]:

        allocations = {}
        assigned_robots = set()

        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t[2], reverse=True)

        for task in sorted_tasks:
            pickup_loc = np.array(task[:2])

            # Find nearest unassigned available robot
            best_robot = None
            best_distance = float('inf')

            for robot in available_robots:
                if robot.robot_id in assigned_robots:
                    continue

                robot_pos = robot.get_position()[:2]
                distance = np.linalg.norm(robot_pos - pickup_loc)

                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot

            if best_robot is not None:
                allocations[best_robot.robot_id] = task
                assigned_robots.add(best_robot.robot_id)

        return allocations

    def _allocate_load_balanced(
        self,
        available_robots: List,
        tasks: List[Tuple]
    ) -> Dict[int, Optional[Tuple]]:

        allocations = {}
        assigned_robots = set()

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t[2], reverse=True)

        for task in sorted_tasks:
            pickup_loc = np.array(task[:2])

            # Find robot with lowest load among nearby robots
            candidates = []
            for robot in available_robots:
                if robot.robot_id in assigned_robots:
                    continue

                robot_pos = robot.get_position()[:2]
                distance = np.linalg.norm(robot_pos - pickup_loc)
                load = self.robot_loads.get(robot.robot_id, 0)

                # Score: prioritize low load, then nearby
                score = load + 0.1 * distance
                candidates.append((robot, score, distance))

            if candidates:
                best_robot = min(candidates, key=lambda x: x[1])[0]
                allocations[best_robot.robot_id] = task
                assigned_robots.add(best_robot.robot_id)
                self.robot_loads[best_robot.robot_id] += 1

        return allocations

    def _allocate_priority_aware(
        self,
        available_robots: List,
        tasks: List[Tuple]
    ) -> Dict[int, Optional[Tuple]]:
        allocations = {}
        assigned_robots = set()

        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t[2], reverse=True)

        for task in sorted_tasks:
            pickup_loc = np.array(task[:2])
            priority = task[2]
            deadline = task[5]

            # For urgent tasks (high priority or low deadline), use best available robot
            urgency_score = priority / 10.0 + (1.0 - deadline / 500.0)

            best_robot = None
            best_score = float('inf')

            for robot in available_robots:
                if robot.robot_id in assigned_robots:
                    continue

                robot_pos = robot.get_position()[:2]
                distance = np.linalg.norm(robot_pos - pickup_loc)
                battery = robot.get_battery_level() / 100.0

                # Score combines distance and battery, weighted by urgency
                if urgency_score > 0.7:
                    # High urgency: prefer battery and distance
                    score = distance - 2.0 * battery
                else:
                    # Normal: standard distance
                    score = distance

                if score < best_score:
                    best_score = score
                    best_robot = robot

            if best_robot is not None:
                allocations[best_robot.robot_id] = task
                assigned_robots.add(best_robot.robot_id)

        return allocations

    def update_robot_load(self, robot_id: int, load: int):
        """Update robot workload."""
        self.robot_loads[robot_id] = load

    def get_robot_utilization(self) -> Dict:
        """Get utilization statistics."""
        return {
            'total_assignments': len(self.allocation_history),
            'robot_loads': self.robot_loads.copy(),
            'avg_load': np.mean(list(self.robot_loads.values())),
        }

    def reset(self):
        """Reset allocator state."""
        self.robot_loads = {i: 0 for i in range(self.num_robots)}
        self.allocation_history = []


class HungarianAllocator:


    def __init__(self, num_robots: int = 5):

        self.num_robots = num_robots

    def allocate_tasks(
        self,
        robots: List,
        tasks: List[Tuple]
    ) -> Dict[int, Optional[Tuple]]:

        available_robots = [r for r in robots if r.can_accept_task()]

        if not available_robots or not tasks:
            return {}

        # Build cost matrix
        num_r = len(available_robots)
        num_t = len(tasks)

        if num_t == 0:
            return {}

        # Pad to square matrix
        size = max(num_r, num_t)
        cost_matrix = np.ones((size, size)) * 1e6

        # Compute costs (distances)
        for i, robot in enumerate(available_robots):
            for j, task in enumerate(tasks):
                pickup_loc = np.array(task[:2])
                robot_pos = robot.get_position()[:2]
                distance = np.linalg.norm(robot_pos - pickup_loc)
                cost_matrix[i, j] = distance

        # Simple greedy matching (Hungarian would be too complex here)
        assignments = self._greedy_matching(cost_matrix, available_robots, tasks)

        return assignments

    def _greedy_matching(
        self,
        cost_matrix: np.ndarray,
        robots: List,
        tasks: List[Tuple]
    ) -> Dict[int, Optional[Tuple]]:
        """Greedy matching on cost matrix."""
        allocations = {}
        assigned_cols = set()

        for i, robot in enumerate(robots):
            min_cost = float('inf')
            best_task_idx = None

            for j, task in enumerate(tasks):
                if j in assigned_cols:
                    continue

                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_task_idx = j

            if best_task_idx is not None:
                allocations[robot.robot_id] = tasks[best_task_idx]
                assigned_cols.add(best_task_idx)

        return allocations
