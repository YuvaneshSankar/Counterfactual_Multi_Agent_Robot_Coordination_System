"""
Task Generator - Dynamic Task Creation and Management

Generates warehouse tasks (pickup/delivery) with priorities and deadlines.
Implements Poisson process for realistic task arrival patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TaskGenerator:
    """
    Generate and manage warehouse tasks.

    Each task has:
    - Pickup location (x, y)
    - Delivery location (x, y)
    - Priority (1-10, higher = more urgent)
    - Deadline (steps until due)

    Task format: (pickup_x, pickup_y, priority, delivery_x, delivery_y, deadline)
    """

    def __init__(
        self,
        warehouse_width: float = 100.0,
        warehouse_height: float = 80.0,
        arrival_rate: float = 0.5,
        max_queue: int = 20,
        min_distance: float = 20.0,
        max_deadline: int = 500,
    ):
        """
        Initialize task generator.

        Args:
            warehouse_width: Warehouse width in meters
            warehouse_height: Warehouse height in meters
            arrival_rate: Task arrival rate (Poisson lambda parameter)
            max_queue: Maximum pending tasks
            min_distance: Minimum distance between pickup and delivery
            max_deadline: Maximum deadline for tasks
        """
        self.warehouse_width = warehouse_width
        self.warehouse_height = warehouse_height
        self.arrival_rate = arrival_rate
        self.max_queue = max_queue
        self.min_distance = min_distance
        self.max_deadline = max_deadline

        # Task queue
        self.pending_tasks: List[Tuple] = []
        self.completed_tasks: List[Tuple] = []
        self.task_counter = 0
        self.current_step = 0

        # Delivery zones (predefined locations)
        self.delivery_zones = self._create_delivery_zones()
        self.pickup_zones = self._create_pickup_zones()

        logger.info(
            f"TaskGenerator initialized: {warehouse_width}x{warehouse_height}, "
            f"arrival_rate={arrival_rate}"
        )

    def _create_delivery_zones(self) -> List[Tuple[float, float]]:
        """
        Create predefined delivery zones.

        Returns:
            List of delivery zone locations
        """
        # Place delivery zones at warehouse perimeter
        zones = []
        margin = 5.0

        # Left side
        zones.append((margin, self.warehouse_height / 2))
        # Right side
        zones.append((self.warehouse_width - margin, self.warehouse_height / 2))
        # Top
        zones.append((self.warehouse_width / 2, self.warehouse_height - margin))
        # Bottom
        zones.append((self.warehouse_width / 2, margin))

        return zones

    def _create_pickup_zones(self) -> List[Tuple[float, float]]:
        """
        Create predefined pickup zones (input dock areas).

        Returns:
            List of pickup zone locations
        """
        zones = []
        margin = 5.0

        # Create zones in grid pattern in warehouse
        num_zones_x = 3
        num_zones_y = 2

        for i in range(num_zones_x):
            for j in range(num_zones_y):
                x = margin + (i + 1) * (self.warehouse_width - 2*margin) / (num_zones_x + 1)
                y = margin + (j + 1) * (self.warehouse_height - 2*margin) / (num_zones_y + 1)
                zones.append((x, y))

        return zones

    def step(self):
        """
        Generate new tasks and update task queue.
        Called once per environment step.
        """
        self.current_step += 1

        # Generate new tasks based on Poisson process
        num_new_tasks = np.random.poisson(self.arrival_rate)

        for _ in range(num_new_tasks):
            if len(self.pending_tasks) < self.max_queue:
                task = self._generate_task()
                self.pending_tasks.append(task)

        # Update deadlines for pending tasks
        self.pending_tasks = [
            (p[0], p[1], p[2], p[3], p[4], p[5] - 1)
            for p in self.pending_tasks
            if p[5] - 1 > 0  # Remove expired tasks
        ]

    def _generate_task(self) -> Tuple:
        """
        Generate a single task.

        Returns:
            task: Task tuple (pickup_x, pickup_y, priority, delivery_x, delivery_y, deadline)
        """
        # Random pickup location from pickup zones (with noise)
        pickup_zone = self.pickup_zones[np.random.randint(len(self.pickup_zones))]
        pickup_loc = (
            pickup_zone[0] + np.random.normal(0, 2.0),
            pickup_zone[1] + np.random.normal(0, 2.0)
        )

        # Clamp to warehouse bounds
        pickup_loc = (
            np.clip(pickup_loc[0], 5.0, self.warehouse_width - 5.0),
            np.clip(pickup_loc[1], 5.0, self.warehouse_height - 5.0)
        )

        # Random delivery location from delivery zones (with noise)
        delivery_zone = self.delivery_zones[np.random.randint(len(self.delivery_zones))]
        delivery_loc = (
            delivery_zone[0] + np.random.normal(0, 2.0),
            delivery_zone[1] + np.random.normal(0, 2.0)
        )

        # Clamp to warehouse bounds
        delivery_loc = (
            np.clip(delivery_loc[0], 5.0, self.warehouse_width - 5.0),
            np.clip(delivery_loc[1], 5.0, self.warehouse_height - 5.0)
        )

        # Random priority (1-10, weighted towards higher priorities)
        # Using exponential distribution
        priority = min(10, int(np.random.exponential(scale=3) + 1))

        # Random deadline (tasks with higher priority get shorter deadlines)
        base_deadline = self.max_deadline // 2
        deadline = int(base_deadline * (1.0 - priority / 10.0) + np.random.uniform(100, 200))

        task = (
            pickup_loc[0],
            pickup_loc[1],
            priority,
            delivery_loc[0],
            delivery_loc[1],
            deadline
        )

        self.task_counter += 1
        return task

    def add_task(self, task: Tuple):
        """
        Manually add a task to the queue.

        Args:
            task: Task tuple
        """
        if len(self.pending_tasks) < self.max_queue:
            self.pending_tasks.append(task)

    def get_nearest_task(self, position: Tuple[float, float]) -> Optional[Tuple]:
        """
        Get nearest pending task to a position.

        Args:
            position: (x, y) position

        Returns:
            Nearest task or None if queue is empty
        """
        if not self.pending_tasks:
            return None

        nearest = min(
            self.pending_tasks,
            key=lambda t: np.sqrt((t[0] - position[0])**2 + (t[1] - position[1])**2)
        )
        return nearest

    def get_highest_priority_task(self) -> Optional[Tuple]:
        """
        Get highest priority pending task.

        Returns:
            Highest priority task or None if queue is empty
        """
        if not self.pending_tasks:
            return None

        highest = max(self.pending_tasks, key=lambda t: t[2])
        return highest

    def complete_task(self, task: Tuple):
        """
        Mark a task as completed.

        Args:
            task: Task to mark as completed
        """
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
            self.completed_tasks.append(task)

    def remove_task(self, task: Tuple):
        """
        Remove a task from pending queue.

        Args:
            task: Task to remove
        """
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

    def get_queue_status(self) -> Dict:
        """
        Get status of task queue.

        Returns:
            Dictionary with queue statistics
        """
        pending_priorities = [t[2] for t in self.pending_tasks]
        pending_deadlines = [t[5] for t in self.pending_tasks]

        return {
            'pending_count': len(self.pending_tasks),
            'completed_count': len(self.completed_tasks),
            'total_generated': self.task_counter,
            'avg_priority': np.mean(pending_priorities) if pending_priorities else 0,
            'avg_deadline': np.mean(pending_deadlines) if pending_deadlines else 0,
            'max_priority': max(pending_priorities) if pending_priorities else 0,
            'min_deadline': min(pending_deadlines) if pending_deadlines else 0,
        }

    def get_pending_tasks(self) -> List[Tuple]:
        """Get list of pending tasks."""
        return self.pending_tasks.copy()

    def get_task_distribution(self) -> Dict:
        """
        Get distribution of tasks by priority.

        Returns:
            Dictionary with priority distribution
        """
        distribution = {}
        for task in self.pending_tasks:
            priority = task[2]
            distribution[priority] = distribution.get(priority, 0) + 1

        return distribution

    def get_urgent_tasks(self, urgency_threshold: float = 0.3) -> List[Tuple]:
        """
        Get tasks that are close to deadline.

        Args:
            urgency_threshold: Fraction of deadline remaining (0-1)

        Returns:
            List of urgent tasks
        """
        urgent = []
        current_deadline_fraction = 1.0

        for task in self.pending_tasks:
            deadline_fraction = task[5] / self.max_deadline
            if deadline_fraction < urgency_threshold:
                urgent.append(task)

        return urgent

    def reset(self):
        """Reset task generator."""
        self.pending_tasks = []
        self.completed_tasks = []
        self.task_counter = 0
        self.current_step = 0

    def get_statistics(self) -> Dict:
        """Get task statistics."""
        return {
            'current_step': self.current_step,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_generated': self.task_counter,
            'queue_status': self.get_queue_status(),
        }
