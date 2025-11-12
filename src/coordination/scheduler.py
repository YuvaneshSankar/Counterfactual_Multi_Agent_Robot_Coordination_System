"""
Scheduler - Real-Time Task Scheduling and Resource Management

Manages task execution order, timing constraints, and resource allocation.
Implements priority-based and deadline-aware scheduling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Real-time task scheduler with priority and deadline awareness.

    Scheduling Policies:
    - FIFO: First-in-first-out
    - Priority: Highest priority first
    - EDF: Earliest deadline first
    - LLF: Least laxity first
    """

    def __init__(
        self,
        num_robots: int = 5,
        scheduling_policy: str = 'priority',
    ):
        """
        Initialize task scheduler.

        Args:
            num_robots: Number of robots
            scheduling_policy: Scheduling policy ('fifo', 'priority', 'edf', 'llf')
        """
        self.num_robots = num_robots
        self.scheduling_policy = scheduling_policy

        # Task queues
        self.task_queue: List[Tuple] = []
        self.in_progress: Dict[int, Tuple] = {}
        self.completed_tasks: List[Tuple] = []

        # Timing
        self.current_time = 0
        self.time_step = 0.01

        logger.info(f"TaskScheduler initialized: policy={scheduling_policy}")

    def add_task(self, task: Tuple):
        """
        Add task to scheduler.

        Args:
            task: Task tuple (pickup_x, pickup_y, priority, delivery_x, delivery_y, deadline)
        """
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: self._priority_score(t), reverse=True)

    def get_next_task(self, robot_id: int) -> Optional[Tuple]:
        """
        Get next task for robot.

        Args:
            robot_id: Robot ID

        Returns:
            Next task or None if queue empty
        """
        if not self.task_queue:
            return None

        # Select task based on scheduling policy
        if self.scheduling_policy == 'fifo':
            task = self.task_queue.pop(0)
        elif self.scheduling_policy == 'priority':
            task = self.task_queue.pop(0)  # Already sorted by priority
        elif self.scheduling_policy == 'edf':
            # Earliest deadline first
            task = min(self.task_queue, key=lambda t: t[5])
            self.task_queue.remove(task)
        elif self.scheduling_policy == 'llf':
            # Least laxity first
            task = self._select_least_laxity()
        else:
            task = self.task_queue.pop(0)

        self.in_progress[robot_id] = task
        return task

    def _priority_score(self, task: Tuple) -> float:
        """
        Compute priority score for task.

        Lower deadline and higher priority increase score.

        Args:
            task: Task tuple

        Returns:
            Priority score
        """
        priority = task[2]  # 1-10
        deadline = task[5]

        # Combine priority and deadline urgency
        deadline_urgency = 1.0 / (deadline + 1)  # Lower deadline = higher urgency
        score = (priority / 10.0) + deadline_urgency

        return score

    def _select_least_laxity(self) -> Tuple:
        """
        Select task with least laxity.

        Laxity = deadline - processing_time
        """
        least_laxity = None
        min_laxity = float('inf')

        for task in self.task_queue:
            deadline = task[5]
            # Assume processing time ~ distance
            pickup = np.array(task[:2])
            delivery = np.array(task[3:5])
            distance = np.linalg.norm(delivery - pickup)
            processing_time = distance / 2.0  # Rough estimate

            laxity = deadline - processing_time

            if laxity < min_laxity:
                min_laxity = laxity
                least_laxity = task

        self.task_queue.remove(least_laxity)
        return least_laxity

    def mark_task_complete(self, robot_id: int):
        """
        Mark task as completed.

        Args:
            robot_id: Robot ID
        """
        if robot_id in self.in_progress:
            task = self.in_progress.pop(robot_id)
            self.completed_tasks.append(task)

    def step(self):
        """Update scheduler (called every timestep)."""
        self.current_time += self.time_step

        # Check for deadline violations
        for robot_id, task in self.in_progress.items():
            if task[5] < 0:  # Deadline exceeded
                logger.warning(f"Task deadline exceeded for robot {robot_id}")

    def get_queue_status(self) -> Dict:
        """Get scheduler queue status."""
        return {
            'pending_tasks': len(self.task_queue),
            'in_progress': len(self.in_progress),
            'completed_tasks': len(self.completed_tasks),
            'current_time': self.current_time,
        }

    def get_statistics(self) -> Dict:
        """Get scheduling statistics."""
        if not self.completed_tasks:
            avg_completion_time = 0
            deadline_miss_rate = 0
        else:
            completion_times = []
            deadline_misses = 0

            for task in self.completed_tasks:
                deadline = task[5]
                if deadline < 0:
                    deadline_misses += 1

            deadline_miss_rate = deadline_misses / len(self.completed_tasks)

        return {
            'completed_tasks': len(self.completed_tasks),
            'pending_tasks': len(self.task_queue),
            'deadline_miss_rate': deadline_miss_rate,
            'scheduling_policy': self.scheduling_policy,
        }

    def reset(self):
        """Reset scheduler."""
        self.task_queue = []
        self.in_progress = {}
        self.completed_tasks = []
        self.current_time = 0


class ResourceAllocator:
    """
    Allocate shared resources (charging stations, work zones) to robots.

    Manages resource conflicts and contention.
    """

    def __init__(
        self,
        num_robots: int = 5,
        num_charging_stations: int = 2,
        num_work_zones: int = 5,
    ):
        """
        Initialize resource allocator.

        Args:
            num_robots: Number of robots
            num_charging_stations: Number of charging stations
            num_work_zones: Number of work zones
        """
        self.num_robots = num_robots
        self.num_charging_stations = num_charging_stations
        self.num_work_zones = num_work_zones

        # Resource allocations
        self.charging_allocations: Dict[int, Optional[int]] = {i: None for i in range(num_robots)}
        self.zone_allocations: Dict[int, Optional[int]] = {i: None for i in range(num_robots)}

        # Resource availability
        self.charging_available: Set[int] = set(range(num_charging_stations))
        self.zone_available: Set[int] = set(range(num_work_zones))

        logger.info(
            f"ResourceAllocator initialized: "
            f"{num_charging_stations} charging stations, "
            f"{num_work_zones} work zones"
        )

    def request_charging_station(self, robot_id: int) -> Optional[int]:
        """
        Request charging station for robot.

        Args:
            robot_id: Robot ID

        Returns:
            Station ID or None if unavailable
        """
        if not self.charging_available:
            return None

        station_id = self.charging_available.pop()
        self.charging_allocations[robot_id] = station_id

        return station_id

    def release_charging_station(self, robot_id: int):
        """
        Release charging station.

        Args:
            robot_id: Robot ID
        """
        if robot_id in self.charging_allocations:
            station_id = self.charging_allocations[robot_id]
            if station_id is not None:
                self.charging_available.add(station_id)
            self.charging_allocations[robot_id] = None

    def request_work_zone(self, robot_id: int) -> Optional[int]:
        """
        Request work zone for robot.

        Args:
            robot_id: Robot ID

        Returns:
            Zone ID or None if unavailable
        """
        if not self.zone_available:
            return None

        zone_id = self.zone_available.pop()
        self.zone_allocations[robot_id] = zone_id

        return zone_id

    def release_work_zone(self, robot_id: int):
        """
        Release work zone.

        Args:
            robot_id: Robot ID
        """
        if robot_id in self.zone_allocations:
            zone_id = self.zone_allocations[robot_id]
            if zone_id is not None:
                self.zone_available.add(zone_id)
            self.zone_allocations[robot_id] = None

    def get_resource_utilization(self) -> Dict:
        """Get resource utilization statistics."""
        charging_used = self.num_charging_stations - len(self.charging_available)
        zone_used = self.num_work_zones - len(self.zone_available)

        return {
            'charging_utilization': charging_used / max(1, self.num_charging_stations),
            'zone_utilization': zone_used / max(1, self.num_work_zones),
            'charging_available': len(self.charging_available),
            'zone_available': len(self.zone_available),
        }

    def reset(self):
        """Reset allocator."""
        self.charging_allocations = {i: None for i in range(self.num_robots)}
        self.zone_allocations = {i: None for i in range(self.num_robots)}
        self.charging_available = set(range(self.num_charging_stations))
        self.zone_available = set(range(self.num_work_zones))
