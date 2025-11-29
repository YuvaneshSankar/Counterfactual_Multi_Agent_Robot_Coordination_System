

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:


    def __init__(self, num_robots: int = 5):

        self.num_robots = num_robots

        # Metric storage
        self.episode_metrics: List[Dict] = []
        self.aggregate_metrics: Dict = {}

        logger.info("PerformanceMetrics initialized")

    def compute_episode_metrics(
        self,
        episode_data: Dict,
        robots: List,
        task_generator,
    ) -> Dict:

        metrics = {}

        # Task performance metrics
        metrics.update(self._compute_task_metrics(episode_data, task_generator))

        # Efficiency metrics
        metrics.update(self._compute_efficiency_metrics(episode_data, robots))

        # Safety metrics
        metrics.update(self._compute_safety_metrics(episode_data))

        # Resource metrics
        metrics.update(self._compute_resource_metrics(episode_data, robots))

        # Coordination metrics
        metrics.update(self._compute_coordination_metrics(episode_data, robots))

        self.episode_metrics.append(metrics)
        return metrics

    def _compute_task_metrics(self, episode_data: Dict, task_generator) -> Dict:
        completed_tasks = episode_data.get('completed_tasks', 0)
        total_tasks = len(task_generator.pending_tasks) + completed_tasks
        episode_steps = episode_data.get('steps', 1)

        # Success rate
        success_rate = completed_tasks / max(1, total_tasks)

        # Throughput (tasks per timestep)
        throughput = completed_tasks / max(1, episode_steps)

        # Average task completion time
        completion_times = episode_data.get('task_completion_times', [])
        avg_completion_time = np.mean(completion_times) if completion_times else 0

        return {
            'task_success_rate': success_rate,
            'task_throughput': throughput,
            'avg_task_completion_time': avg_completion_time,
            'completed_tasks': completed_tasks,
            'total_tasks': total_tasks,
        }

    def _compute_efficiency_metrics(self, episode_data: Dict, robots: List) -> Dict:
        episode_steps = episode_data.get('steps', 1)
        completed_tasks = episode_data.get('completed_tasks', 0)

        # Time per task
        time_per_task = episode_steps / max(1, completed_tasks)

        # Total distance traveled by all robots
        total_distance = 0.0
        for robot in robots:
            # Estimate distance from velocity history
            velocity_history = getattr(robot, 'velocity_history', [])
            if velocity_history:
                total_distance += sum(np.linalg.norm(v[:2]) for v in velocity_history)

        # Distance per task
        distance_per_task = total_distance / max(1, completed_tasks)

        # Makespan (total time to complete all tasks)
        makespan = episode_steps

        # Fleet utilization (average robots active)
        active_robots_per_step = episode_data.get('active_robots_per_step', [])
        fleet_utilization = np.mean(active_robots_per_step) / self.num_robots if active_robots_per_step else 0

        return {
            'time_per_task': time_per_task,
            'distance_per_task': distance_per_task,
            'total_distance': total_distance,
            'makespan': makespan,
            'fleet_utilization': fleet_utilization,
        }

    def _compute_safety_metrics(self, episode_data: Dict) -> Dict:
        """Compute safety-related metrics."""
        total_collisions = episode_data.get('collisions', 0)
        episode_steps = episode_data.get('steps', 1)

        # Collision rate
        collision_rate = total_collisions / max(1, episode_steps)

        # Near-misses (if tracked)
        near_misses = episode_data.get('near_misses', 0)
        near_miss_rate = near_misses / max(1, episode_steps)

        # Safety score (0-1, higher is safer)
        safety_score = max(0, 1.0 - collision_rate * 10)

        return {
            'collision_rate': collision_rate,
            'near_miss_rate': near_miss_rate,
            'total_collisions': total_collisions,
            'safety_score': safety_score,
        }

    def _compute_resource_metrics(self, episode_data: Dict, robots: List) -> Dict:
        """Compute resource utilization metrics."""
        # Battery metrics
        total_energy_consumed = sum(
            100 - robot.get_battery_level() for robot in robots
        )
        avg_battery_level = np.mean([robot.get_battery_level() for robot in robots])

        # Battery efficiency (distance per energy unit)
        total_distance = episode_data.get('total_distance', 0)
        battery_efficiency = total_distance / max(1, total_energy_consumed)

        # Charging events
        charging_events = episode_data.get('charging_events', 0)
        charging_overhead = charging_events / max(1, episode_data.get('steps', 1))

        return {
            'total_energy_consumed': total_energy_consumed,
            'avg_battery_level': avg_battery_level,
            'battery_efficiency': battery_efficiency,
            'charging_events': charging_events,
            'charging_overhead': charging_overhead,
        }

    def _compute_coordination_metrics(self, episode_data: Dict, robots: List) -> Dict:
        """Compute coordination quality metrics."""
        # Communication overhead
        messages_sent = episode_data.get('messages_sent', 0)
        episode_steps = episode_data.get('steps', 1)
        communication_overhead = messages_sent / max(1, episode_steps)

        # Task allocation quality
        task_switches = episode_data.get('task_switches', 0)
        allocation_stability = 1.0 - (task_switches / max(1, episode_steps))

        # Coordination score (based on various factors)
        fleet_utilization = episode_data.get('fleet_utilization', 0.5)
        collision_rate = episode_data.get('collision_rate', 0)
        coordination_score = fleet_utilization * (1.0 - collision_rate)

        return {
            'communication_overhead': communication_overhead,
            'allocation_stability': allocation_stability,
            'coordination_score': coordination_score,
        }

    def compute_aggregate_metrics(self, window: int = 100) -> Dict:

        if not self.episode_metrics:
            return {}

        recent_metrics = self.episode_metrics[-window:]

        aggregate = {}

        # Compute mean and std for each metric
        metric_keys = recent_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in recent_metrics]
            aggregate[f'{key}_mean'] = np.mean(values)
            aggregate[f'{key}_std'] = np.std(values)
            aggregate[f'{key}_min'] = np.min(values)
            aggregate[f'{key}_max'] = np.max(values)

        self.aggregate_metrics = aggregate
        return aggregate

    def get_summary_statistics(self) -> Dict:
        if not self.episode_metrics:
            return {}

        all_metrics = {}
        metric_keys = self.episode_metrics[0].keys()

        for key in metric_keys:
            values = [m[key] for m in self.episode_metrics]
            all_metrics[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1] if values else 0,
            }

        return all_metrics

    def compare_to_baseline(self, baseline_metrics: Dict) -> Dict:

        if not self.aggregate_metrics:
            self.compute_aggregate_metrics()

        comparison = {}

        for key, baseline_value in baseline_metrics.items():
            if key in self.aggregate_metrics:
                current_value = self.aggregate_metrics[key]
                improvement = ((current_value - baseline_value) / max(abs(baseline_value), 1e-6)) * 100

                comparison[key] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'improvement_percent': improvement,
                }

        return comparison

    def export_metrics(self, filepath: str) -> None:

        import json

        export_data = {
            'episode_metrics': self.episode_metrics,
            'aggregate_metrics': self.aggregate_metrics,
            'summary_statistics': self.get_summary_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")


class ComparisonMetrics:

    def __init__(self):
        self.experiments: Dict[str, PerformanceMetrics] = {}

    def add_experiment(self, name: str, metrics: PerformanceMetrics):
        self.experiments[name] = metrics

    def compare_experiments(self) -> Dict:
        comparison = {}

        if not self.experiments:
            return comparison

        # Get common metrics
        first_exp = list(self.experiments.values())[0]
        metric_keys = first_exp.aggregate_metrics.keys()

        for key in metric_keys:
            comparison[key] = {}
            for exp_name, metrics in self.experiments.items():
                comparison[key][exp_name] = metrics.aggregate_metrics.get(key, 0)

        return comparison

    def rank_experiments(self, metric: str = 'task_success_rate_mean') -> List[Tuple[str, float]]:

        rankings = []

        for exp_name, metrics in self.experiments.items():
            value = metrics.aggregate_metrics.get(metric, 0)
            rankings.append((exp_name, value))

        # Sort descending (higher is better)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings
