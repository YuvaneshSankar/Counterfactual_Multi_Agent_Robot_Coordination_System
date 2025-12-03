import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PolicyEvaluator:

    def __init__(self, num_robots: int = 5):
        self.num_robots = num_robots
        self.evaluation_history: List[Dict] = []

    def evaluate_policy(
        self,
        env,
        algorithm,
        num_episodes: int = 10,
        render: bool = False,
    ) -> Dict:
        metrics = {
            'episode_rewards': [],
            'success_rates': [],
            'collision_rates': [],
            'completion_times': [],
            'battery_efficiency': [],
            'fleet_utilization': [],
        }

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_metrics = {
                'reward': 0.0,
                'tasks_completed': 0,
                'collisions': 0,
                'steps': 0,
                'total_battery_used': 0.0,
            }

            done = False
            while not done:
                # Get action from policy (deterministic)
                action_list, _ = algorithm.select_actions(obs, deterministic=True)
                actions = np.array(action_list)

                # Step environment
                obs, reward, done, info = env.step(actions)

                # Track metrics
                episode_metrics['reward'] += reward
                episode_metrics['tasks_completed'] += info.get('completed_tasks', 0)
                episode_metrics['collisions'] += info.get('collisions', 0)
                episode_metrics['steps'] += 1

                if render:
                    env.render()

            # Compute episode metrics
            success_rate = episode_metrics['tasks_completed'] / max(1, env.current_step)
            collision_rate = episode_metrics['collisions'] / max(1, episode_metrics['steps'])

            metrics['episode_rewards'].append(episode_metrics['reward'])
            metrics['success_rates'].append(success_rate)
            metrics['collision_rates'].append(collision_rate)
            metrics['completion_times'].append(episode_metrics['steps'])

        # Aggregate metrics
        aggregated = {
            'mean_reward': np.mean(metrics['episode_rewards']),
            'std_reward': np.std(metrics['episode_rewards']),
            'mean_success_rate': np.mean(metrics['success_rates']),
            'mean_collision_rate': np.mean(metrics['collision_rates']),
            'mean_completion_time': np.mean(metrics['completion_times']),
        }

        self.evaluation_history.append(aggregated)

        return aggregated

    def compute_metrics_from_trajectory(
        self,
        rewards: List[float],
        info_list: List[Dict],
    ) -> Dict:
        total_reward = np.sum(rewards)
        avg_reward_per_step = np.mean(rewards)

        collisions = sum(info.get('collisions', 0) for info in info_list)
        collision_rate = collisions / max(1, len(info_list))

        tasks_completed = sum(info.get('completed_tasks', 0) for info in info_list)
        completion_rate = tasks_completed / max(1, len(info_list))

        return {
            'total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'collision_rate': collision_rate,
            'completion_rate': completion_rate,
            'num_steps': len(info_list),
        }

    def get_summary(self) -> Dict:
        if not self.evaluation_history:
            return {}

        history = np.array([list(d.values()) for d in self.evaluation_history])

        return {
            'num_evaluations': len(self.evaluation_history),
            'mean_reward': np.mean(history[:, 0]),
            'max_reward': np.max(history[:, 0]),
            'min_reward': np.min(history[:, 0]),
        }


class PerformanceMonitor:

    def __init__(self):
        self.metrics_history: Dict = {}

    def record_metric(self, name: str, value: float, step: int):
        if name not in self.metrics_history:
            self.metrics_history[name] = []

        self.metrics_history[name].append({
            'step': step,
            'value': value,
        })

    def get_metric_history(self, name: str) -> List[Dict]:
        return self.metrics_history.get(name, [])

    def get_latest_metrics(self) -> Dict:
        latest = {}
        for name, history in self.metrics_history.items():
            if history:
                latest[name] = history[-1]['value']
        return latest

    def get_mean_metrics(self, window_size: int = 100) -> Dict:
        means = {}
        for name, history in self.metrics_history.items():
            if len(history) > window_size:
                values = [h['value'] for h in history[-window_size:]]
                means[name] = np.mean(values)
            elif history:
                values = [h['value'] for h in history]
                means[name] = np.mean(values)
        return means

    def get_statistics(self) -> Dict:
        stats = {}
        for name, history in self.metrics_history.items():
            if history:
                values = [h['value'] for h in history]
                stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1],
                }
        return stats


class MetricComputer:

    @staticmethod
    def compute_success_rate(completed_tasks: int, total_tasks: int) -> float:
        return completed_tasks / max(1, total_tasks)

    @staticmethod
    def compute_collision_rate(num_collisions: int, num_steps: int) -> float:
        return num_collisions / max(1, num_steps)

    @staticmethod
    def compute_energy_efficiency(energy_used: float, distance_traveled: float) -> float:
        return distance_traveled / max(1, energy_used)

    @staticmethod
    def compute_fleet_utilization(robots_active: int, total_robots: int) -> float:
        return robots_active / max(1, total_robots)

    @staticmethod
    def compute_average_task_time(total_time: float, num_tasks: int) -> float:
        return total_time / max(1, num_tasks)

    @staticmethod
    def compute_makespan(task_times: List[float]) -> float:
        return sum(task_times) if task_times else 0

    @staticmethod
    def compute_arrival_time_variance(arrival_times: List[float]) -> float:
        return np.var(arrival_times) if arrival_times else 0
