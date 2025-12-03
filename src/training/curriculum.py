import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CurriculumLearning:

    def __init__(self, config: Dict):
        self.config = config

        # Extract curriculum config
        curriculum_config = config.get('curriculum', {})
        self.enabled = curriculum_config.get('enabled', False)

        # Define stages
        stages_config = curriculum_config.get('stages', {})
        self.stages = [
            {
                'name': 'stage1',
                'num_robots': stages_config.get('stage1', {}).get('num_robots', 2),
                'task_arrival_rate': stages_config.get('stage1', {}).get('task_arrival_rate', 0.1),
                'duration': stages_config.get('stage1', {}).get('duration', 100000),
            },
            {
                'name': 'stage2',
                'num_robots': stages_config.get('stage2', {}).get('num_robots', 5),
                'task_arrival_rate': stages_config.get('stage2', {}).get('task_arrival_rate', 0.5),
                'duration': stages_config.get('stage2', {}).get('duration', 200000),
            },
            {
                'name': 'stage3',
                'num_robots': stages_config.get('stage3', {}).get('num_robots', 10),
                'task_arrival_rate': stages_config.get('stage3', {}).get('task_arrival_rate', 1.0),
                'duration': stages_config.get('stage3', {}).get('duration', 200000),
            },
        ]

        # Current stage
        self.current_stage = 0
        self.steps_in_stage = 0
        self.total_steps = 0

        logger.info(f"CurriculumLearning initialized with {len(self.stages)} stages")

    def get_current_stage(self) -> Dict:
        return self.stages[self.current_stage].copy()

    def update_step(self) -> Optional[Dict]:
        self.steps_in_stage += 1
        self.total_steps += 1

        if not self.enabled or self.current_stage >= len(self.stages) - 1:
            return None

        # Check if should transition to next stage
        current_stage_config = self.stages[self.current_stage]
        if self.steps_in_stage >= current_stage_config['duration']:
            self._transition_to_next_stage()
            return self.get_current_stage()

        return None

    def _transition_to_next_stage(self):
        old_stage = self.current_stage
        self.current_stage += 1
        self.steps_in_stage = 0

        logger.info(
            f"Curriculum transition: {self.stages[old_stage]['name']} -> "
            f"{self.stages[self.current_stage]['name']}"
        )

    def should_increase_difficulty(self, mean_reward: float, threshold: float = 0.8) -> bool:
        if not self.enabled or self.current_stage >= len(self.stages) - 1:
            return False

        # Heuristic: if mean reward exceeds threshold, can increase difficulty
        return mean_reward > threshold

    def get_difficulty_level(self) -> float:
        if not self.enabled:
            return 1.0

        return (self.current_stage + 1) / len(self.stages)

    def get_progress(self) -> Dict:
        current_config = self.stages[self.current_stage]

        return {
            'current_stage': self.current_stage,
            'stage_name': current_config['name'],
            'steps_in_stage': self.steps_in_stage,
            'stage_duration': current_config['duration'],
            'total_steps': self.total_steps,
            'difficulty_level': self.get_difficulty_level(),
            'num_robots': current_config['num_robots'],
            'task_arrival_rate': current_config['task_arrival_rate'],
        }

    def reset(self):
        self.current_stage = 0
        self.steps_in_stage = 0
        self.total_steps = 0


class AdaptiveCurriculum:

    def __init__(
        self,
        initial_config: Dict,
        performance_threshold: float = 0.85,
        patience: int = 10000,
    ):
        self.config = initial_config
        self.performance_threshold = performance_threshold
        self.patience = patience
        self.steps_since_advance = 0

        # Performance tracking
        self.recent_rewards: List[float] = []
        self.max_recent_reward = -np.inf

    def record_performance(self, reward: float):
        self.recent_rewards.append(reward)

        # Keep recent window
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

        self.max_recent_reward = max(self.max_recent_reward, reward)
        self.steps_since_advance += 1

    def should_advance(self) -> bool:
        if len(self.recent_rewards) < 50:
            return False

        mean_reward = np.mean(self.recent_rewards)

        # Advance if threshold met or patience exceeded
        performance_met = mean_reward > self.performance_threshold
        patience_exceeded = self.steps_since_advance > self.patience

        return performance_met or patience_exceeded

    def get_current_difficulty(self) -> Dict:
        mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0

        # Adjust difficulty based on performance
        difficulty_factor = min(1.0, max(0, mean_reward / 10.0))

        return {
            'difficulty_factor': difficulty_factor,
            'mean_reward': mean_reward,
            'current_performance': len(self.recent_rewards),
        }
