

import numpy as np
from typing import Dict, List, Callable, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingCallback:

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_episode_begin(self, episode: int):
        pass

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        pass

    def on_step(self, step: int, action, reward: float, done: bool, info: Dict):
        pass

    def on_update(self, update_num: int, update_info: Dict):
        pass


class LoggingCallback(TrainingCallback):


    def __init__(self, log_frequency: int = 100):

        self.log_frequency = log_frequency
        self.step_count = 0
        self.episode_count = 0

    def on_step(self, step: int, action, reward: float, done: bool, info: Dict):
        self.step_count += 1

        if self.step_count % self.log_frequency == 0:
            logger.info(
                f"Step {self.step_count}: reward={reward:.3f}, "
                f"active_tasks={info.get('active_tasks', 0)}"
            )

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        self.episode_count += 1

        logger.info(
            f"Episode {self.episode_count}: "
            f"total_reward={reward:.2f}, "
            f"completed_tasks={info.get('completed_tasks', 0)}, "
            f"collisions={info.get('collisions', 0)}"
        )


class CheckpointCallback(TrainingCallback):

    def __init__(
        self,
        checkpoint_dir: str = 'results/checkpoints',
        save_frequency: int = 10000,
        save_best: bool = True,
    ):

        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.save_best = save_best

        self.step_count = 0
        self.best_reward = -np.inf

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_step(self, step: int, action, reward: float, done: bool, info: Dict):
        self.step_count += 1

        if self.step_count % self.save_frequency == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.checkpoint_dir, f'model_{self.step_count}_{timestamp}.pt')
            logger.info(f"Saving checkpoint: {path}")

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        if self.save_best and reward > self.best_reward:
            self.best_reward = reward
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            logger.info(f"Saving best model with reward {reward:.2f}")


class EarlyStoppingCallback(TrainingCallback):

    def __init__(
        self,
        metric: str = 'reward',
        threshold: float = 0.9,
        patience: int = 10,
    ):

        self.metric = metric
        self.threshold = threshold
        self.patience = patience

        self.best_metric = -np.inf
        self.patience_counter = 0
        self.should_stop = False

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        metric_value = reward

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered at episode {episode}")


class MetricsCallback(TrainingCallback):

    def __init__(self, log_dir: str = 'results/logs'):

        self.log_dir = log_dir
        self.metrics_history: Dict = {}

        os.makedirs(log_dir, exist_ok=True)

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        if 'episode_rewards' not in self.metrics_history:
            self.metrics_history['episode_rewards'] = []

        self.metrics_history['episode_rewards'].append({
            'episode': episode,
            'reward': reward,
            'completed_tasks': info.get('completed_tasks', 0),
            'collisions': info.get('collisions', 0),
        })

    def on_train_end(self, trainer):
        """Save metrics to file."""
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')

        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")


class CallbackManager:

    def __init__(self):
        self.callbacks: List[TrainingCallback] = []

    def add_callback(self, callback: TrainingCallback):
        self.callbacks.append(callback)

    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_episode_begin(self, episode: int):
        for callback in self.callbacks:
            callback.on_episode_begin(episode)

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        for callback in self.callbacks:
            callback.on_episode_end(episode, reward, info)

    def on_step(self, step: int, action, reward: float, done: bool, info: Dict):
        for callback in self.callbacks:
            callback.on_step(step, action, reward, done, info)

    def on_update(self, update_num: int, update_info: Dict):
        for callback in self.callbacks:
            callback.on_update(update_num, update_info)


class CustomCallback(TrainingCallback):

    def __init__(
        self,
        on_step_fn: Optional[Callable] = None,
        on_episode_end_fn: Optional[Callable] = None,
        on_train_end_fn: Optional[Callable] = None,
    ):

        self.on_step_fn = on_step_fn
        self.on_episode_end_fn = on_episode_end_fn
        self.on_train_end_fn = on_train_end_fn

    def on_step(self, step: int, action, reward: float, done: bool, info: Dict):
        if self.on_step_fn:
            self.on_step_fn(step, action, reward, done, info)

    def on_episode_end(self, episode: int, reward: float, info: Dict):
        if self.on_episode_end_fn:
            self.on_episode_end_fn(episode, reward, info)

    def on_train_end(self, trainer):
        if self.on_train_end_fn:
            self.on_train_end_fn(trainer)
