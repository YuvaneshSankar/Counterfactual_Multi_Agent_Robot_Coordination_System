#!/usr/bin/env python3
"""
Training Script - Main Entry Point for COMAR Training

Trains COMA agents in the warehouse environment with full monitoring.
"""

import argparse
import yaml
import torch
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.warehouse_env import WarehouseEnv
from src.algorithms.coma_continuous import COMAcontinuous
from src.training.trainer import COMATrainer
from src.training.curriculum import CurriculumLearning
from src.training.callbacks import (
    CallbackManager,
    LoggingCallback,
    CheckpointCallback,
    MetricsCallback,
    EarlyStoppingCallback
)
from src.visualization.dashboard import create_dashboard
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train COMAR agents')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable rendering'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable web dashboard'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"Random seed set to {seed}")


def create_environment(config: dict, render: bool = False):
    """Create warehouse environment."""
    env = WarehouseEnv(config=config, render=render)
    logger.info(f"Environment created with {env.num_robots} robots")
    return env


def create_algorithm(config: dict, env, device: torch.device):
    """Create COMA algorithm."""
    # Get dimensions from environment
    state_dim = env.state_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[1]
    num_agents = env.num_robots

    # Create algorithm
    algorithm = COMAcontinuous(
        state_dim=state_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        config=config,
        device=device
    )

    logger.info(f"COMA algorithm initialized on {device}")
    return algorithm


def setup_callbacks(config: dict) -> CallbackManager:
    """Setup training callbacks."""
    callback_manager = CallbackManager()

    # Logging callback
    callback_manager.add_callback(
        LoggingCallback(log_frequency=100)
    )

    # Checkpoint callback
    callback_manager.add_callback(
        CheckpointCallback(
            checkpoint_dir=config.get('logging', {}).get('checkpoint_dir', 'results/checkpoints'),
            save_frequency=10000,
            save_best=True
        )
    )

    # Metrics callback
    callback_manager.add_callback(
        MetricsCallback(
            log_dir=config.get('logging', {}).get('log_dir', 'results/logs')
        )
    )

    # Early stopping (optional)
    if config.get('training', {}).get('early_stopping', False):
        callback_manager.add_callback(
            EarlyStoppingCallback(
                patience=config.get('training', {}).get('early_stopping_patience', 10)
            )
        )

    logger.info("Callbacks configured")
    return callback_manager


def start_dashboard(log_dir: str, port: int = 8050):
    """Start monitoring dashboard in background thread."""
    dashboard = create_dashboard(log_dir=log_dir, port=port)
    dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
    dashboard_thread.start()
    logger.info(f"Dashboard started at http://localhost:{port}")
    return dashboard


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Start dashboard if requested
    if args.dashboard:
        log_dir = config.get('logging', {}).get('log_dir', 'results/logs')
        start_dashboard(log_dir)

    # Create environment
    env = create_environment(config, render=args.render)

    # Create algorithm
    algorithm = create_algorithm(config, env, device)

    # Load checkpoint if provided
    if args.checkpoint:
        algorithm.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Setup curriculum learning
    curriculum = CurriculumLearning(config)

    # Setup callbacks
    callback_manager = setup_callbacks(config)

    # Create trainer
    trainer = COMATrainer(
        config=config,
        env=env,
        algorithm=algorithm,
        device=device
    )

    # Start training
    logger.info("=" * 80)
    logger.info("Starting COMAR Training")
    logger.info("=" * 80)

    try:
        stats = trainer.train()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best reward: {trainer.best_reward:.2f}")
        logger.info(f"Total episodes: {trainer.total_episodes}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer._save_checkpoint('interrupted_model')

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        env.close()
        logger.info("Environment closed")


if __name__ == '__main__':
    main()
