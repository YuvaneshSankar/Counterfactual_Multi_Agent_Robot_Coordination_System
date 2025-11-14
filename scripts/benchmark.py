#!/usr/bin/env python3
"""
Benchmark Script - Compare Different Configurations

Benchmarks COMAR with different hyperparameters and configurations.
"""

import argparse
import yaml
import torch
import logging
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.warehouse_env import WarehouseEnv
from src.algorithms.coma_continuous import COMAcontinuous
from src.visualization.metrics import PerformanceMetrics, ComparisonMetrics
from src.visualization.plots import PlotGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark COMAR')

    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        required=True,
        help='List of configuration files to benchmark'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        required=True,
        help='List of model checkpoints (same order as configs)'
    )
    parser.add_argument(
        '--names',
        type=str,
        nargs='+',
        help='Names for each configuration'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=20,
        help='Number of episodes per configuration'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmark',
        help='Output directory for results'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_configuration(
    config: dict,
    checkpoint_path: str,
    num_episodes: int = 20
) -> dict:
    """
    Evaluate a single configuration.

    Returns:
        Dictionary with performance metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    env = WarehouseEnv(config=config, render=False)

    # Create algorithm
    state_dim = env.state_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[1]
    num_agents = env.num_robots

    algorithm = COMAcontinuous(
        state_dim=state_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        config=config,
        device=device
    )

    # Load checkpoint
    algorithm.load_checkpoint(checkpoint_path)

    # Initialize metrics
    metrics = PerformanceMetrics(num_robots=num_agents)

    # Run episodes
    episode_data_list = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        episode_info = {
            'steps': 0,
            'completed_tasks': 0,
            'collisions': 0,
            'active_robots_per_step': [],
        }

        while not done:
            # Get actions
            action_list, _ = algorithm.select_actions(obs, deterministic=True)
            actions = np.array(action_list)

            # Step
            obs, reward, done, info = env.step(actions)

            episode_reward += reward
            episode_length += 1

            # Track metrics
            episode_info['steps'] = episode_length
            episode_info['completed_tasks'] = info.get('completed_tasks', 0)
            episode_info['collisions'] = info.get('collisions', 0)

            # Count active robots
            active = sum(1 for robot in env.robots if robot.has_task)
            episode_info['active_robots_per_step'].append(active)

        # Compute episode metrics
        ep_metrics = metrics.compute_episode_metrics(
            episode_info,
            env.robots,
            env.task_generator
        )

        episode_data_list.append({
            'reward': episode_reward,
            'length': episode_length,
            'metrics': ep_metrics
        })

        logger.info(
            f"Episode {episode + 1}/{num_episodes}: "
            f"reward={episode_reward:.2f}, tasks={ep_metrics['completed_tasks']}"
        )

    # Compute aggregate metrics
    aggregate = metrics.compute_aggregate_metrics(window=num_episodes)

    # Close environment
    env.close()

    return {
        'episodes': episode_data_list,
        'aggregate': aggregate,
        'summary': metrics.get_summary_statistics()
    }


def compare_results(
    results: dict,
    names: list
) -> dict:
    """
    Compare results from multiple configurations.

    Args:
        results: Dictionary mapping config_name -> results
        names: List of configuration names

    Returns:
        Comparison dictionary
    """
    comparison = {
        'mean_rewards': {},
        'std_rewards': {},
        'success_rates': {},
        'collision_rates': {},
        'efficiency': {},
    }

    for name in names:
        if name in results:
            summary = results[name]['summary']

            comparison['mean_rewards'][name] = summary['task_success_rate']['mean']
            comparison['std_rewards'][name] = summary['task_success_rate']['std']
            comparison['success_rates'][name] = summary['task_success_rate']['mean']
            comparison['collision_rates'][name] = summary['collision_rate']['mean']
            comparison['efficiency'][name] = summary.get('battery_efficiency', {}).get('mean', 0)

    return comparison


def generate_comparison_plots(
    results: dict,
    names: list,
    output_dir: str
):
    """Generate comparison plots."""
    plotter = PlotGenerator(save_dir=output_dir)

    # Extract rewards for each config
    reward_data = {}
    for name in names:
        if name in results:
            rewards = [ep['reward'] for ep in results[name]['episodes']]
            reward_data[name] = rewards

    # Plot comparison
    plotter.plot_comparison(
        experiment_names=names,
        metric_values=reward_data,
        metric_name='Episode Reward',
        save_name='reward_comparison.png'
    )

    # Bar comparison
    means = [np.mean(reward_data[name]) for name in names if name in reward_data]
    stds = [np.std(reward_data[name]) for name in names if name in reward_data]

    plotter.plot_bar_comparison(
        experiment_names=names,
        metric_means=means,
        metric_stds=stds,
        metric_name='Mean Reward',
        save_name='bar_comparison.png'
    )

    # Box plot
    plotter.plot_box_plot(
        data_dict=reward_data,
        ylabel='Episode Reward',
        title='Reward Distribution Comparison',
        save_name='boxplot_comparison.png'
    )

    logger.info(f"Plots saved to {output_dir}")


def save_results(
    results: dict,
    comparison: dict,
    output_dir: str
):
    """Save results to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save full results
    results_path = f"{output_dir}/benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save comparison
    comparison_path = f"{output_dir}/comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def print_comparison(comparison: dict, names: list):
    """Print comparison table."""
    logger.info("\n" + "=" * 100)
    logger.info("BENCHMARK COMPARISON")
    logger.info("=" * 100)

    # Header
    logger.info(f"{'Configuration':<30} {'Mean Reward':<15} {'Success Rate':<15} {'Collision Rate':<15}")
    logger.info("-" * 100)

    # Data
    for name in names:
        mean_reward = comparison['mean_rewards'].get(name, 0)
        success = comparison['success_rates'].get(name, 0)
        collision = comparison['collision_rates'].get(name, 0)

        logger.info(f"{name:<30} {mean_reward:<15.4f} {success:<15.4f} {collision:<15.4f}")

    logger.info("=" * 100 + "\n")

    # Best configuration
    best_config = max(names, key=lambda n: comparison['mean_rewards'].get(n, -float('inf')))
    logger.info(f"🏆 Best Configuration: {best_config}")
    logger.info(f"   Mean Reward: {comparison['mean_rewards'][best_config]:.4f}")
    logger.info(f"   Success Rate: {comparison['success_rates'][best_config]:.4f}\n")


def main():
    """Main benchmark function."""
    args = parse_args()

    # Validate inputs
    if len(args.configs) != len(args.checkpoints):
        raise ValueError("Number of configs must match number of checkpoints")

    # Generate names if not provided
    if args.names is None:
        args.names = [f"Config_{i+1}" for i in range(len(args.configs))]
    elif len(args.names) != len(args.configs):
        raise ValueError("Number of names must match number of configs")

    logger.info(f"Benchmarking {len(args.configs)} configurations...")
    logger.info(f"Output directory: {args.output_dir}")

    # Evaluate each configuration
    results = {}

    for i, (config_path, checkpoint_path, name) in enumerate(zip(
        args.configs, args.checkpoints, args.names
    )):
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {name} ({i+1}/{len(args.configs)})")
        logger.info(f"Config: {config_path}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"{'='*80}\n")

        config = load_config(config_path)

        result = evaluate_configuration(
            config=config,
            checkpoint_path=checkpoint_path,
            num_episodes=args.num_episodes
        )

        results[name] = result

    # Compare results
    comparison = compare_results(results, args.names)

    # Print comparison
    print_comparison(comparison, args.names)

    # Generate plots
    generate_comparison_plots(results, args.names, args.output_dir)

    # Save results
    save_results(results, comparison, args.output_dir)

    logger.info("Benchmark completed!")


if __name__ == '__main__':
    main()
