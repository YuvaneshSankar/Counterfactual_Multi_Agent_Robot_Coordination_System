

import argparse
import yaml
import torch
import logging
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.warehouse_env import WarehouseEnv
from src.algorithms.coma_continuous import COMAcontinuous
from src.training.evaluator import PolicyEvaluator
from src.visualization.metrics import PerformanceMetrics
from src.visualization.plots import PlotGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate COMAR agents')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable rendering'
    )
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save video of episodes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_policy(
    env,
    algorithm,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
):
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    collision_counts = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        logger.info(f"Evaluating episode {episode + 1}/{num_episodes}")

        while not done:
            # Get actions
            action_list, _ = algorithm.select_actions(obs, deterministic=deterministic)
            actions = np.array(action_list)

            # Step environment
            obs, reward, done, info = env.step(actions)

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rates.append(info.get('completed_tasks', 0) / max(1, episode_length))
        collision_counts.append(info.get('collisions', 0))

        logger.info(
            f"Episode {episode + 1}: reward={episode_reward:.2f}, "
            f"length={episode_length}, tasks={info.get('completed_tasks', 0)}, "
            f"collisions={info.get('collisions', 0)}"
        )

    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_success_rate': np.mean(success_rates),
        'mean_collisions': np.mean(collision_counts),
        'episode_rewards': episode_rewards,
    }

    return results


def print_results(results: dict):
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"Min Reward: {results['min_reward']:.2f}")
    logger.info(f"Max Reward: {results['max_reward']:.2f}")
    logger.info(f"Mean Episode Length: {results['mean_length']:.2f}")
    logger.info(f"Mean Success Rate: {results['mean_success_rate']:.4f}")
    logger.info(f"Mean Collisions: {results['mean_collisions']:.2f}")
    logger.info("=" * 80 + "\n")


def save_results(results: dict, output_path: str = 'results/evaluation.json'):
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def plot_results(results: dict, save_dir: str = 'results/plots'):
    plotter = PlotGenerator(save_dir=save_dir)

    # Plot episode rewards
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['episode_rewards'], marker='o', linewidth=2)
    ax.axhline(
        results['mean_reward'],
        color='red',
        linestyle='--',
        label=f"Mean: {results['mean_reward']:.2f}"
    )
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Evaluation Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_rewards.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {save_dir}")


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create environment
    env = WarehouseEnv(config=config, render=args.render)
    logger.info(f"Environment created with {env.num_robots} robots")

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
    algorithm.load_checkpoint(args.checkpoint)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Evaluate
    logger.info(f"Starting evaluation for {args.num_episodes} episodes...")

    results = evaluate_policy(
        env=env,
        algorithm=algorithm,
        num_episodes=args.num_episodes,
        render=args.render,
        deterministic=args.deterministic
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results)

    # Generate plots
    plot_results(results)

    # Close environment
    env.close()
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
