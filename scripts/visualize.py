#!/usr/bin/env python3

import argparse
import yaml
import torch
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.warehouse_env import WarehouseEnv
from src.algorithms.coma_continuous import COMAcontinuous
from src.visualization.renderer import PyBulletRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COMAR agents')

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
        help='Path to trained model'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=5,
        help='Number of episodes to visualize'
    )
    parser.add_argument(
        '--follow-robot',
        type=int,
        default=None,
        help='Robot ID to follow with camera'
    )
    parser.add_argument(
        '--show-paths',
        action='store_true',
        help='Show planned paths'
    )
    parser.add_argument(
        '--show-communication',
        action='store_true',
        help='Show communication links'
    )
    parser.add_argument(
        '--slow-motion',
        type=float,
        default=1.0,
        help='Slow motion factor (< 1.0 for slower)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_episode(
    env,
    algorithm,
    renderer: PyBulletRenderer,
    follow_robot_id: int = None,
    show_paths: bool = False,
    show_communication: bool = False,
    slow_motion: float = 1.0
):
    import time

    obs, _ = env.reset()
    episode_reward = 0.0
    step_count = 0
    done = False

    logger.info("Starting episode visualization...")

    while not done:
        # Get actions
        action_list, _ = algorithm.select_actions(obs, deterministic=True)
        actions = np.array(action_list)

        # Step environment
        obs, reward, done, info = env.step(actions)
        episode_reward += reward
        step_count += 1

        # Clear previous debug items
        renderer.clear_debug_items()

        # Render robots with battery indicators
        renderer.render_robots(env.robots)

        # Render tasks
        renderer.render_tasks(env.task_generator.pending_tasks)

        # Render charging stations
        if hasattr(env.warehouse_layout, 'charging_stations'):
            renderer.render_charging_stations(env.warehouse_layout.charging_stations)

        # Show paths if requested
        if show_paths:
            for robot in env.robots:
                if robot.has_task:
                    # Create simple path to task
                    task = robot.current_task
                    if task:
                        pickup_loc = np.array(task[:2])
                        robot_pos = robot.get_position()[:2]
                        path = [robot_pos, pickup_loc]
                        renderer.render_paths(robot.robot_id, path)

        # Show communication if requested
        if show_communication:
            # Build connectivity (robots within range)
            connectivity = {}
            for i, robot_i in enumerate(env.robots):
                connectivity[i] = []
                pos_i = robot_i.get_position()[:2]
                for j, robot_j in enumerate(env.robots):
                    if i != j:
                        pos_j = robot_j.get_position()[:2]
                        if np.linalg.norm(pos_i - pos_j) < 30.0:
                            connectivity[i].append(j)

            renderer.render_communication_links(env.robots, connectivity)

        # Follow robot if specified
        if follow_robot_id is not None and follow_robot_id < len(env.robots):
            renderer.follow_robot(env.robots[follow_robot_id])

        # Render text overlay with stats
        renderer.render_text_overlay(
            f"Step: {step_count} | Reward: {episode_reward:.1f} | Tasks: {info.get('completed_tasks', 0)}",
            (env.warehouse_width / 2, env.warehouse_height / 2)
        )

        # Slow motion
        if slow_motion < 1.0:
            time.sleep(0.01 / slow_motion)

    logger.info(f"Episode finished: reward={episode_reward:.2f}, steps={step_count}")
    return episode_reward


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment with rendering
    env = WarehouseEnv(config=config, render=True)
    logger.info(f"Environment created with {env.num_robots} robots")

    # Create renderer
    renderer = PyBulletRenderer(
        client=env.client,
        warehouse_width=env.warehouse_width,
        warehouse_height=env.warehouse_height
    )

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
    logger.info(f"Loaded model from {args.checkpoint}")

    # Visualize episodes
    logger.info(f"\nVisualizing {args.num_episodes} episodes...")
    logger.info(f"Follow robot: {args.follow_robot}")
    logger.info(f"Show paths: {args.show_paths}")
    logger.info(f"Show communication: {args.show_communication}")
    logger.info(f"Slow motion: {args.slow_motion}x\n")

    episode_rewards = []

    try:
        for episode in range(args.num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode + 1}/{args.num_episodes}")
            logger.info(f"{'='*60}")

            reward = visualize_episode(
                env=env,
                algorithm=algorithm,
                renderer=renderer,
                follow_robot_id=args.follow_robot,
                show_paths=args.show_paths,
                show_communication=args.show_communication,
                slow_motion=args.slow_motion
            )

            episode_rewards.append(reward)

    except KeyboardInterrupt:
        logger.info("\nVisualization stopped by user")

    finally:
        # Print summary
        if episode_rewards:
            logger.info(f"\n{'='*60}")
            logger.info("SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Episodes: {len(episode_rewards)}")
            logger.info(f"Mean reward: {np.mean(episode_rewards):.2f}")
            logger.info(f"Std reward: {np.std(episode_rewards):.2f}")
            logger.info(f"{'='*60}\n")

        env.close()
        logger.info("Visualization completed!")


if __name__ == '__main__':
    main()
