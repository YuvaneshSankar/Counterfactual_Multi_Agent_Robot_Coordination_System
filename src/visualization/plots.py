

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class PlotGenerator:
    """
    Generate static plots for analysis and publication.

    Plot Types:
    - Training curves (rewards, losses)
    - Performance metrics over time
    - Comparative analysis
    - Distribution plots
    - Heatmaps
    """

    def __init__(self, save_dir: str = 'results/plots'):
        """
        Initialize plot generator.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"PlotGenerator initialized: {save_dir}")

    def plot_training_curves(
        self,
        steps: List[int],
        rewards: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
        save_name: str = 'training_curves.png'
    ) -> None:
        """
        Plot training curves (rewards and losses).

        Args:
            steps: Training steps
            rewards: Episode rewards
            actor_losses: Actor losses
            critic_losses: Critic losses
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(steps, rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= 100:
            window = 100
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(steps[window-1:], smoothed, color='darkblue', linewidth=2, label='Smoothed (100)')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Actor loss
        axes[0, 1].plot(actor_losses, color='green', linewidth=1.5)
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Actor Loss')
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Critic loss
        axes[1, 0].plot(critic_losses, color='red', linewidth=1.5)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Critic Loss')
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Reward distribution (histogram)
        axes[1, 1].hist(rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Episode Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Training curves saved to {save_path}")

    def plot_performance_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_name: str = 'performance_metrics.png'
    ) -> None:
        """
        Plot multiple performance metrics.

        Args:
            metrics: Dictionary of metric_name -> values
            save_name: Filename to save
        """
        num_metrics = len(metrics)
        cols = 2
        rows = (num_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if num_metrics > 1 else [axes]

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]

            # Plot raw values
            ax.plot(values, alpha=0.5, linewidth=1)

            # Plot smoothed if enough data
            if len(values) >= 50:
                window = min(50, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(values)), smoothed, linewidth=2, label='Smoothed')

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance metrics saved to {save_path}")

    def plot_comparison(
        self,
        experiment_names: List[str],
        metric_values: Dict[str, List[float]],
        metric_name: str = 'Success Rate',
        save_name: str = 'comparison.png'
    ) -> None:
        """
        Plot comparison between experiments.

        Args:
            experiment_names: List of experiment names
            metric_values: Dictionary of experiment_name -> metric values
            metric_name: Name of metric being compared
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for exp_name in experiment_names:
            if exp_name in metric_values:
                values = metric_values[exp_name]
                ax.plot(values, label=exp_name, linewidth=2, alpha=0.8)

        ax.set_xlabel('Episode')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plot saved to {save_path}")

    def plot_bar_comparison(
        self,
        experiment_names: List[str],
        metric_means: List[float],
        metric_stds: List[float],
        metric_name: str = 'Mean Reward',
        save_name: str = 'bar_comparison.png'
    ) -> None:
        """
        Plot bar chart comparison with error bars.

        Args:
            experiment_names: List of experiment names
            metric_means: Mean values for each experiment
            metric_stds: Standard deviations
            metric_name: Name of metric
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(experiment_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_names)))

        bars = ax.bar(
            x_pos,
            metric_means,
            yerr=metric_stds,
            color=colors,
            alpha=0.8,
            capsize=5,
            edgecolor='black'
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, metric_means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{mean:.2f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Bar comparison saved to {save_path}")

    def plot_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = 'Heatmap',
        save_name: str = 'heatmap.png'
    ) -> None:
        """
        Plot heatmap.

        Args:
            data: 2D array of values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=15)

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(
                    j, i, f'{data[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if data[i, j] > data.max()/2 else 'black'
                )

        ax.set_title(title)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Heatmap saved to {save_path}")

    def plot_box_plot(
        self,
        data_dict: Dict[str, List[float]],
        ylabel: str = 'Value',
        title: str = 'Box Plot Comparison',
        save_name: str = 'boxplot.png'
    ) -> None:
        """
        Plot box plot comparison.

        Args:
            data_dict: Dictionary of label -> data values
            ylabel: Y-axis label
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        labels = list(data_dict.keys())
        data = [data_dict[label] for label in labels]

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Box plot saved to {save_path}")

    def plot_correlation_matrix(
        self,
        metrics_df,
        save_name: str = 'correlation.png'
    ) -> None:
        """
        Plot correlation matrix of metrics.

        Args:
            metrics_df: Pandas DataFrame with metrics
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        correlation = metrics_df.corr()

        sns.heatmap(
            correlation,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )

        ax.set_title('Metric Correlation Matrix')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation matrix saved to {save_path}")
