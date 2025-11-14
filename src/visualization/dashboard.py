"""
Dashboard - Real-Time Training Monitoring

Web-based dashboard for monitoring training progress, metrics, and performance.
Uses Plotly Dash for interactive visualizations.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """
    Real-time training monitoring dashboard.

    Displays:
    - Episode rewards over time
    - Loss curves (actor and critic)
    - Success rate and collision rate
    - Battery efficiency
    - Task completion metrics
    """

    def __init__(
        self,
        log_dir: str = 'results/logs',
        port: int = 8050,
        update_interval: int = 1000,
    ):
        """
        Initialize dashboard.

        Args:
            log_dir: Directory containing training logs
            port: Port to run dashboard on
            update_interval: Update interval in milliseconds
        """
        self.log_dir = log_dir
        self.port = port
        self.update_interval = update_interval

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

        # Data storage
        self.metrics_data = {
            'steps': [],
            'episode_rewards': [],
            'actor_loss': [],
            'critic_loss': [],
            'success_rate': [],
            'collision_rate': [],
            'battery_efficiency': [],
        }

        logger.info(f"Dashboard initialized on port {port}")

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1(
                'COMAR Training Dashboard',
                style={'textAlign': 'center', 'color': '#2c3e50'}
            ),

            # Status indicators
            html.Div([
                html.Div([
                    html.H3('Training Status', style={'color': '#27ae60'}),
                    html.P(id='status-text', children='Running...'),
                ], className='status-card'),

                html.Div([
                    html.H3('Total Steps', style={'color': '#3498db'}),
                    html.P(id='steps-text', children='0'),
                ], className='status-card'),

                html.Div([
                    html.H3('Mean Reward', style={'color': '#e74c3c'}),
                    html.P(id='reward-text', children='0.0'),
                ], className='status-card'),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),

            # Graphs
            html.Div([
                # Episode rewards
                dcc.Graph(id='reward-graph'),

                # Loss curves
                html.Div([
                    dcc.Graph(id='actor-loss-graph', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='critic-loss-graph', style={'width': '50%', 'display': 'inline-block'}),
                ]),

                # Performance metrics
                html.Div([
                    dcc.Graph(id='success-rate-graph', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='collision-rate-graph', style={'width': '50%', 'display': 'inline-block'}),
                ]),

                # Battery efficiency
                dcc.Graph(id='battery-graph'),
            ]),

            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [
                Output('status-text', 'children'),
                Output('steps-text', 'children'),
                Output('reward-text', 'children'),
                Output('reward-graph', 'figure'),
                Output('actor-loss-graph', 'figure'),
                Output('critic-loss-graph', 'figure'),
                Output('success-rate-graph', 'figure'),
                Output('collision-rate-graph', 'figure'),
                Output('battery-graph', 'figure'),
            ],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # Load latest metrics
            self._load_metrics()

            # Status
            status = 'Running...' if len(self.metrics_data['steps']) > 0 else 'Waiting...'
            total_steps = self.metrics_data['steps'][-1] if self.metrics_data['steps'] else 0
            mean_reward = np.mean(self.metrics_data['episode_rewards'][-100:]) if len(self.metrics_data['episode_rewards']) > 0 else 0.0

            # Create figures
            reward_fig = self._create_reward_figure()
            actor_loss_fig = self._create_actor_loss_figure()
            critic_loss_fig = self._create_critic_loss_figure()
            success_fig = self._create_success_rate_figure()
            collision_fig = self._create_collision_rate_figure()
            battery_fig = self._create_battery_figure()

            return (
                status,
                f'{total_steps:,}',
                f'{mean_reward:.2f}',
                reward_fig,
                actor_loss_fig,
                critic_loss_fig,
                success_fig,
                collision_fig,
                battery_fig
            )

    def _load_metrics(self):
        """Load metrics from log files."""
        metrics_file = os.path.join(self.log_dir, 'training_metrics.json')

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)

                # Update metrics data
                if 'episode_rewards' in data:
                    self.metrics_data['episode_rewards'] = [
                        ep['reward'] for ep in data['episode_rewards']
                    ]
                    self.metrics_data['steps'] = [
                        ep.get('step', i) for i, ep in enumerate(data['episode_rewards'])
                    ]

                if 'losses' in data:
                    self.metrics_data['actor_loss'] = data['losses'].get('actor', [])
                    self.metrics_data['critic_loss'] = data['losses'].get('critic', [])

            except Exception as e:
                logger.error(f"Error loading metrics: {e}")

    def _create_reward_figure(self):
        """Create episode reward figure."""
        steps = self.metrics_data['steps']
        rewards = self.metrics_data['episode_rewards']

        # Compute running average
        window = 100
        if len(rewards) >= window:
            running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            running_steps = steps[window-1:]
        else:
            running_avg = rewards
            running_steps = steps

        fig = go.Figure()

        # Raw rewards
        fig.add_trace(go.Scatter(
            x=steps,
            y=rewards,
            mode='lines',
            name='Episode Reward',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))

        # Running average
        fig.add_trace(go.Scatter(
            x=running_steps,
            y=running_avg,
            mode='lines',
            name='Running Average (100 eps)',
            line=dict(color='blue', width=3)
        ))

        fig.update_layout(
            title='Episode Rewards Over Time',
            xaxis_title='Training Steps',
            yaxis_title='Reward',
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def _create_actor_loss_figure(self):
        """Create actor loss figure."""
        losses = self.metrics_data['actor_loss']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=losses,
            mode='lines',
            name='Actor Loss',
            line=dict(color='green')
        ))

        fig.update_layout(
            title='Actor Loss',
            xaxis_title='Update',
            yaxis_title='Loss',
            template='plotly_white'
        )

        return fig

    def _create_critic_loss_figure(self):
        """Create critic loss figure."""
        losses = self.metrics_data['critic_loss']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=losses,
            mode='lines',
            name='Critic Loss',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='Critic Loss',
            xaxis_title='Update',
            yaxis_title='Loss',
            template='plotly_white'
        )

        return fig

    def _create_success_rate_figure(self):
        """Create success rate figure."""
        success_rates = self.metrics_data['success_rate']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=success_rates,
            mode='lines',
            name='Success Rate',
            line=dict(color='green'),
            fill='tozeroy'
        ))

        fig.update_layout(
            title='Task Success Rate',
            xaxis_title='Episode',
            yaxis_title='Success Rate',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )

        return fig

    def _create_collision_rate_figure(self):
        """Create collision rate figure."""
        collision_rates = self.metrics_data['collision_rate']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=collision_rates,
            mode='lines',
            name='Collision Rate',
            line=dict(color='red'),
            fill='tozeroy'
        ))

        fig.update_layout(
            title='Collision Rate',
            xaxis_title='Episode',
            yaxis_title='Collision Rate',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )

        return fig

    def _create_battery_figure(self):
        """Create battery efficiency figure."""
        battery_eff = self.metrics_data['battery_efficiency']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=battery_eff,
            mode='lines',
            name='Battery Efficiency',
            line=dict(color='orange')
        ))

        fig.update_layout(
            title='Fleet Battery Efficiency',
            xaxis_title='Episode',
            yaxis_title='Efficiency',
            template='plotly_white'
        )

        return fig

    def run(self, debug: bool = False):
        """
        Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting dashboard on http://localhost:{self.port}")
        self.app.run_server(debug=debug, port=self.port)


def create_dashboard(log_dir: str = 'results/logs', port: int = 8050):
    """
    Create and return dashboard instance.

    Args:
        log_dir: Log directory
        port: Server port

    Returns:
        Dashboard instance
    """
    dashboard = TrainingDashboard(log_dir=log_dir, port=port)
    return dashboard
