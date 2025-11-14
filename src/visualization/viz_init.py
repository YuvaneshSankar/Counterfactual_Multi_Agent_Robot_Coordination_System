"""
Visualization Module - Monitoring and Analysis Tools

Provides real-time monitoring, rendering, and analysis visualization
for the COMAR training system.
"""

from .dashboard import TrainingDashboard, create_dashboard
from .renderer import PyBulletRenderer
from .metrics import PerformanceMetrics, ComparisonMetrics
from .plots import PlotGenerator

__all__ = [
    'TrainingDashboard',
    'create_dashboard',
    'PyBulletRenderer',
    'PerformanceMetrics',
    'ComparisonMetrics',
    'PlotGenerator',
]