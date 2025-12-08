# Multi-Agent Warehouse Robot Coordination with Continuous COMA

## Project Overview

A cutting-edge **multi-agent reinforcement learning system** that extends the Counterfactual Multi-Agent (COMA) algorithm to continuous action spaces for coordinating autonomous mobile robots (AMRs) in a dynamic warehouse environment.

**Key Innovation**: Combines **centralized training with decentralized execution (CTDE)**, where robots learn coordinated behaviors using full global state during training, but execute policies independently using only local observations. This makes it practical for real-world warehouse deployment.

---

## Features

### Multi-Domain Integration
- Multi-agent policy gradients (COMA with counterfactual reasoning)
- Robotics kinematics (differential drive control)
- Task scheduling and optimization
- Distributed coordination
- Collision avoidance and motion planning
- Battery management

### Continuous Action Space
Extended the original discrete COMA algorithm to continuous control using Gaussian policies for smooth robot velocity commands (linear + angular velocity).

### Real-World Applicability
Warehouse robot coordination is a billion-dollar industry problem. This system handles:
- Dynamic task arrival with priorities
- Battery constraints and charging coordination
- Robot-robot and robot-obstacle collision avoidance
- Real-time task reallocation based on fleet state

### Scalability
Designed to scale from 3 robots to 20+ robots, demonstrating true multi-agent challenges as complexity grows non-linearly.



## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, optional but recommended)
- 16GB+ RAM

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/warehouse-coma-marl.git
cd warehouse-coma-marl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

##  Observation & Action Spaces

### Observation Space (per robot)
- **LiDAR readings**: 360° coverage, 16 beams with distance measurements
- **Self state**: Position (x, y), orientation (θ), velocity (v, ω), battery level
- **Task information**: Current task location, priority, deadline
- **Communication**: Nearest robot positions and movement intentions

### Action Space (continuous, per robot)
- **Linear velocity**: [-1.0, 1.0] m/s (forward/backward motion)
- **Angular velocity**: [-π/2, π/2] rad/s (rotation)
- Sampled from Gaussian policy: π(a|o) = N(μ(o), σ²(o))

---

## Reward Structure

```
Task completion bonus:           +10 per successful delivery
Time penalty:                    -0.01 per timestep
Robot-robot collision:           -5
Static obstacle collision:       -2
Battery depletion:               -10
Charging station reached:        +1 (low battery)
Idle robot penalty:              -0.05 per timestep
```

The reward is **shared among all agents** to encourage cooperative behavior and team-level optimization rather than selfish individual rewards.

---

## Quick Start

### Training from Scratch

```bash
# Default configuration (5 robots, small warehouse)
python scripts/train.py --config configs/default_config.yaml

# Custom configuration
python scripts/train.py --config configs/large_warehouse.yaml --num-robots 10

# With GPU
python scripts/train.py --config configs/default_config.yaml --device cuda
```

### Evaluating Trained Model

```bash
# Evaluate on test scenarios
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt

# Visualize policy in action
python scripts/visualize.py --checkpoint results/checkpoints/best_model.pt --render
```

### Running Benchmarks

```bash
# Compare different configurations
python scripts/benchmark.py --configs configs/small_warehouse.yaml configs/large_warehouse.yaml
```

---

##  Training Configuration

Key hyperparameters in `config.yaml`:

```yaml
# Algorithm
algorithm:
  name: "COMA"
  learning_rate: 3e-4
  critic_learning_rate: 1e-3
  discount_factor: 0.99
  gae_lambda: 0.95
  entropy_coefficient: 0.001

# Environment
environment:
  num_robots: 5
  warehouse_size: [100, 100]  # meters
  max_episode_steps: 2000
  task_arrival_rate: 0.5

# Training
training:
  total_timesteps: 500000
  batch_size: 32
  buffer_size: 10000
  rollout_steps: 2000
```

See `config.yaml` for full configuration details.

---

## Expected Performance

After training ~500K episodes with 5 robots:

| Metric | Expected Value |
|--------|-----------------|
| Task Success Rate | 85-95% |
| Collision Rate | <5% |
| Avg Delivery Time | 30-45 seconds |
| Fleet Utilization | 70-85% |
| Energy Efficiency | 90%+ |

### Learning Curve Progression

1. **Exploration Phase** (Episodes 0-100K): Random exploration, basic obstacle avoidance
2. **Learning Phase** (Episodes 100K-300K): Task completion improves, coordination emerges
3. **Refinement Phase** (Episodes 300K-500K): Efficient multi-robot coordination, collision minimization
4. **Fine-tuning Phase** (Episodes 500K+): Handling edge cases and rare scenarios

---

## Technical Details

### Centralized Training, Decentralized Execution (CTDE)

**Training Phase**:
- Access to global state (all robot positions, all task locations)
- Centralized critic Q(s, u₁, u₂, ..., uₙ) sees everything
- Computes advantages for credit assignment

**Execution Phase**:
- Each robot only has local observations (LiDAR, nearby robots)
- Policy π^i(aᵢ|oᵢ) takes only agent i's observation
- No communication needed during inference

### Counterfactual Advantage Estimation

For each agent i, we compute:

```
A^i(s, u) = Q(s, u) - Σ_{u'_i} π^i(u'_i|o^i) Q(s, (u^{-i}, u'_i))
```

This isolates the contribution of agent i's action to the overall team reward, properly attributing credit/blame in multi-agent settings.

### Continuous Action Gaussian Policy

```
μ(o) = tanh(fc_layers(o)) * action_scale
log_σ(o) = fc_layers(o)
π(a|o) = N(μ(o), exp(log_σ(o))²)
```

Squashed with tanh to bound actions to valid ranges.

---

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_coma.py -v

# Test environment rendering
python -m pytest tests/test_environment.py --render
```

---

##  Algorithm Details

### COMA (Counterfactual Multi-Agent Policy Gradients)

The policy gradient update for agent i:

```
∇π^i L = E[∇_θ log π^i(u^i|o^i) * A^i(s, u)]
```

Where A^i uses the counterfactual baseline to isolate agent i's contribution to the joint reward.

**Key Advantages**:
- Addresses credit assignment problem in multi-agent settings
- Counterfactual reasoning: "What if I hadn't taken that action?"
- Reduced variance compared to independent learners
- Encourages cooperative behavior

### Extensions to Continuous Actions

Original COMA assumes discrete action spaces. Our extension:
- Uses Gaussian policies instead of categorical
- Computes advantages in continuous action space
- Applies reparameterization trick for low-variance gradients
- Entropy regularization to encourage exploration

---

##  Curriculum Learning Stages

The system trains progressively through increasing difficulty:

```
Stage 1: 2-3 robots, small warehouse, few tasks
         Focus: Basic navigation and collision avoidance

Stage 2: 5 robots, medium warehouse, moderate tasks
         Focus: Task coordination and scheduling

Stage 3: 10+ robots, large warehouse, high task load
         Focus: Emergent coordination under congestion
```

Configure in `configs/curriculum_stages.yaml`.

---

## Monitoring & Visualization

### TensorBoard

```bash
tensorboard --logdir results/logs/
```

Tracks:
- Episode rewards and success rates
- Collision rates and safety metrics
- Task completion time
- Policy entropy and losses
- Network weight distributions

### Real-Time Dashboard

```bash
python -m src.visualization.dashboard --logdir results/logs/
```

Displays:
- Live robot positions and movements
- Task queue and assignments
- Fleet battery levels
- Performance metrics



## Advanced Usage

### Custom Warehouse Layout

```python
from src.environments.warehouse_layout import WarehouseLayout

layout = WarehouseLayout(
    width=200,
    height=150,
    num_shelves=12,
    num_charging_stations=3
)
```

### Monitoring During Training

```python
from src.visualization.metrics import MetricsTracker

tracker = MetricsTracker()
tracker.log_episode_metrics(
    episode=100,
    success_rate=0.92,
    collision_rate=0.03,
    avg_delivery_time=35.2
)
```

### Loading and Continuing Training

```python
# Load checkpoint and resume
trainer = Trainer(config)
trainer.load_checkpoint('results/checkpoints/episode_250k.pt')
trainer.train(additional_steps=250000)
```

---

