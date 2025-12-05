# Environment Specification

## Warehouse Environment Overview

The COMAR warehouse environment is a multi-robot coordination simulation built on PyBullet physics engine. It models a realistic warehouse with mobile robots, tasks (pickup/delivery), and various constraints.

## State Space

### Global State
- **Dimensions**: [num_robots * 9 + num_tasks * 6 + warehouse_info]
- **Components per robot**: position (3D), velocity (2D), battery level (1), task indicator (1), status (1)
- **Components per task**: pickup location (2D), delivery location (2D), priority, deadline

### Local Observation
- **Dimensions per agent**: ~40-50 (configurable)
- **Components**:
  - LiDAR readings (16 beams, max_range=20m)
  - Relative position to target task
  - Battery level
  - Current task info (if assigned)
  - Relative positions of nearby robots
  - Velocity

## Action Space

### Continuous Actions
- **Type**: Continuous, bounded [-1, 1]
- **Dimensions**: 2 per agent
  - Linear velocity: [-max_velocity, +max_velocity]
  - Angular velocity: [-max_angular_velocity, +max_angular_velocity]
- **Actualization**: Differential drive kinematics

### Action Dynamics
```
v_linear = action[0] * max_linear_velocity
v_angular = action[1] * max_angular_velocity
x_next = x + v_linear * cos(theta) * dt
y_next = y + v_linear * sin(theta) * dt
theta_next = theta + v_angular * dt
```

## Task Specification

### Task Format
Tuple: (pickup_x, pickup_y, priority, delivery_x, delivery_y, deadline)

### Task Properties
- **Priority**: 1-10 scale (higher = more urgent)
- **Deadline**: Maximum steps to complete
- **Reward**: 10 points + bonus for early completion
- **Penalty**: -0.01 per step if unassigned

## Reward Structure

### Task Completion
- **Reward**: +10.0 base reward
- **Early Bonus**: +5.0 if completed before deadline
- **Per-step Penalty**: -0.01 for time cost

### Safety
- **Collision Penalty**: -5.0 per collision
- **Battery Critical**: -20.0 when battery < 10%

### Efficiency
- **Distance Efficiency**: +0.1 per meter traveled toward goal
- **Coordination Bonus**: +1.0 for good team coordination

## Environment Constraints

### Robot Constraints
- **Battery Capacity**: 100.0 units
- **Discharge Rate**: 0.05 * velocity per step
- **Charge Rate**: 0.3 units per step at charging station
- **Max Velocity**: 2.0 m/s linear, 3.14 rad/s angular
- **Radius**: 0.5 m (collision detection)

### Warehouse Constraints
- **Width**: 100m (configurable)
- **Height**: 80m (configurable)
- **Shelves**: 8 total (2.5m width, 1.5m height each)
- **Charging Stations**: 2 locations
- **Delivery Zones**: 3 locations

### Physics
- **Gravity**: 9.81 m/s²
- **Time Step**: 0.01 seconds
- **Friction Coefficient**: 0.5
- **Physics Substeps**: 5

## Observation Normalization

All observations are normalized to [-1, 1] range:

```
obs_normalized = (obs - obs_min) / (obs_max - obs_min) * 2 - 1
```

### Typical Ranges
- Position: [0, 100] → [-1, 1]
- Velocity: [-2, 2] → [-1, 1]
- Battery: [0, 100] → [-1, 1]
- LiDAR: [0, 20m] → [-1, 1]

## Episode Termination

An episode terminates when:
1. **Max Steps Reached**: 2000 steps by default
2. **Done Flag**: Can be set for specific scenarios

## Reset Behavior

- **Robot Positions**: Random within free space
- **Robot Orientation**: Random [0, 2π]
- **Battery Level**: Full (100%)
- **Pending Tasks**: 0 (new tasks generated according to arrival rate)

## Task Generation

### Poisson Process
Tasks arrive according to a Poisson process with rate λ:

```
P(k tasks in interval) = (λ^k * e^(-λ)) / k!
```

### Default Parameters
- **Arrival Rate**: 0.3 tasks/step
- **Priority Distribution**: Uniform [1, 10]
- **Deadline Range**: [100, 500] steps
- **Max Pending Tasks**: 50

## Rendering

### PyBullet Visualization
- **Camera View**: Top-down by default
- **Overlays**: Robot battery indicators, task locations, paths
- **Frame Rate**: 60 FPS (configurable)

### Rendering Elements
- Green spheres: Task pickup locations
- Red spheres: Task delivery locations
- Blue cylinders: Robots
- Yellow bars above robots: Battery level
- Lines connecting: Planned paths

## Configuration

See `configs/default_config.yaml` for all environment parameters.

---

## Environment Scenarios

### Small Warehouse (Training)

Simplified scenario for initial training and debugging:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Warehouse Size** | 50m × 40m | Smaller for faster learning |
| **Num Robots** | 2-3 | Reduced complexity |
| **Task Arrival Rate** | 0.1-0.2 | Sparse tasks |
| **Episode Length** | 500 steps | Shorter episodes |
| **Obstacles** | 5 shelves | Minimal obstacles |

**Use Case**: Initial training, hyperparameter tuning, debugging

---

### Medium Warehouse (Standard)

Balanced scenario for general training:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Warehouse Size** | 100m × 80m | Standard warehouse |
| **Num Robots** | 5 | Moderate coordination |
| **Task Arrival Rate** | 0.5 | Regular task flow |
| **Episode Length** | 1000 steps | Standard duration |
| **Obstacles** | 10-15 shelves | Realistic navigation |

**Use Case**: Main training, evaluation, benchmarking

---

### Large Warehouse (Challenge)

Complex scenario for testing scalability:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Warehouse Size** | 150m × 120m | Large-scale |
| **Num Robots** | 10+ | High coordination demand |
| **Task Arrival Rate** | 1.0+ | High task density |
| **Episode Length** | 2000 steps | Extended operation |
| **Obstacles** | 20+ shelves | Complex navigation |

**Use Case**: Stress testing, scalability evaluation, advanced research

---

## Common Use Cases

### 1. Battery Management Testing

```yaml
environment:
  battery_capacity: 50.0  # Reduced capacity
  battery_discharge_rate: 0.2  # Faster drain
  num_charging_stations: 3  # Multiple stations
```

**Focus**: Tests charging strategy and battery-aware coordination

### 2. High-Priority Tasks

```yaml
task_generator:
  priority_range: [7, 10]  # Only high-priority
  deadline_range: [50, 100]  # Tight deadlines
```

**Focus**: Tests urgency handling and task prioritization

### 3. Dense Robot Coordination

```yaml
environment:
  num_robots: 10
  warehouse_size: {width: 80, height: 60}  # Smaller space
  collision_penalty: -10.0  # Higher penalty
```

**Focus**: Tests collision avoidance and dense coordination

### 4. Sparse Long-Range Tasks

```yaml
environment:
  warehouse_size: {width: 150, height: 120}
  task_arrival_rate: 0.1
  task_spread: large  # Tasks far apart
```

**Focus**: Tests path planning and long-distance coordination

---

## Observation Space Details

### Complete Observation Breakdown (35D)

| Component | Dimensions | Range | Description |
|-----------|------------|-------|-------------|
| **Position** | 2 | [-1, 1] | Normalized x, y position |
| **Velocity** | 2 | [-1, 1] | Normalized linear and angular velocity |
| **Battery** | 1 | [-1, 1] | Normalized battery level |
| **Current Task** | 6 | [-1, 1] | Pickup (2), priority (1), delivery (2), deadline (1) |
| **Nearest Task** | 2 | [-1, 1] | Relative position to nearest pending task |
| **Nearby Robots** | 6 | [-1, 1] | Relative positions of 3 nearest robots (x, y each) |
| **LiDAR** | 16 | [-1, 1] | Distance readings in 16 directions |

**Total**: 2 + 2 + 1 + 6 + 2 + 6 + 16 = **35 dimensions**

### LiDAR Configuration

```
Beam angles: [0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°,
              180°, 202.5°, 225°, 247.5°, 270°, 292.5°, 315°, 337.5°]
Max range: 20m
Normalization: distance / max_range → [0, 1] → [-1, 1]
```

---

## Tips for Environment Customization

### Adding New Obstacles

```python
# In warehouse_layout.py
def add_custom_obstacle(self, position, size, shape='box'):
    if shape == 'box':
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2]
        )
    # Create visual and collision shapes
    obstacle_id = p.createMultiBody(baseMass=0, ...)
    self.obstacles.append(obstacle_id)
```

### Modifying Reward Function

```python
# In warehouse_env.py, modify _compute_reward()
def _compute_reward(self, ...):
    reward = 0.0
    reward += self.config['rewards']['task_completion'] * completed_tasks
    reward += self.config['rewards']['collision'] * collisions
    reward += self.config['rewards']['time_step_penalty']
    # Add custom rewards here
    return reward
```

### Custom Task Generation

```python
# In task_generator.py
def generate_priority_clusters(self):
    # Generate tasks in spatial clusters
    cluster_centers = [(x1, y1), (x2, y2), ...]
    for center in cluster_centers:
        self.add_task_near(center, radius=10.0)
```
