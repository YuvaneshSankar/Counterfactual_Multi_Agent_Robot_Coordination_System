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
