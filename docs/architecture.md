# COMAR Architecture

## System Overview

COMAR (Counterfactual Multi-Agent Robot Coordination) is a comprehensive multi-agent reinforcement learning system for warehouse robot coordination. The architecture is modular and extensible.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Train Script    │  │  Curriculum      │  │  Callbacks   │  │
│  │  (train.py)      │  │  Learning        │  │  Management  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
│           │                      │                    │          │
│           └──────────────────────┼────────────────────┘          │
│                                  ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           COMATrainer (Main Loop)                       │   │
│  │  - Collect rollouts                                     │   │
│  │  - Update networks                                      │   │
│  │  - Checkpoint management                               │   │
│  └────────────┬────────────────────────────┬───────────────┘   │
│               │                            │                    │
└───────────────┼────────────────────────────┼────────────────────┘
                │                            │
┌───────────────┼────────────────────────────┼────────────────────┐
│               ↓                            ↓                    │
│  ┌────────────────────────┐  ┌──────────────────────────┐      │
│  │  ENVIRONMENT           │  │  ALGORITHM (COMA)        │      │
│  ├────────────────────────┤  ├──────────────────────────┤      │
│  │ - WarehouseEnv         │  │ - COMAcontinuous         │      │
│  │ - Robot dynamics       │  │ - Actor networks (N)     │      │
│  │ - Task generation      │  │ - Critic network         │      │
│  │ - Collision detection  │  │ - Replay buffer          │      │
│  │ - Physics simulation   │  │ - Experience replay      │      │
│  │ - LiDAR sensors        │  │ - Counterfactual logic   │      │
│  └────────────────────────┘  └──────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              COORDINATION LAYER                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │Task Allocator│  │Path Planner  │  │Communication│           │
│  │              │  │              │  │ Network     │           │
│  │- Greedy      │  │- A* Search   │  │- Broadcasting           │
│  │- Load-bal.   │  │- Smoothing   │  │- Direct msg │           │
│  │- Priority    │  │- Trajectories│  │- Connectivity           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │Battery Mgr   │  │Scheduler     │                             │
│  │              │  │              │                             │
│  │- Levels      │  │- FIFO/Priority                             │
│  │- Routing     │  │- EDF/LLF     │                             │
│  │- Charging    │  │- Deadlines   │                             │
│  └──────────────┘  └──────────────┘                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│           VISUALIZATION & MONITORING                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │ Dashboard       │  │ Renderer     │  │ Metrics Compute│      │
│  │ (Plotly Dash)   │  │ (PyBullet)   │  │ (20+ metrics)  │      │
│  │ - Real-time     │  │ - 3D vis     │  │ - Performance  │      │
│  │ - Web UI        │  │ - Overlays   │  │ - Efficiency   │      │
│  │ - Graphs        │  │ - Camera     │  │ - Coordination │      │
│  └─────────────────┘  └──────────────┘  └────────────────┘      │
│                                                                   │
│  ┌──────────────────────────────────────┐                       │
│  │ Plot Generator (Matplotlib/Seaborn)  │                       │
│  │ - Training curves                    │                       │
│  │ - Comparisons                        │                       │
│  │ - Heatmaps, box plots                │                       │
│  │ - Publication quality (300 DPI)      │                       │
│  └──────────────────────────────────────┘                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
scripts/ (Entry points)
  ├── train.py → trainer.py → COMATrainer → COMAcontinuous
  ├── evaluate.py → PolicyEvaluator → COMAcontinuous
  ├── visualize.py → PyBulletRenderer → WarehouseEnv
  └── benchmark.py → ComparisonMetrics → multiple configs

src/algorithms/
  ├── coma_continuous.py (main algorithm)
  ├── actor_network.py (policy networks)
  ├── critic_network.py (value networks)
  ├── replay_buffer.py (experience storage)
  └── utils.py (helpers)

src/environments/
  ├── warehouse_env.py (main env)
  ├── robot.py (robot dynamics)
  ├── task_generator.py (task creation)
  ├── collision_checker.py (collision detection)
  ├── warehouse_layout.py (map generation)
  └── sensors.py (LiDAR simulation)

src/coordination/
  ├── task_allocator.py (task assignment)
  ├── path_planner.py (motion planning)
  ├── communication.py (message passing)
  ├── battery_manager.py (energy management)
  └── scheduler.py (task scheduling)

src/training/
  ├── trainer.py (main training loop)
  ├── evaluator.py (performance evaluation)
  ├── curriculum.py (progressive learning)
  └── callbacks.py (event hooks)

src/visualization/
  ├── dashboard.py (web dashboard)
  ├── renderer.py (3D visualization)
  ├── metrics.py (metric computation)
  └── plots.py (publication plots)

tests/
  ├── test_environment.py (env tests)
  ├── test_coma.py (algorithm tests)
  └── test_coordination.py (coordination tests)
```

## Data Flow

### Training Loop

```
1. Reset Environment
   → robots at random positions
   → tasks generated (Poisson)

2. Collect Rollout (2000 steps)
   → obs from environment
   → for each step:
      a) Get actions from actors
      b) Execute in environment
      c) Get reward + next_obs
      d) Store in replay buffer

3. Update Networks (10 epochs × 4 updates)
   a) Sample batch from buffer (64 samples)
   b) Compute critic loss
      L_critic = (r + γQ'(s',a') - Q(s,a))²
   c) Update critic with gradient
   d) For each agent i:
      - Compute counterfactual advantage
      - Compute actor loss
      - Update actor with gradient
   e) Soft-update target networks

4. Evaluate (every 5000 steps)
   → 10 deterministic episodes
   → compute metrics (reward, success rate, etc)
   → save if best

5. Checkpoint (every 10000 steps)
   → save model state
   → save training state
```

### Inference Flow

```
1. Load checkpoint
2. For each episode:
   a) Reset environment
   b) For each step:
      - Get observations from robots
      - Forward through actors (deterministic mode)
      - Get actions for all robots
      - Execute actions in environment
      - Render if enabled
3. Compute metrics
4. Save results
```

## Key Components

### 1. Actor Network
- Per-agent policy network
- Input: local observation (40-50 dims)
- Output: mean and std of Gaussian policy
- Architecture: 256-256 ReLU

### 2. Critic Network
- Centralized value function
- Input: global state + joint actions
- Output: Q-value per agent
- Architecture: 256-256 ReLU

### 3. Replay Buffer
- Circular buffer: 100,000 transitions
- Stores: states, obs, actions, rewards, next states
- Sampling: uniform random batches of 64

### 4. Environment
- PyBullet physics engine
- Differential drive robots
- Poisson task generation
- Collision detection via spatial hashing

### 5. Coordination
- Task allocation (greedy, priority-aware)
- Path planning (A* search)
- Communication (range-based)
- Battery management
- Deadline-aware scheduling

## Scaling Considerations

### Small Scale (2-5 robots)
- Single machine sufficient
- CPU training: ~1 hour
- GPU training: ~10 minutes

### Medium Scale (5-10 robots)
- GPU recommended
- 4x CPU training speedup
- Curriculum learning helps

### Large Scale (20+ robots)
- Multi-GPU setup beneficial
- Distributed environment sampling
- Need careful reward tuning

## Extension Points

### 1. Custom Algorithm
- Inherit from `COMAcontinuous`
- Override `update()` method
- New credit assignment logic

### 2. Custom Environment
- Inherit from `WarehouseEnv`
- Add new robots/obstacles
- Custom task types

### 3. Custom Coordination
- Implement new allocator strategies
- Add communication protocols
- Custom charging policies

### 4. Custom Rewards
- Edit `reward_structure` in config
- Add new reward components
- Task-specific shaping

## Performance Metrics

### Computed During Training
- Episode reward
- Task success rate
- Collision rate
- Actor/critic loss
- Fleet utilization

### Evaluation Metrics
- Mean reward ± std
- Success rate
- Battery efficiency
- Average completion time
- Coordination quality

## Configuration Hierarchy

```
1. defaults (hardcoded)
2. config file (configs/*.yaml)
3. command line arguments (--seed 42)

Later values override earlier ones
```

## Testing Strategy

```
Unit Tests (45+ test cases)
├── Environment
│   ├── Initialization
│   ├── Physics
│   ├── Collisions
│   └── Task generation
├── Algorithm
│   ├── Actor/Critic
│   ├── Replay buffer
│   ├── Updates
│   └── Checkpoints
└── Coordination
    ├── Task allocation
    ├── Path planning
    ├── Communication
    └── Battery management

Integration Tests (via scripts)
├── Training → Evaluation
├── Curriculum learning
└── Multi-agent coordination
```

## File Structure

```
warehouse-coma-marl/
├── src/                    (Main source code)
├── scripts/                (Entry points)
├── configs/                (Configuration files)
├── tests/                  (Unit tests)
├── docs/                   (Documentation)
├── results/                (Training outputs - created at runtime)
│   ├── checkpoints/        (Model files)
│   ├── logs/               (TensorBoard logs)
│   ├── videos/             (Recordings)
│   └── metrics/            (Performance data)
└── notebooks/              (Jupyter notebooks)
```

This modular architecture enables:
- Easy testing and debugging
- Independent component development
- Clear separation of concerns
- Simple extension and customization
