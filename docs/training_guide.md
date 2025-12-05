# COMAR Training Guide

## Quick Start

### Installation

```bash
git clone <repository>
cd warehouse-coma-marl
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### First Training Run

```bash
# Train on small warehouse for quick testing
python scripts/train.py --config configs/small_warehouse.yaml --device cpu --seed 42

# Monitor training
python scripts/visualize.py --checkpoint results/checkpoints/best_model.pt
```

## Training Configurations

### Small Warehouse (Testing)
- **Robots**: 2
- **Warehouse**: 50×40m
- **Training Time**: ~1 hour (CPU), ~15 min (GPU)
- **Best for**: Quick iterations, debugging

```bash
python scripts/train.py --config configs/small_warehouse.yaml
```

### Default Configuration (Recommended)
- **Robots**: 5
- **Warehouse**: 100×80m
- **Training Time**: ~4 hours (CPU), ~1 hour (GPU)
- **Best for**: Balanced performance/compute

```bash
python scripts/train.py --config configs/default_config.yaml --device cuda
```

### Large Warehouse (Scalability)
- **Robots**: 20
- **Warehouse**: 200×150m
- **Training Time**: ~12 hours (CPU), ~2 hours (GPU)
- **Best for**: Testing scalability, stress testing

```bash
python scripts/train.py --config configs/large_warehouse.yaml --device cuda
```

## Advanced Training

### Curriculum Learning

The curriculum automatically progresses through stages:
1. **Stage 1**: Single robot, simple environment
2. **Stage 2**: 2 robots, sparse tasks
3. **Stage 3**: 5 robots, moderate tasks
4. **Stage 4**: 10 robots, high density
5. **Stage 5**: 20 robots, extreme density

Enable curriculum:
```bash
python scripts/train.py --config configs/curriculum_stages.yaml
```

### Distributed Training

Train with multiple environments:
```bash
python scripts/train.py --config configs/default_config.yaml \
    --num-envs 4  # Uses 4 parallel environments
```

### Resume Training

Continue from checkpoint:
```bash
python scripts/train.py \
    --config configs/default_config.yaml \
    --checkpoint results/checkpoints/model_500000.pt
```

### Custom Configuration

Edit or create new config files:
```bash
cp configs/default_config.yaml configs/my_config.yaml
# Edit my_config.yaml
python scripts/train.py --config configs/my_config.yaml
```

## Monitoring Training

### Web Dashboard

Start dashboard automatically during training:
```bash
python scripts/train.py --config configs/default_config.yaml --dashboard
# Open http://localhost:8050 in browser
```

Dashboard displays:
- Episode rewards
- Actor/Critic losses
- Success rate
- Collision rate
- Battery efficiency

### TensorBoard Logs

```bash
tensorboard --logdir=results/logs
# Open http://localhost:6006
```

### Console Output

Training logs are printed to console and saved to `training.log`:
```
2025-11-14 10:30:45 - INFO - Step 1000: reward=5.2, actor_loss=0.45, critic_loss=0.23
```

## Hyperparameter Tuning

### Key Parameters

| Parameter | Impact | Tuning Tips |
|-----------|--------|------------|
| `actor_lr` | Learning speed | Reduce if unstable, increase if slow |
| `batch_size` | Stability vs speed | Larger = more stable but slower |
| `gamma` | Future importance | Higher (0.99+) for long horizons |
| `buffer_size` | Memory usage | Larger = better correlation |

### Tuning Workflow

1. **Start with defaults** - Use provided configs
2. **Monitor metrics**:
   - Actor/Critic loss should decrease
   - Reward should increase over time
   - Collision rate should decrease
3. **Adjust if needed**:
   - Unstable? → Reduce actor_lr, increase tau
   - Too slow? → Increase batch_size, reduce epochs
   - Poor coordination? → Increase coordination_bonus

## Evaluation

### Quick Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --num-episodes 10 \
    --deterministic
```

### Performance Benchmarking

Compare different models:
```bash
python scripts/benchmark.py \
    --configs config1.yaml config2.yaml config3.yaml \
    --checkpoints model1.pt model2.pt model3.pt \
    --names "Baseline" "COMA" "COMA+" \
    --num-episodes 50
```

### Detailed Analysis

```bash
# Evaluate with rendering
python scripts/evaluate.py \
    --checkpoint best_model.pt \
    --num-episodes 5 \
    --render

# Visualize behavior
python scripts/visualize.py \
    --checkpoint best_model.pt \
    --num-episodes 3 \
    --show-paths \
    --show-communication \
    --follow-robot 0
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_environment.py::TestRobot::test_battery_discharge -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Training is too slow

**Symptoms**: Very slow episode generation

**Solutions**:
- Use GPU: `--device cuda`
- Increase num_workers: `num_workers: 8` in config
- Reduce rollout_steps: `rollout_steps: 1000`
- Use smaller network: `hidden_dims: [128, 128]`

### Training is unstable

**Symptoms**: Rewards fluctuating wildly, NaN losses

**Solutions**:
- Reduce actor learning rate: `actor_lr: 1e-4`
- Increase tau (softer updates): `tau: 0.01`
- Enable gradient clipping: `max_grad_norm: 0.5`
- Increase batch size: `batch_size: 128`

### Poor coordination

**Symptoms**: Robots colliding frequently, low success rate

**Solutions**:
- Increase coordination bonus: `coordination_bonus: 5.0`
- Reduce collision penalty: `collision_penalty: -3.0`
- Enable communication: `communication.enabled: true`
- Use priority-aware allocation: `task_allocator.strategy: priority_aware`

### Out of memory

**Symptoms**: CUDA out of memory error

**Solutions**:
- Reduce buffer_size: `buffer_size: 50000`
- Reduce batch_size: `batch_size: 32`
- Reduce num_envs: `num_envs: 1`
- Use CPU: `--device cpu`

## Best Practices

### 1. Start Small
- Use small_warehouse config for initial testing
- Verify code works before running large experiments

### 2. Progressive Scaling
- Train on 2 robots first
- Gradually increase to 5, then 10, then 20
- Use curriculum learning

### 3. Regular Checkpoints
- Save checkpoints frequently (every 10k steps)
- Keep best models separately
- Version your experiments

### 4. Monitor Everything
- Watch loss curves for anomalies
- Track success rate and collision rate
- Check battery efficiency

### 5. Reproducibility
- Always set seed: `--seed 42`
- Document hyperparameters
- Save configs with results

## Advanced Topics

### Custom Reward Shaping

Edit `configs/default_config.yaml`:
```yaml
rewards:
  task_completion: 15.0  # Increase to prioritize tasks
  coordination_bonus: 5.0  # Increase for better cooperation
  collision_penalty: -20.0  # Increase to reduce collisions
```

### Custom Task Generation

Modify task arrival rate:
```yaml
environment:
  task_arrival_rate: 0.5  # 0.5 tasks per step
  task_priority_range: [1, 10]  # Priority from 1-10
  task_deadline_range: [100, 500]  # Deadline range
```

### Custom Warehouse Layouts

Edit warehouse dimensions:
```yaml
environment:
  warehouse_size:
    width: 150.0  # Custom width
    height: 120.0  # Custom height
  obstacles:
    num_shelves: 20  # More obstacles
```

## Performance Targets

These are typical performance metrics to aim for:

| Metric | Small | Default | Large |
|--------|-------|---------|-------|
| Success Rate | 85% | 75% | 60% |
| Collision Rate | <5% | <8% | <10% |
| Avg Reward | 80+ | 50+ | 30+ |
| Training Time (GPU) | 15 min | 1 hour | 2 hours |

## Common Questions

**Q: How long should I train?**
A: Start with 100k steps to verify setup, then 1M+ steps for good performance.

**Q: Should I use CPU or GPU?**
A: GPU is 5-10x faster. Use CPU only for testing small configs.

**Q: Why is my model not improving?**
A: Check learning rates, batch size, and reward shaping. Verify data is flowing correctly.

**Q: Can I train multiple seeds in parallel?**
A: Yes, use different seed values and output directories for each.

For more help, see documentation in `docs/` and inline code comments.

---

## Troubleshooting Guide

### Problem: Training Loss Explodes (NaN values)

**Symptoms:**
- Actor/Critic loss becomes NaN
- Rewards suddenly drop to zero
- Training crashes

**Solutions:**
```yaml
# Reduce learning rates
training:
  actor_lr: 1e-4  # Lower from 3e-4
  critic_lr: 5e-4  # Lower from 1e-3

# Add gradient clipping
  grad_clip: 5.0  # Lower from 10.0

# Increase batch size for stability
  batch_size: 128
```

---

### Problem: No Learning Progress (Flat Reward)

**Symptoms:**
- Reward stays constant for >50k steps
- Agents don't move or move randomly
- Task completion rate = 0%

**Solutions:**

1. **Check reward scaling:**
   ```yaml
   rewards:
     task_completion: 10.0  # Increase if too small
     time_step_penalty: -0.01  # Make sure penalty isn't too large
   ```

2. **Verify environment:**
   ```bash
   # Test environment manually
   python -c "from src.environments.warehouse_env import WarehouseEnv; \
              import yaml; \
              config = yaml.safe_load(open('configs/default_config.yaml')); \
              env = WarehouseEnv(config); \
              obs, _ = env.reset(); \
              print('Obs shape:', len(obs), 'x', len(obs[0]))"
   ```

3. **Reduce complexity temporarily:**
   ```yaml
   environment:
     num_robots: 2  # Start simple
     task_arrival_rate: 0.1
   ```

---

### Problem: High Collision Rate

**Symptoms:**
- Robots constantly colliding
- Negative rewards dominate
- Training unstable

**Solutions:**
```yaml
# Increase collision penalty
rewards:
  collision: -10.0  # Increase from -5.0

# Add safety margin
environment:
  robot_radius: 0.6  # Increase from 0.5

# Improve LiDAR
  lidar_range: 25.0  # Increase from 20.0
```

---

### Problem: Out of Memory (OOM)

**Symptoms:**
- CUDA out of memory error
- System freezes
- Training crashes after X steps

**Solutions:**
```yaml
# Reduce batch size
training:
  batch_size: 32  # Lower from 64

# Reduce buffer size
  buffer_size: 50000  # Lower from 100000

# Reduce episode length
environment:
  max_episode_steps: 500  # Lower from 1000
```

Or use CPU if GPU memory is insufficient:
```bash
python scripts/train.py --device cpu
```

---

### Problem: Slow Training Speed

**Symptoms:**
- < 100 steps/second on GPU
- < 10 steps/second on CPU

**Solutions:**

1. **Profile bottlenecks:**
   ```bash
   python -m cProfile -o profile.stats scripts/train.py --config configs/small_warehouse.yaml
   python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
   ```

2. **Optimize config:**
   ```yaml
   environment:
     render: false  # Disable rendering
     max_episode_steps: 500  # Shorter episodes
   ```

3. **Increase batch size (if memory allows):**
   ```yaml
   training:
     batch_size: 128  # Larger batches = fewer updates
   ```

---

## Advanced Training Techniques

### 1. Curriculum Learning

Progressively increase difficulty for better convergence:

```yaml
curriculum:
  enabled: true
  stages:
    stage1:
      num_robots: 2
      task_arrival_rate: 0.1
      duration: 100000
    stage2:
      num_robots: 5
      task_arrival_rate: 0.5
      duration: 200000
    stage3:
      num_robots: 10
      task_arrival_rate: 1.0
      duration: 200000
```

**Benefits:**
- Faster initial learning
- Better final performance
- More stable training

---

### 2. Hyperparameter Tuning with Grid Search

```bash
# Create tuning script
for lr in 1e-4 3e-4 1e-3; do
  for bs in 32 64 128; do
    python scripts/train.py \
      --config configs/default_config.yaml \
      --actor-lr $lr \
      --batch-size $bs \
      --seed 42 \
      --output-dir "results/tune_lr${lr}_bs${bs}"
  done
done

# Compare results
python scripts/benchmark.py --configs results/tune_*/config.yaml
```

---

### 3. Transfer Learning

Use pretrained model as starting point:

```bash
# Train on small warehouse
python scripts/train.py --config configs/small_warehouse.yaml

# Fine-tune on large warehouse
python scripts/train.py \
  --config configs/large_warehouse.yaml \
  --checkpoint results/checkpoints/small_warehouse_best.pt \
  --actor-lr 1e-4  # Lower LR for fine-tuning
```

---

### 4. Multi-Seed Training for Robustness

```bash
# Train with multiple random seeds
for seed in 42 123 456 789 1024; do
  python scripts/train.py \
    --config configs/default_config.yaml \
    --seed $seed \
    --output-dir "results/seed_${seed}" &
done
wait

# Aggregate results
python scripts/benchmark.py \
  --configs results/seed_*/config.yaml \
  --names "Seed 42" "Seed 123" "Seed 456" "Seed 789" "Seed 1024"
```

---

### 5. Distributed Training (Advanced)

For training on multiple GPUs:

```bash
# Train on 2 GPUs (if available)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/large_warehouse.yaml --device cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/large_warehouse.yaml --device cuda --seed 123 &
wait
```

---

## Performance Benchmarks

### Expected Training Metrics

| Stage | Episodes | Mean Reward | Success Rate | Collision Rate |
|-------|----------|-------------|--------------|----------------|
| **Early** | 0-50k | -20 to 0 | 0-10% | 20-40% |
| **Mid** | 50k-200k | 0 to 50 | 10-50% | 10-20% |
| **Late** | 200k-500k | 50 to 150 | 50-80% | 5-10% |
| **Converged** | 500k+ | 150+ | 80-95% | < 5% |

### Hardware Performance

| Hardware | Steps/Sec | Episodes/Hour | Time to 500k |
|----------|-----------|---------------|--------------|
| **CPU (8 cores)** | 50-100 | 60-120 | 24-48 hrs |
| **GPU (RTX 3060)** | 500-800 | 600-900 | 4-8 hrs |
| **GPU (RTX 3080)** | 1000-1500 | 1200-1800 | 2-4 hrs |
| **GPU (A100)** | 2000-3000 | 2400-3600 | 1-2 hrs |

---

## Best Practices Checklist

Before starting training:

- [ ] Verify environment setup: `python -c "import torch; import pybullet; print('OK')"`
- [ ] Test config loads: `python -c "import yaml; yaml.safe_load(open('configs/default_config.yaml'))"`
- [ ] Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Set random seed for reproducibility
- [ ] Create output directory: `mkdir -p results/checkpoints results/logs`
- [ ] Enable checkpointing in config
- [ ] Monitor with TensorBoard: `tensorboard --logdir results/logs`

During training:

- [ ] Monitor loss curves (should decrease)
- [ ] Check reward trends (should increase)
- [ ] Watch for NaN values
- [ ] Verify checkpoint saves every N steps
- [ ] Monitor GPU/CPU utilization
- [ ] Check for memory leaks

After training:

- [ ] Evaluate on multiple episodes (20+)
- [ ] Visualize learned behaviors
- [ ] Compare with baseline/random policy
- [ ] Save final model and config
- [ ] Document hyperparameters and results
