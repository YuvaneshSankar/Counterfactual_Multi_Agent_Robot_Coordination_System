# COMA Algorithm Documentation

## Overview

**Counterfactual Multi-Agent (COMA)** is an advanced multi-agent reinforcement learning algorithm that uses centralized training with decentralized execution (CTDE). This document provides comprehensive details about the COMA implementation in the COMAR system, extended to support continuous action spaces.

## Table of Contents

1. [Algorithm Fundamentals](#algorithm-fundamentals)
2. [Continuous Action Extension](#continuous-action-extension)
3. [Architecture](#architecture)
4. [Training Process](#training-process)
5. [Credit Assignment](#credit-assignment)
6. [Implementation Details](#implementation-details)

## Algorithm Fundamentals

### Background

COMA is designed for cooperative multi-agent tasks where agents share a common reward signal. It solves the credit assignment problem—determining how much each agent contributed to the final reward—using counterfactual reasoning.

### Key Innovation: Counterfactual Baseline

The counterfactual baseline for agent $i$ is computed as:

$$Q^{CF}_i(s, u) = Q(s, u^{-i}, u_i^{default})$$

Where:
- $s$ = global state
- $u$ = joint action of all agents
- $u^{-i}$ = actions of all agents except $i$
- $u_i^{default}$ = default action of agent $i$ (e.g., no-op)

The advantage for agent $i$ is then:

$$A_i = Q(s, u) - Q^{CF}_i(s, u)$$

This measures how much agent $i$'s action improved upon the default action, holding others' actions constant.

## Continuous Action Extension

### Gaussian Policy

Traditional COMA uses discrete policies. We extend it to continuous spaces using **Gaussian policies**:

$$\pi_i(a|o_i) = \mathcal{N}(\mu_i(o_i), \sigma_i^2(o_i))$$

Where:
- $\mu_i(o_i)$ = mean action (from neural network)
- $\sigma_i(o_i)$ = action standard deviation (learned)
- $o_i$ = agent $i$'s local observation

### Policy Gradient

The actor network learns using the policy gradient:

$$\nabla_\theta \mathcal{L}_{\text{actor}} = -\mathbb{E}[A_i \nabla_\theta \log \pi_i(a|o_i)]$$

Where the advantage $A_i$ comes from the centralized critic.

### Action Squashing

Actions are squashed to valid bounds using tanh:

$$a_{\text{squashed}} = \text{tanh}(a_{\text{raw}})$$

Log probabilities are adjusted accordingly:

$$\log \pi(a_{\text{squashed}}) = \log \pi(a_{\text{raw}}) - \sum_i \log(1 - a_i^2)$$

## Architecture

### Actor Networks

One actor network per agent:
- **Input**: Local observation $o_i$ (from sensors)
- **Hidden Layers**: [256, 256] with ReLU activation
- **Output**: Mean and log-std of action distribution

```
Input (obs_dim)
    ↓
Dense(256) + ReLU
    ↓
Dense(256) + ReLU
    ↓
[Mean (action_dim), LogStd (action_dim)]
```

### Critic Network

Single centralized critic:
- **Input**: Global state + joint actions
- **Hidden Layers**: [256, 256] with ReLU activation
- **Output**: Q-value for each agent

```
[State + Actions]
    ↓
Dense(256) + ReLU
    ↓
Dense(256) + ReLU
    ↓
Q-values (num_agents)
```

### Target Networks

Soft-target networks with update rate $\tau = 0.005$:

$$\theta'^{\text{new}} = \tau \theta^{\text{online}} + (1-\tau) \theta'^{\text{old}}$$

## Training Process

### Experience Collection

1. Environment returns observations $o = [o_1, ..., o_N]$
2. Each actor samples action: $a_i \sim \pi_i(·|o_i)$
3. Environment executes joint action, returns reward $r$
4. Store transition in replay buffer

### Network Updates

#### Critic Update

Minimize TD loss:

$$\mathcal{L}_{\text{critic}} = \mathbb{E}[(r + \gamma Q'(s', a') - Q(s, a))^2]$$

Where:
- $Q'$ = target critic network
- $a'$ = actions from target actor networks
- $\gamma$ = discount factor (0.99)

#### Actor Update

Maximize advantage using policy gradient:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}[A_i \log \pi_i(a_i|o_i)]$$

Where advantage:

$$A_i = Q(s, u) - Q^{CF}_i(s, u^{-i}, u_i^{\text{default}})$$

### Batch Training

- **Batch Size**: 64
- **Epochs**: 10 per rollout
- **Rollout Length**: 2000 steps
- **Optimizer**: Adam with learning rate 3e-4

## Credit Assignment

### Counterfactual Reasoning

The credit assignment problem is solved by asking: "How much better did agent $i$ perform compared to default?"

Steps:
1. **Compute Q(s, u)**: Value with all agents' actual actions
2. **Compute Q^CF(s, u^-i, u_i^default)**: Value with agent $i$ using default action
3. **Advantage**: Difference between above

### Advantages of This Approach

- **Unbiased**: Each agent's contribution is measured fairly
- **Stable**: Doesn't require explicit cooperation signal
- **Scalable**: Works with any number of agents
- **Continuous Actions**: Naturally extends to Gaussian policies

## Implementation Details

### Replay Buffer

- **Capacity**: 100,000 transitions
- **Sampling**: Uniform random sampling
- **Storage**: Separate buffers for states, observations, actions, rewards

### Gradient Clipping

- **Max Gradient Norm**: 0.5
- **Applied to**: Both actor and critic

### Entropy Regularization (Optional)

Can be enabled for exploration:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}[A_i \log \pi_i(a_i|o_i) + \alpha H(\pi_i)]$$

Where $H(\pi_i)$ is entropy and $\alpha$ is regularization coefficient.

### Training Loop Pseudocode

```python
for step in range(total_steps):
    # Collect experience
    obs = env.reset()
    for t in range(rollout_length):
        actions = [actor_i.sample(obs_i) for i in agents]
        obs_next, reward, done, _ = env.step(actions)
        replay_buffer.add(state, obs, actions, reward, state_next, obs_next, done)
        obs = obs_next

    # Update networks
    for epoch in range(num_epochs):
        batch = replay_buffer.sample(batch_size)

        # Update critic
        critic_loss = compute_critic_loss(batch)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optimizer.step()

        # Update actors
        for i in range(num_agents):
            advantage_i = compute_advantage(batch, i)
            actor_loss_i = -advantage_i * log_prob_i
            actor_optimizer_i.zero_grad()
            actor_loss_i.backward()
            clip_grad_norm_(actor_i.parameters(), max_grad_norm)
            actor_optimizer_i.step()

        # Soft update targets
        soft_update(critic_target, critic)
        for i in agents:
            soft_update(actor_target_i, actor_i)
```

## Hyperparameter Tuning

### Critical Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Actor LR | 3e-4 | [1e-5, 1e-3] | Lower = more stable, slower |
| Critic LR | 3e-4 | [1e-5, 1e-3] | Controls value estimation speed |
| Gamma | 0.99 | [0.95, 0.999] | Higher = longer horizons |
| Tau | 0.005 | [0.001, 0.1] | Lower = more stable targets |
| Batch Size | 64 | [32, 256] | Larger = more stable, slower |

### Tuning Strategy

1. **Start with defaults** from config
2. **If learning unstable**: Reduce actor_lr, increase tau
3. **If learning slow**: Increase batch_size, reduce tau
4. **If overfitting**: Increase entropy_coefficient, reduce buffer_size

## Performance Metrics

### Key Metrics to Monitor

- **Episode Reward**: Total reward per episode
- **Task Success Rate**: Percentage of completed tasks
- **Collision Rate**: Number of collisions per episode
- **Actor Loss**: Should decrease over time
- **Critic Loss**: Should decrease over time

## References

- [COMA: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [CTDE in Multi-Agent RL](https://arxiv.org/abs/1810.08779)
- [Continuous Control with Deep RL](https://arxiv.org/abs/1512.04455)
