# Algorithms Directory

This directory contains the core reinforcement learning algorithms used in the CMARCOS system.

---

## Actor Network

The **Actor Network** implements a Gaussian policy network for continuous action spaces. It takes an **observation vector** as input and outputs action distributions.

### Input: Observation Vector

The observation is a **35-dimensional float array** containing all information collected by a single robot about itself and its local environment.

#### Observation Components

| Component | Dimensions | Description |
|-----------|------------|-------------|
| **1. Position** | 2D | Robot's current position in 2D space |
| | | • `Robot_X` |
| | | • `Robot_Y` |
| | | *(Z coordinate not considered)* |
| **2. Velocity** | 2D | Robot's current velocity |
| | | • `Velocity_X` |
| | | • `Velocity_Y` |
| **3. Battery Level** | 1D | Current battery charge |
| | | • Normalized: `battery / 100.0` → `[0, 1]` |
| **4. Current Task** | 6D | Information about assigned task |
| | | • `Pickup_X` - Pickup location X |
| | | • `Pickup_Y` - Pickup location Y |
| | | • `Priority` - Normalized: `/10.0` |
| | | • `Delivery_X` - Delivery location X |
| | | • `Delivery_Y` - Delivery location Y |
| | | • `Deadline` - Normalized: `/500.0` |
| **5. Nearest Pending Task** | 2D | Location of closest unassigned task |
| | | • `Pickup_X` |
| | | • `Pickup_Y` |
| **6. Nearby Robots** | 6D | Relative positions of 3 nearest robots |
| | | • `Robot1_X`, `Robot1_Y` |
| | | • `Robot2_X`, `Robot2_Y` |
| | | • `Robot3_X`, `Robot3_Y` |
| **7. LiDAR Readings** | 16D | Distance measurements in 16 directions |
| | | • Beam angles: `[0°, 22.5°, 45°, ..., 337.5°]` |
| | | • 16 evenly-spaced readings covering 360° |

**Total: 35 dimensions** (2 + 2 + 1 + 6 + 2 + 6 + 16)


---

### Output: Action Space

Now let's talk about the outputs:

Each actor network will output **linear velocity** (forward/backward speed) and **angular velocity** (rotation speed).

The actor network's job is to produce commands that control the robot's movement in a smooth, continuous way by specifying how fast to move forward and how fast to turn, based on its local observation input.

---

### Network Heads

Now let's discuss the two heads we have here:

#### 1. Mean Head
The normal Mean head gives the 2 outputs that we discussed above.

**Example mean output:** `[0.8, -0.3]`

#### 2. Log-Std Head
The Log-Std head basically says **"How sure I am"**.

**Example log-std output:** `[-1.5, -2.0]`

We take exponential values of these since they are standard deviations and they shouldn't be negative:

```
exp([-1.5, -2.0]) = [0.223, 0.135]
```

This means: **"Add ±0.223 randomness to forward, ±0.135 to turning"**

---

### Action Sampling

Then the magic happens:

```
action = mean + std × random_noise
```

**Example:**
```
action[0] = 0.8 + 0.223 × 0.5 = 0.912
action[1] = -0.3 + 0.135 × (-1.2) = -0.462
```

These are the velocity results for the robot.
