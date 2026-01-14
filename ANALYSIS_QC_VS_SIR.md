# Analysis: QC/RoboMimic vs SIR Data Comparison

Date: 2026-01-13

## Summary

This analysis compares the QC/RoboMimic dataset (`square-mh-low_dim`) with the converted SIR dataset (`sir-square-low_dim`) for training on the NutAssemblySquare task.

**Key Findings:**
1. Both datasets have identical observation dimensions (23D) and sparse binary rewards (0/1)
2. **CRITICAL: Action ranges are significantly different** - QC/RoboMimic uses full [-1, 1] range while SIR uses narrower ranges (e.g., [-0.73, 0.82])
3. This action range mismatch is likely causing the 0% vs 20% success rate difference

---

## 1. Observation Structure

### Both datasets use 23D observations with the same structure:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| `robot0_eef_pos` / `eef_pos` | 3 | End-effector position (x, y, z) |
| `robot0_eef_quat` / `eef_quat` | 4 | End-effector orientation (quaternion) |
| `robot0_gripper_qpos` / `gripper_qpos` | 2 | Gripper joint positions (left, right) |
| `object` / `environment_state` | 14 | Nut state (relative + absolute pose) |
| **Total** | **23** | |

### Object/Environment State (14D) breakdown:
- `nut_to_eef_pos` (3D): Relative position of nut to end-effector
- `nut_to_eef_quat` (4D): Relative orientation of nut to end-effector
- `nut_pos` (3D): Absolute nut position
- `nut_quat` (4D): Absolute nut orientation

### Verification:
```
QC obs structure: robot0_eef_pos(3) + robot0_eef_quat(4) + robot0_gripper_qpos(2) + object(14) = 23D
SIR obs structure: observation.state(9) + observation.environment_state(14) = 23D
```

---

## 2. Reward Structure

### Both datasets use sparse binary rewards:

| Dataset | Unique Values | Interpretation |
|---------|---------------|----------------|
| QC/RoboMimic | `[0.0, 1.0]` | 0 = not done, 1 = success |
| SIR | `[0.0, 1.0]` | 0 = not done, 1 = success |

**Important:** Neither dataset uses staged/shaped rewards in the stored data. The staged rewards documented in robosuite's `NutAssembly.staged_rewards()` are only used when `reward_shaping=True` is set at environment runtime, but the demonstration datasets store sparse rewards only.

### QC Training with `--sparse=True`:
When `--sparse=True` is set in training, the rewards are transformed:
```python
sparse_rewards = (ds["rewards"] != 0.0) * -1.0
```
This converts: `0 -> 0, 1 -> -1` (goal-conditioned negative reward formulation)

---

## 3. CRITICAL: Action Range Mismatch

**This is the most likely cause of the 0% success rate with SIR data.**

### Per-dimension action statistics:

| Dim | Semantic | QC Min | QC Max | SIR Min | SIR Max |
|-----|----------|--------|--------|---------|---------|
| 0 | delta_eef_pos_x | -1.0 | 1.0 | -0.74 | 0.83 |
| 1 | delta_eef_pos_y | -1.0 | 1.0 | -0.68 | 0.75 |
| 2 | delta_eef_pos_z | -1.0 | 1.0 | -0.83 | 0.62 |
| 3 | delta_eef_rot_x | -0.45 | 0.21 | -0.25 | 0.23 |
| 4 | delta_eef_rot_y | -1.0 | 1.0 | -0.20 | 0.25 |
| 5 | delta_eef_rot_z | -1.0 | 1.0 | -0.33 | 0.33 |
| 6 | gripper_action | -1.0 | 1.0 | -1.0 | 1.0 |

### Key observations:
- **Position actions (0-2):** QC spans full [-1, 1], SIR is ~25% narrower
- **Rotation actions (4-5):** QC spans full [-1, 1], SIR is ~65-80% narrower
- **Gripper (6):** Both span [-1, 1] (identical)

### Why this matters:
1. The policy learns from SIR data where actions are in a narrower range
2. During evaluation, the environment expects actions in [-1, 1] range
3. The policy outputs actions similar to training data (narrow range)
4. This results in smaller movements than intended, leading to failures

### Potential fixes:
1. **Normalize SIR actions to [-1, 1]** during conversion or loading
2. **Re-collect data** with action scaling that matches RoboMimic
3. **Scale actions at inference time** to expand the narrow range to [-1, 1]

### Root cause:
The SIR codebase uses a different controller configuration (likely with different `input_max`/`input_min` settings or action scaling) compared to how RoboMimic collected demonstrations.

---

## 4. Robosuite NutAssemblySquare Staged Rewards (Reference)

From `robosuite/environments/manipulation/nut_assembly.py:290-373`:

The staged rewards are computed at **runtime** when `reward_shaping=True`:

| Stage | Reward Range | Condition |
|-------|--------------|-----------|
| **Reaching** | [0, 0.1] | Proportional to distance between gripper and closest nut |
| **Grasping** | {0, 0.35} | Nonzero if gripper is grasping a nut |
| **Lifting** | {0, [0.35, 0.5]} | Proportional to lifting height (requires grasp) |
| **Hovering** | {0, [0.5, 0.7]} | Proportional to distance from nut to peg (requires lift) |
| **Success** | 1.0 | Nut placed on correct peg |

**Note:** These are NOT in the training data. The demonstration datasets use sparse rewards only.

---

## 5. Data Sources

### QC/RoboMimic (`square-mh-low_dim`):
- **File:** `~/.robomimic/square/mh/low_dim_v15.hdf5`
- **Source:** Official RoboMimic multi-human (mh) demonstrations
- **Format:** Separate observation keys in HDF5

### SIR (`sir-square-low_dim`):
- **File:** `~/.robomimic/sir_square/combined/low_dim.hdf5`
- **Source:** Converted from LeRobot datasets:
  - `ankile/square-teleop-v5` (teleoperation demos)
  - `ankile/square-dagger-state-seq2-r1` through `r8` (DAgger iterations)
- **Format:** Combined state vector in HDF5

---

## 6. Training Configuration Comparison

Both training scripts use identical hyperparameters:

```bash
# QC/RoboMimic: scripts/qc_robomimic_square.sh
python main.py \
    --run_group=qc_robomimic \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=square-mh-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --eval_interval=100000

# SIR: scripts/qc_sir_square.sh
python main.py \
    --run_group=qc_sir \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=sir-square-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --eval_interval=100000
```

---

## 7. Key Files

| Purpose | File |
|---------|------|
| QC data loading | `envs/robomimic_utils.py` |
| SIR data loading | `envs/sir_utils.py` |
| Data conversion | `convert_sir_to_qc.py` |
| Environment dispatch | `envs/env_utils.py` |
| Robosuite task | `robosuite/environments/manipulation/nut_assembly.py` |

---

## 8. Dataset Statistics Summary

| Metric | QC/RoboMimic | SIR |
|--------|--------------|-----|
| Total demos | 300 | 500 |
| Total transitions | 80,731 | 106,757 |
| Mean episode length | 269.1 | 213.5 |
| Min episode length | 123 | 102 |
| Max episode length | 1,051 | 591 |
| Demo success rate | ~100% | 99.8% |

---

## 9. Conclusions

1. **Observation dimensions are identical** (23D) - NOT a source of difference
2. **Reward structure is identical** (sparse 0/1) - NOT a source of difference
3. **CRITICAL: Action ranges are different** - LIKELY the cause of 0% success rate
   - QC/RoboMimic: Actions span full [-1, 1] range
   - SIR: Actions span narrower range (e.g., [-0.74, 0.83] for position dims)
   - This causes the learned policy to output smaller actions than needed
4. **Staged rewards are NOT used** - demonstration data uses sparse rewards only
5. **Data quantity:** SIR has more data (500 demos vs 300, 107k vs 81k transitions)

## 10. CRITICAL: Object State Differences

**UPDATE (after deeper analysis):** The object observation semantics appear to be different!

### Object State Z-coordinate Analysis:

| Dataset | Dims [16:19] (pos2) z-mean | Interpretation |
|---------|---------------------------|----------------|
| QC/RoboMimic | 0.051 | Very low - different coordinate frame |
| SIR | 0.830 | Around table height - world frame |

### Key Observations:
1. **QC pos2 z-values ~0.05**: This is way below table height (~0.82), suggesting it might be:
   - Relative to robot base
   - A delta/offset value
   - In a different coordinate frame entirely

2. **SIR pos2 z-values ~0.83**: This matches expected world-frame nut position (on table)

3. **Quaternion patterns are completely different:**
   - QC: dims [12:16] often have x,y ≈ 0 (e.g., [0, 0, 0.916, -0.400])
   - SIR: dims [12:16] have spread values (e.g., [-0.587, 0.807, -0.036, 0.052])

### This suggests:
The `object` key in robomimic HDF5 may NOT be the same as `object-state` from robosuite!
- **robomimic `object`**: Possibly processed/transformed observations
- **robosuite `object-state`**: Raw concatenation of object modality sensors

This coordinate frame mismatch would explain why the policy trained on SIR data fails at evaluation time - the observations don't match what the model learned from.

---

## 11. ROOT CAUSE IDENTIFIED: Object Observation Order is Different!

### Current robosuite (1.5.x) object-state order (used by SIR):
```
[0:3]   = SquareNut_to_robot0_eef_pos  (RELATIVE position, z ≈ 0.05-0.2)
[3:7]   = SquareNut_to_robot0_eef_quat (RELATIVE quaternion)
[7:10]  = SquareNut_pos                (ABSOLUTE position, z ≈ 0.83)
[10:14] = SquareNut_quat               (ABSOLUTE quaternion)
```

### QC/RoboMimic data order (robosuite 1.4.1):
```
[0:3]   = ??? (z ≈ 0.88, appears to be ABSOLUTE position)
[3:7]   = ??? (quaternion)
[7:10]  = ??? (z ≈ 0.05, appears to be relative or table-relative)
[10:14] = ??? (quaternion)
```

### Evidence:
- QC obj[0:3] z-mean: **0.888** (HIGH, like absolute nut position)
- SIR obj[0:3] z-mean: **0.053** (LOW, relative position)
- QC obj[7:10] z-mean: **0.051** (LOW)
- SIR obj[7:10] z-mean: **0.872** (HIGH, absolute nut position)

### Why this causes 0% success:
The policy trained on SIR data expects:
- dims 9-11: relative position (changes as robot moves)
- dims 16-18: absolute position (nut on table ~0.83)

But QC evaluation provides:
- dims 9-11: absolute position (nut at ~0.88)
- dims 16-18: something else (~0.05)

The model receives completely wrong input semantics!

---

## 12. THE FIX

### Option 1: Reorder SIR data during conversion
In `convert_sir_to_qc.py` or `sir_utils.py`, swap the object observation order:
```python
# Current SIR order: [rel_pos(3), rel_quat(4), abs_pos(3), abs_quat(4)]
# Target QC order:   [abs_pos(3), abs_quat(4), rel_pos(3), rel_quat(4)]

env_state = obs[:, 9:23]  # 14D object state
reordered = np.concatenate([
    env_state[:, 7:10],   # abs_pos -> dims 0:3
    env_state[:, 10:14],  # abs_quat -> dims 3:7
    env_state[:, 0:3],    # rel_pos -> dims 7:10
    env_state[:, 3:7],    # rel_quat -> dims 10:14
], axis=-1)
```

### Option 2: Use same robosuite version
Ensure both data collection and evaluation use the exact same robosuite version (1.4.1 for QC compatibility).

### Option 3: Update QC evaluation environment
Modify `robomimic_utils.py` to reorder observations from the environment to match SIR's order.
