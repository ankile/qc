# Research Log: SIR Data Compatibility with QC Framework

**Date:** 2026-01-13

## Problem Statement

Training QC (Q-Chunking) on SIR-converted data showed 0% success rate while the same algorithm on RoboMimic data achieved ~20%. Goal: identify and fix compatibility issues.

**Initial Hypothesis:** QC data uses more observations and staged rewards, while SIR doesn't.

**Finding:** Both datasets have identical observation dimensions (23D) and use sparse binary rewards. The issues were more subtle.

<!-- TODO: Add wandb screenshot showing 0% SIR success vs ~20% RoboMimic success -->

---

## Debugging Process

### Step 1: Rule Out Obvious Differences

Verified that both datasets have:
- Same observation dimensionality (23D)
- Same reward structure (sparse 0/1)
- Same action dimensionality (7D)

### Step 2: Compare Data Distributions

Created `analyze_datasets.py` to load and compare both datasets interactively.

<!-- TODO: Add histogram comparing action distributions (QC vs SIR) -->

**Action Distribution Findings:**

| Metric | RoboMimic | SIR |
|--------|-----------|-----|
| Position action std | ~0.26 | ~0.19 (25% narrower) |
| Action range | [-1, 1] | [-0.74, 0.83] |

Initially suspected action range was the issue, but user confirmed SIR data works with other policies (80% success), so this wasn't the root cause.

### Step 3: Deep Dive into Observation Structure

Compared per-dimension statistics and found the critical issue:

<!-- TODO: Add bar chart showing z-coordinate means for obj[0:3] and obj[7:10] for both datasets (before fix) -->

**Object Observation Z-Coordinate Comparison (Before Fix):**

| Dimension | Expected Content | RoboMimic z-mean | SIR z-mean |
|-----------|------------------|------------------|------------|
| obj[0:3] | abs_pos (high z ~0.88) | 0.885 | 0.05 |
| obj[7:10] | rel_pos (low z ~0.04) | 0.034 | 0.87 |

The values were **swapped** - clear evidence of different observation ordering.

---

## Finding 1: Observation Ordering (CRITICAL FIX)

Robosuite changed object observation ordering between versions:

| Version | Order |
|---------|-------|
| 1.4.x (RoboMimic) | `[abs_pos, abs_quat, rel_pos, rel_quat]` |
| 1.5.x (SIR) | `[rel_pos, rel_quat, abs_pos, abs_quat]` |

**Fix:** Reorder in `convert_sir_to_qc.py`:
```python
obs_env_reordered = np.concatenate([
    obs_env[:, 7:10],   # abs_pos -> dims 0:3
    obs_env[:, 10:14],  # abs_quat -> dims 3:7
    obs_env[:, 0:3],    # rel_pos -> dims 7:10
    obs_env[:, 3:7],    # rel_quat -> dims 10:14
], axis=1)
```

<!-- TODO: Add bar chart showing z-coordinate means AFTER fix (should match) -->

**After Fix:**
| Dimension | RoboMimic z-mean | SIR z-mean |
|-----------|------------------|------------|
| obj[0:3] | 0.885 | 0.882 |
| obj[7:10] | 0.034 | 0.038 |

---

## Finding 2: Reward Transform (CRITICAL FIX)

After fixing observations, training metrics (Q-values, grad norms, flow loss) still differed significantly.

<!-- TODO: Add wandb screenshot showing Q-value divergence between RoboMimic and SIR (before reward fix) -->

**Debugging:** Traced through reward processing in `main.py` and found QC applies a penalty transform for RoboMimic envs:

```python
if is_robomimic_env(FLAGS.env_name):
    penalty_rewards = ds["rewards"] - 1.0  # 0→-1, 1→0
```

Then with `--sparse=True`:
```python
sparse_rewards = (ds["rewards"] != 0.0) * -1.0
```

**Combined effect:**
- Original: `[0, 0, 0, ..., 1, 1, 1]`
- After penalty: `[-1, -1, -1, ..., 0, 0, 0]`
- After sparse: `[-1, -1, -1, ..., 0, 0, 0]` (non-terminal=-1, terminal=0)

SIR data was missing the penalty transform:
- Original: `[0, 0, 0, ..., 0, 0, 1]`
- After sparse only: `[0, 0, 0, ..., 0, 0, -1]` (non-terminal=0, terminal=-1)

**Result:** Completely opposite reward structures!

| Step Type | RoboMimic (correct) | SIR (broken) |
|-----------|---------------------|--------------|
| Non-terminal | -1 | 0 |
| Terminal success | 0 | -1 |

**Fix:** Added `is_sir_env()` check in `main.py`:
```python
if is_robomimic_env(FLAGS.env_name) or is_sir_env(FLAGS.env_name):
    penalty_rewards = ds["rewards"] - 1.0
```

<!-- TODO: Add wandb screenshot showing Q-values NOW matching after reward fix -->

---

## Other Differences (Not Fixed, Likely Minor)

| Metric | RoboMimic | SIR |
|--------|-----------|-----|
| Demos | 300 | 500 |
| Transitions | 81k | 107k |
| Episode length mean | 269 | 214 |
| Terminal reward steps | 5 per episode | 1 per episode |

The terminal reward difference (5 vs 1 steps with reward=1) doesn't affect Q-values since those states have `done=True` (no bootstrapping). The `-1` rewards accumulate through the ~200 non-terminal steps.

---

## Experiments Launched

Four experiment scripts comparing BC vs RL and chunking vs no chunking:

| Experiment | Chunk Size | Num Samples | Run Group |
|------------|-----------|-------------|-----------|
| Unchunked BC | 1 | 1 | `sir_unchunked_bc` |
| Unchunked RL | 1 | 32 | `sir_unchunked_rl` |
| Chunked BC | 5 | 1 | `sir_chunked_bc` |
| Chunked RL | 5 | 32 | `sir_chunked_rl` |

Each with 3 seeds. Scripts in `scripts/sir_*.sh`.

<!-- TODO: Add results table with success rates -->

<!-- TODO: Add learning curves (success rate vs training steps) for all 4 conditions -->

---

## Files Modified

- `convert_sir_to_qc.py` - Added observation reordering
- `main.py` - Added SIR to penalty reward transform
- `scripts/sir_*.sh` - Created 4 experiment scripts

---

## Summary

Two critical bugs prevented SIR data from working with QC:

1. **Observation ordering:** Robosuite 1.5.x swapped relative/absolute object positions compared to 1.4.x
2. **Reward transform:** SIR data wasn't getting the penalty transform that converts sparse rewards to per-step negative rewards

Both fixes were required. After applying them, training metrics (Q-values) matched between datasets.

---

## Next Steps

- [ ] Analyze overnight run results
- [ ] Add plots to placeholders above
- [ ] Compare BC vs RL and chunked vs unchunked results
