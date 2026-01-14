# SIR with DQC Hyperparameters

**Date:** 2026-01-14
**Git Hash:** `95be7054d608090cb51b17eee5e2583f77e45b00`
**Research Log:** [RESEARCH_LOG_SIR_QC_COMPAT.md](../../RESEARCH_LOG_SIR_QC_COMPAT.md)

## Motivation

Previous experiments (2026-01-13) showed:
- Chunking improves performance over no chunking (expected)
- RL (best-of-32) doesn't improve much over BC, only slightly faster execution

Hypothesis: QC's default hyperparameters may be limiting Q-function learning. DQC paper (arxiv:2512.10926) uses significantly different hyperparameters that improved performance.

## Hyperparameter Changes (QC -> DQC)

| Parameter | QC Default | DQC | Rationale |
|-----------|-----------|-----|-----------|
| `batch_size` | 256 | 4096 | More stable Q-learning |
| `actor_hidden_dims` | (512,512,512,512) | (1024,1024,1024,1024) | More capacity |
| `value_hidden_dims` | (512,512,512,512) | (1024,1024,1024,1024) | More capacity |
| `discount` | 0.99 | 0.995 | Better long-horizon credit (~200 vs ~100 effective steps) |
| `actor_layer_norm` | False | True | Training stability |

## Conditions

| Script | Num Samples | W&B Run Group |
|--------|-------------|---------------|
| `sir_chunked_bc_dqc.sh` | 1 | `sir_chunked_bc_dqc` |
| `sir_chunked_rl_dqc.sh` | 32 | `sir_chunked_rl_dqc` |
| `sir_chunked_rl_dqc_n128.sh` | 128 | `sir_chunked_rl_dqc_n128` |

Comparing against previous runs:
- `sir_chunked_bc` (old hyperparams, BC)
- `sir_chunked_rl` (old hyperparams, RL)

## Common Settings

- `env_name=sir-square-low_dim`
- `sparse=True`
- `horizon_length=5`
- `eval_interval=100000`
- 3 seeds per condition (SLURM array 1-3)

## Launch Commands

```bash
sbatch sir_chunked_bc_dqc.sh
sbatch sir_chunked_rl_dqc.sh
sbatch sir_chunked_rl_dqc_n128.sh
```

## Results

<!-- TODO: Add results comparing old vs new hyperparams for BC and RL -->
