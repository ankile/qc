# SIR BC vs RL + Chunking Ablation

**Date:** 2026-01-13
**Git Hash:** `95be7054d608090cb51b17eee5e2583f77e45b00`
**Research Log:** [RESEARCH_LOG_SIR_QC_COMPAT.md](../../RESEARCH_LOG_SIR_QC_COMPAT.md)

## Experiment Description

Comparing BC vs RL (Q-based action selection) and chunked vs unchunked policies on SIR data after fixing observation ordering and reward transform bugs.

## Conditions

| Script | Chunk Size | Num Samples | W&B Run Group |
|--------|-----------|-------------|---------------|
| `sir_unchunked_bc.sh` | 1 | 1 | `sir_unchunked_bc` |
| `sir_unchunked_rl.sh` | 1 | 32 | `sir_unchunked_rl` |
| `sir_chunked_bc.sh` | 5 | 1 | `sir_chunked_bc` |
| `sir_chunked_rl.sh` | 5 | 32 | `sir_chunked_rl` |

- **BC (num_samples=1):** Pure behavior cloning, takes the single action from the policy
- **RL (num_samples=32):** Samples 32 action chunks, selects best according to Q-value

## Common Settings

- `env_name=sir-square-low_dim`
- `sparse=True`
- `eval_interval=100000`
- 3 seeds per condition (SLURM array 1-3)

## Launch Commands

```bash
sbatch sir_unchunked_bc.sh
sbatch sir_unchunked_rl.sh
sbatch sir_chunked_bc.sh
sbatch sir_chunked_rl.sh
```

## Results

<!-- TODO: Add results table after runs complete -->
