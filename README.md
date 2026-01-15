<div align="center">

# [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)

## [[website](https://colinqiyangli.github.io/qc/)]      [[pdf](https://arxiv.org/pdf/2507.07969)]

</div>

<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="teaser figure" src="./assets/teaser.png" width="48%">
  </a>
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="aggregated results" src="./assets/agg.png" width="48%">
  </a>
</p>


## Overview
Q-chunking runs RL on a *temporally extended action (action chunking) space* with an expressive behavior constraint to leverage prior data for improved exploration and online sample efficiency.

## Installation
```bash
pip install -r requirements.txt
```


## Datasets
For robomimic, we assume the datasets are located at `~/.robomimic/lift/mh/low_dim_v15.hdf5`, `~/.robomimic/can/mh/low_dim_v15.hdf5`, and `~/.robomimic/square/mh/low_dim_v15.hdf5`. The datasets can be downloaded from https://robomimic.github.io/docs/datasets/robomimic_v0.1.html (under Method 2: Using Direct Download Links - Multi-Human (MH)).

For cube-quadruple, we use the 100M-size offline dataset. It can be downloaded from https://github.com/seohongpark/horizon-reduction via
```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```
and include this flag in the command line `--ogbench_dataset_dir=[realpath/to/your/cube-quadruple-play-100m-v0/]` to make sure it is using the 100M-size dataset.

## Reproducing paper results

We include the example command for all the methods we evaluate in our paper below. For `scene` and `puzzle-3x3` domains, use `--sparse=True`. We also release our plot data at [plot_data/README.md](plot_data/README.md).

```bash
# QC
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# BFN-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# BFN
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# QC-FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# FQL-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1 

# RLPD-AC
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# QC-RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.bc_alpha=0.01
```

```
@inproceedings{
li2025reinforcement,
title={Reinforcement Learning with Action Chunking},
author={Qiyang Li and Zhiyuan Zhou and Sergey Levine},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=XUks1Y96NR}
}
```

---

## SIR (Self-Improving Robots) Data Experiments

This fork adds support for training on data from the [self-improving-robots](https://github.com/ankile/self-improving-robots) project.

### Environment Setup (micromamba)

```bash
# Install micromamba if not already installed
# See: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

# Create environment with Python 3.10
micromamba create -n qc310 python=3.10 -y

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate qc310

# Install dependencies
pip install -r requirements.txt

# Install lerobot for dataset loading (required for SIR data conversion)
pip install lerobot

# Install robomimic for environment (required for evaluation)
pip install robomimic

# Set up MuJoCo paths (add to your .bashrc or run before training)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl
```

### SIR Dataset Conversion

Convert LeRobot datasets from HuggingFace to QC-compatible HDF5 format:

```bash
# Convert all data (teleop + DAgger rounds 1-8)
python convert_sir_to_qc.py --output ~/.robomimic/sir_square/combined/low_dim.hdf5

# Convert human-only data (for BC baselines matching SIR repo)
# Filters for: source=HUMAN AND success=SUCCESS
python convert_sir_to_qc.py --human_only --output ~/.robomimic/sir_square/human_only/low_dim.hdf5
```

**Dataset sizes:**
- Combined: ~500 episodes, ~107k transitions
- Human-only: 288 episodes, 28,416 transitions

### SIR Experiment Launch Scripts

All experiment scripts are organized in dated folders under `scripts/`:

#### 2026-01-13: BC vs RL + Chunking Ablation
Location: `scripts/2026-01-13_sir_bc_rl_chunking/`

| Script | Chunk Size | Num Samples | Description |
|--------|-----------|-------------|-------------|
| `sir_unchunked_bc.sh` | 1 | 1 | Unchunked BC baseline |
| `sir_unchunked_rl.sh` | 1 | 32 | Unchunked RL (best-of-32) |
| `sir_chunked_bc.sh` | 5 | 1 | Chunked BC |
| `sir_chunked_rl.sh` | 5 | 32 | Chunked RL (best-of-32) |

#### 2026-01-14: DQC Hyperparameters + Human-Only Baselines
Location: `scripts/2026-01-14_sir_dqc_hyperparams/`

**DQC-inspired hyperparameter changes:**
| Parameter | QC Default | DQC |
|-----------|-----------|-----|
| `batch_size` | 256 | 4096 |
| `hidden_dims` | 512 | 1024 |
| `discount` | 0.99 | 0.995 |
| `actor_layer_norm` | False | True |

| Script | Description |
|--------|-------------|
| `sir_chunked_bc_dqc.sh` | Chunked BC with DQC hyperparams |
| `sir_chunked_rl_dqc.sh` | Chunked RL (n=32) with DQC hyperparams |
| `sir_chunked_rl_dqc_n128.sh` | Chunked RL (n=128) with DQC hyperparams |
| `sir_chunked_rl_online.sh` | Chunked RL + online finetuning (1M offline + 1M online) |
| `sir_human_only_unchunked_bc.sh` | Unchunked BC on human-only data |
| `sir_human_only_chunked_bc.sh` | Chunked BC on human-only data |

### Running SIR Experiments

```bash
# Navigate to scripts folder
cd scripts/2026-01-14_sir_dqc_hyperparams/

# Launch individual jobs (SLURM)
sbatch sir_chunked_bc_dqc.sh

# Or run directly (single seed)
MUJOCO_GL=egl python main.py \
    --wandb_project=qc-comparison-1 \
    --run_group=sir_chunked_rl \
    --seed=1 \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=sir-square-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --eval_interval=100000
```

**Environment names:**
- `sir-square-low_dim` - All data (combined)
- `sir-square-human_only-low_dim` - Human-only data (for BC baselines)

---

## Acknowledgments
This codebase is built on top of [FQL](https://github.com/seohongpark/fql). The two rlpd_* folders are directly taken from [RLPD](https://github.com/ikostrikov/rlpd).
