#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00
#SBATCH --job-name=sir_chunked_rl_dqc
#SBATCH --output=logs/%x-%j.out
#SBATCH --array=1-3
#SBATCH --requeue

# Chunked RL with DQC hyperparameters:
# - batch_size: 256 -> 4096
# - hidden_dims: 512 -> 1024
# - discount: 0.99 -> 0.995
# - actor_layer_norm: False -> True

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate qc310

# Set up paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl

# Change to QC directory
cd /iris/u/ankile/self-improving-robots-workspace/qc

# Run training
python main.py \
    --wandb_project=qc-comparison-1 \
    --run_group=sir_chunked_rl_dqc \
    --seed=${SLURM_ARRAY_TASK_ID} \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=sir-square-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --eval_interval=100000 \
    --discount=0.995 \
    --agent.batch_size=4096 \
    --agent.actor_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.value_hidden_dims="(1024, 1024, 1024, 1024)" \
    --agent.actor_layer_norm=True
