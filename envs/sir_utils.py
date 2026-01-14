"""Utilities for loading self-improving-robots (SIR) datasets.

This module provides functions to load datasets converted from the
self-improving-robots LeRobot format to HDF5.
"""

import os
from pathlib import Path

import h5py
import numpy as np

from utils.datasets import Dataset


# Observation keys for SIR datasets (just 'state' which contains full observation)
SIR_OBS_KEYS = ['state']


def is_sir_env(env_name: str) -> bool:
    """Check if environment name is a SIR dataset."""
    return env_name.startswith("sir-")


def _get_dataset_path(env_name: str) -> str:
    """Get the HDF5 file path for a SIR dataset.

    Args:
        env_name: Environment name like "sir-square-low_dim"

    Returns:
        Path to the HDF5 file
    """
    # Parse env_name: sir-<task>-<type>
    # e.g., "sir-square-low_dim" -> task="square", type="low_dim"
    parts = env_name.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid SIR env_name: {env_name}. "
            f"Expected format: sir-<task>-<type> (e.g., sir-square-low_dim)"
        )

    task = parts[1]
    # Join remaining parts for type (in case of underscores)
    data_type = "-".join(parts[2:])

    # Build path: ~/.robomimic/sir_<task>/combined/<type>.hdf5
    base_path = os.path.expanduser(f"~/.robomimic/sir_{task}/combined/{data_type}.hdf5")

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"SIR dataset not found at {base_path}. "
            f"Run convert_sir_to_qc.py first to create the dataset."
        )

    return base_path


def get_dataset(env_name: str) -> Dataset:
    """Load a SIR dataset from HDF5.

    Args:
        env_name: Environment name like "sir-square-low_dim"

    Returns:
        Dataset object ready for training
    """
    dataset_path = _get_dataset_path(env_name)

    print(f"Loading SIR dataset from {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        # Sort by demo index
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]))

        num_timesteps = 0
        for demo in demos:
            num_timesteps += f[f'data/{demo}/actions'].shape[0]

        print(f"Dataset has {len(demos)} demos, {num_timesteps:,} timesteps")

        # Collect all data
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        masks = []

        for demo in demos:
            demo_grp = f[f'data/{demo}']

            # Load observations (stored under obs/state)
            obs = np.array(demo_grp['obs/state']).astype(np.float32)
            next_obs = np.array(demo_grp['next_obs/state']).astype(np.float32)
            acts = np.array(demo_grp['actions']).astype(np.float32)
            rews = np.array(demo_grp['rewards']).astype(np.float32)
            dones = np.array(demo_grp['dones']).astype(np.float32)

            observations.append(obs)
            actions.append(acts)
            next_observations.append(next_obs)
            rewards.append(rews)
            terminals.append(dones)
            masks.append(1.0 - dones)

    # Concatenate all demos
    dataset = Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
    )

    print(f"Loaded dataset with {dataset.size:,} transitions")
    return dataset
